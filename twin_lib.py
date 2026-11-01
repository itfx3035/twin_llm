"""Twin-agent discussion framework backed by Ollama.

Two agents answer a question independently, debate round by round until they
agree (or the round budget runs out), and a neutral prompt then synthesises the
consensus answer.

This module is a library: it does not print. Round-by-round output is emitted
through the ``twin_llm`` logger, so callers control presentation with
``logging.basicConfig(level=logging.INFO)``.

Example:
    import logging
    from twin_lib import ask_question

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = ask_question("What makes a good API?", model_name="ministral-3:latest")
    print(result.final_answer, result.agreed, result.rounds_used)
"""

from __future__ import annotations

import functools
import logging
import os
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger("twin_llm")

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


DEFAULT_OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = _env("OLLAMA_MODEL", "ministral-3:latest")
DEFAULT_TEMPERATURE = float(_env("TWIN_TEMPERATURE", "0.7"))
DEFAULT_MAX_ROUNDS = int(_env("TWIN_MAX_ROUNDS", "5"))
DEFAULT_NUM_CTX = int(_env("TWIN_NUM_CTX", "8192"))
DEFAULT_MAX_HISTORY_TURNS = int(_env("TWIN_MAX_HISTORY_TURNS", "20"))
DEFAULT_RETRIES = int(_env("TWIN_RETRIES", "3"))
DEFAULT_TIMEOUT = float(_env("TWIN_TIMEOUT", "120"))
DEFAULT_SEED: int | None = int(_env("TWIN_SEED")) if _env("TWIN_SEED", "") else None

PERSONA_ALPHA = (
    "A careful analyst who favours practical, step-by-step reasoning and checks "
    "assumptions before answering."
)
PERSONA_BETA = (
    "A skeptical reviewer who challenges assumptions, looks for counter-arguments and "
    "edge cases, and only agrees when the other agent's reasoning truly holds."
)
PERSONA_CONSENSUS = "A neutral synthesiser who merges two perspectives into one clear answer."

VERDICT_INSTRUCTION = (
    "End every reply with one final line formatted exactly as `AGREE: YES` or `AGREE: NO`, "
    "stating whether you fundamentally agree with the other agent's conclusion."
)

_VERDICT_RE = re.compile(r"\b(?:agree|verdict|decision)\b\s*[:=]\s*(yes|no)\b", re.IGNORECASE)


class TwinLLMError(RuntimeError):
    """Raised when the LLM cannot be reached or cannot answer."""


def parse_verdict(text: str) -> bool | None:
    """Return True/False for an explicit agreement verdict, or None if unclear.

    Only an explicit ``AGREE: YES|NO`` (or ``VERDICT:``/``DECISION:``) marker counts;
    the last match wins. Natural language such as "no, but I agree" is deliberately
    treated as unclear so the caller can run a low-temperature fallback check.
    """
    matches = _VERDICT_RE.findall(text or "")
    if not matches:
        return None
    return matches[-1].lower() == "yes"


def strip_verdict(text: str) -> str:
    """Drop the trailing machine-readable verdict line from a reply."""
    lines = (text or "").splitlines()
    while lines and _VERDICT_RE.search(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def quote(text: str) -> str:
    """Wrap agent output in delimiters so it is treated as data, not instructions."""
    return '"""' + (text or "").strip() + '"""'


@dataclass
class Turn:
    """One message in an agent's conversation history."""

    role: str
    content: str


class LLM(Protocol):
    """Minimal LLM interface: anything with ``invoke`` works (real client or test stub)."""

    def invoke(self, messages: list[Any]) -> Any:
        """Return a response object with ``content`` and optional ``usage_metadata``."""
        raise NotImplementedError


def to_chat_messages(turns: list[Turn]) -> list[Any]:
    """Convert internal turns to LangChain message objects when available.

    Falls back to ``(role, content)`` tuples so the module works (and can be unit
    tested) without ``langchain_core`` installed.
    """
    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    except ImportError:
        return [(turn.role, turn.content) for turn in turns]

    classes = {ROLE_SYSTEM: SystemMessage, ROLE_USER: HumanMessage, ROLE_ASSISTANT: AIMessage}
    return [classes[turn.role](content=turn.content) for turn in turns]


def message_role(message: Any) -> str:
    """Best-effort role extraction for LangChain messages and ``(role, content)`` tuples."""
    if isinstance(message, tuple):
        return str(message[0]).lower()
    role = getattr(message, "type", None) or getattr(message, "role", None)
    mapping = {"system": ROLE_SYSTEM, "human": ROLE_USER, "ai": ROLE_ASSISTANT}
    return mapping.get(str(role).lower(), str(role).lower())


@dataclass
class AgentReply:
    """A debate turn: the reply text plus the verdict parsed from it."""

    text: str
    agrees: bool | None = None


@dataclass
class LogEntry:
    """A logged debate turn, with the verdict when one was produced."""

    round: int
    agent: str
    text: str
    agrees: bool | None = None

    def __str__(self) -> str:
        """Flat ``[Agent]: text`` form used by the 1.x string log."""
        return f"[{self.agent}]: {self.text}"


def build_llm(
    model_name: str,
    ollama_url: str | None = None,
    temperature: float = 0.7,
    num_ctx: int = DEFAULT_NUM_CTX,
    seed: int | None = None,
    timeout: float | None = DEFAULT_TIMEOUT,
) -> Any:
    """Create a ChatOllama client, with a friendly error when the package is missing."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TwinLLMError(
            "langchain-ollama is not installed. Install it with: pip install langchain-ollama"
        ) from exc

    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "base_url": ollama_url or DEFAULT_OLLAMA_URL,
        "num_ctx": num_ctx,
    }
    if timeout:
        kwargs["request_timeout"] = timeout
    if seed is not None:
        kwargs["seed"] = seed
    return ChatOllama(**kwargs)


class Agent:
    """An individual debater with its own persona, history and usage counters."""

    def __init__(
        self,
        name: str,
        model_name: str = DEFAULT_MODEL,
        ollama_url: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        persona: str = "A thoughtful and knowledgeable assistant.",
        num_ctx: int = DEFAULT_NUM_CTX,
        seed: int | None = DEFAULT_SEED,
        max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS,
        retries: int = DEFAULT_RETRIES,
        timeout: float | None = DEFAULT_TIMEOUT,
        llm: LLM | None = None,
        verdict_llm: LLM | None = None,
    ):
        """Create an agent.

        Args:
            name: Display name, also used in the system prompt.
            model_name: Ollama model name (must be pulled).
            ollama_url: Ollama server URL.
            temperature: Debate temperature, validated to 0.0-2.0.
            persona: System-prompt persona; make it distinct per agent.
            num_ctx: Context window passed to Ollama.
            seed: Generation seed for reproducibility.
            max_history_turns: Turns kept before pruning.
            retries: LLM call attempts with capped exponential backoff.
            timeout: Per-request timeout in seconds.
            llm: Injected client (used by tests); built from the settings when omitted.
            verdict_llm: Optional low-temperature client used for verdicts.

        Raises:
            ValueError: If temperature, num_ctx or max_history_turns are out of range.
            TwinLLMError: If langchain-ollama is not installed.
        """
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {temperature}")
        if num_ctx <= 0:
            raise ValueError(f"num_ctx must be positive, got {num_ctx}")
        if max_history_turns < 1:
            raise ValueError(f"max_history_turns must be >= 1, got {max_history_turns}")

        self.name = name
        self.model_name = model_name
        self.ollama_url = ollama_url or DEFAULT_OLLAMA_URL
        self.temperature = temperature
        self.persona = persona
        self.num_ctx = num_ctx
        self.seed = seed
        self.max_history_turns = max_history_turns
        self.retries = max(1, int(retries))
        self.timeout = timeout

        # Verdicts run at temperature 0 for determinism; injected stubs are used as-is.
        self.llm = llm
        self.verdict_llm = verdict_llm
        if self.llm is None:
            self.llm = build_llm(model_name, self.ollama_url, temperature, num_ctx, seed, timeout)
            if self.verdict_llm is None:
                self.verdict_llm = build_llm(
                    model_name, self.ollama_url, 0.0, num_ctx, seed, timeout
                )

        self.conversation_history: list[Turn] = []
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def system_prompt(self) -> str:
        """Persona plus the verdict output contract."""
        return f"You are {self.name}. {self.persona} {VERDICT_INSTRUCTION}"

    def reset(self) -> None:
        """Forget the current conversation (required between questions)."""
        self.conversation_history = []

    def messages(
        self,
        extra: Turn | None = None,
        fresh: bool = False,
        system_prompt: str | None = None,
    ) -> list[Any]:
        """System message first, then a pruned user/assistant history."""
        history = [] if fresh else self.conversation_history[-self.max_history_turns :]
        turns = [Turn(ROLE_SYSTEM, system_prompt or self.system_prompt)]
        turns.extend(history)
        if extra:
            turns.append(extra)
        return to_chat_messages(turns)

    def _call(self, llm: LLM, messages: list[Any]) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = llm.invoke(messages)
            except Exception as exc:  # noqa: BLE001 - library boundary
                last_error = exc
                if attempt >= self.retries:
                    break
                delay = min(30.0, 1.5 * 2 ** (attempt - 1))
                logger.warning(
                    "%s: LLM call failed (%s); retry %d/%d in %.1fs",
                    self.name,
                    exc,
                    attempt + 1,
                    self.retries,
                    delay,
                )
                time.sleep(delay)
                continue

            self.llm_calls += 1
            usage = getattr(response, "usage_metadata", None) or {}
            if isinstance(usage, dict):
                self.prompt_tokens += int(usage.get("input_tokens") or 0)
                self.completion_tokens += int(usage.get("output_tokens") or 0)
            return str(getattr(response, "content", response))

        hint = f"model {self.model_name!r} on {self.ollama_url}"
        raise TwinLLMError(
            f"{self.name}: LLM call failed after {self.retries} attempts ({hint}). "
            f"Is Ollama running and has the model been pulled? ({last_error})"
        ) from last_error

    def generate(
        self,
        user_prompt: str,
        fresh: bool = False,
        system_prompt: str | None = None,
        deterministic: bool = False,
    ) -> str:
        """Run one completion; ``fresh=True`` does not reuse the debate history."""
        llm = self.verdict_llm or self.llm if deterministic else self.llm
        return self._call(llm, self.messages(Turn(ROLE_USER, user_prompt), fresh, system_prompt))

    def get_initial_response(self, question: str) -> str:
        """First, independent answer to the question."""
        self.reset()
        raw = self.generate(question)
        answer = strip_verdict(raw) or raw
        self.conversation_history.append(Turn(ROLE_ASSISTANT, answer))
        return answer

    def respond_to_other_agent(
        self, other_agent_name: str, other_agent_response: str, original_question: str
    ) -> AgentReply:
        """Reply to the other agent; the verdict comes from the reply itself when possible."""
        user_prompt = (
            f"{other_agent_name} said:\n{quote(other_agent_response)}\n\n"
            f"Do you agree or disagree with their answer about: {original_question}"
        )
        raw = self.generate(user_prompt)
        agrees = parse_verdict(raw)
        if agrees is None:
            agrees = self.check_agreement(other_agent_response, other_agent_name)

        text = strip_verdict(raw) or raw
        self.conversation_history.append(Turn(ROLE_USER, user_prompt))
        self.conversation_history.append(Turn(ROLE_ASSISTANT, text))
        return AgentReply(text=text, agrees=agrees)

    def check_agreement(
        self, other_agent_response: str, other_agent_name: str = "the other agent"
    ) -> bool | None:
        """Fallback low-temperature verdict call, used when a reply has no marker."""
        prompt = (
            f"{other_agent_name} said:\n{quote(other_agent_response)}\n\n"
            "Based on our discussion, do you fundamentally agree with that conclusion? "
            "Answer with only 'AGREE: YES' or 'AGREE: NO'."
        )
        return parse_verdict(self.generate(prompt, fresh=True, deterministic=True))

    def usage(self) -> dict[str, int]:
        """Call and token counters for this agent."""
        return {
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass
class DiscussionResult:
    """Outcome of one discussion. ``str(result)`` is the final answer."""

    question: str
    final_answer: str
    agreed: bool
    rounds_used: int
    entries: list[LogEntry] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def log(self) -> list[str]:
        """Flat string log, compatible with the 1.x ``get_discussion_log()`` output."""
        return [str(entry) for entry in self.entries]

    def __str__(self) -> str:
        """The final answer, so ``print(result)`` behaves like the 1.x API."""
        return self.final_answer

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form of the whole discussion."""
        return {
            "question": self.question,
            "final_answer": self.final_answer,
            "agreed": self.agreed,
            "rounds_used": self.rounds_used,
            "usage": self.usage,
            "log": [
                {
                    "round": entry.round,
                    "agent": entry.agent,
                    "text": entry.text,
                    "agrees": entry.agrees,
                }
                for entry in self.entries
            ],
        }


class TwinAgentDiscussion:
    """Two agents that discuss a question until they agree or max_rounds is reached."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        ollama_url: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        num_ctx: int = DEFAULT_NUM_CTX,
        seed: int | None = DEFAULT_SEED,
        max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS,
        retries: int = DEFAULT_RETRIES,
        timeout: float | None = DEFAULT_TIMEOUT,
        agent1: Agent | None = None,
        agent2: Agent | None = None,
        consensus_agent: Agent | None = None,
        parallel: bool = True,
        concurrent_rounds: bool = False,
    ):
        """Create the discussion.

        Args:
            model_name: Ollama model used by both agents unless per-agent models are given.
            ollama_url: Ollama server URL (defaults to the ``OLLAMA_URL`` env var).
            temperature: Creativity for debate replies (verdicts always run at 0.0).
            max_rounds: Debate rounds before giving up (0 skips the debate).
            num_ctx: Context window passed to Ollama.
            seed: Seed for reproducible runs.
            max_history_turns: History kept per agent (older turns are pruned).
            retries: LLM call attempts with capped exponential backoff.
            timeout: Per-request timeout in seconds.
            agent1: Pre-built first agent (dependency injection; used by the tests).
            agent2: Pre-built second agent.
            consensus_agent: Pre-built neutral synthesiser.
            parallel: Run the two initial answers concurrently.
            concurrent_rounds: Each round, both agents reply to the other agent's
                *previous-round* answer simultaneously instead of turn by turn.

        Raises:
            ValueError: If max_rounds is negative.
            TwinLLMError: If langchain-ollama is not installed.
        """
        if max_rounds < 0:
            raise ValueError(f"max_rounds must be >= 0, got {max_rounds}")

        self.model_name = model_name
        self.ollama_url = ollama_url or DEFAULT_OLLAMA_URL
        self.temperature = temperature
        self.max_rounds = max_rounds
        self.num_ctx = num_ctx
        self.seed = seed
        self.max_history_turns = max_history_turns
        self.retries = max(1, int(retries))
        self.timeout = timeout
        self.parallel = parallel
        self.concurrent_rounds = concurrent_rounds

        common = {
            "ollama_url": self.ollama_url,
            "num_ctx": self.num_ctx,
            "seed": self.seed,
            "max_history_turns": self.max_history_turns,
            "retries": self.retries,
            "timeout": self.timeout,
        }
        self.agent1 = agent1 or Agent(
            name="Agent-Alpha",
            model_name=model_name,
            temperature=temperature,
            persona=PERSONA_ALPHA,
            **common,
        )
        self.agent2 = agent2 or Agent(
            name="Agent-Beta",
            model_name=model_name,
            temperature=temperature,
            persona=PERSONA_BETA,
            **common,
        )
        self.consensus_agent = consensus_agent or Agent(
            name="Consensus-Builder",
            model_name=model_name,
            temperature=min(temperature, 0.3),
            persona=PERSONA_CONSENSUS,
            **common,
        )
        self.last_result: DiscussionResult | None = None

    def _two(self, fn1: Callable[[], Any], fn2: Callable[[], Any]) -> tuple[Any, Any]:
        if self.parallel:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="twin") as pool:
                return pool.submit(fn1).result(), pool.submit(fn2).result()
        return fn1(), fn2()

    def get_answer(self, question: str) -> DiscussionResult:
        """Run the discussion and return a structured result."""
        for agent in (self.agent1, self.agent2, self.consensus_agent):
            agent.reset()

        entries: list[LogEntry] = []
        logger.info("=" * 60)
        logger.info("Question: %s", question)
        logger.info(
            "Agents: %s vs %s | model=%s | max_rounds=%d",
            self.agent1.name,
            self.agent2.name,
            self.model_name,
            self.max_rounds,
        )
        logger.info("=" * 60)

        answer1, answer2 = self._two(
            lambda: self.agent1.get_initial_response(question),
            lambda: self.agent2.get_initial_response(question),
        )
        entries.append(LogEntry(0, self.agent1.name, answer1))
        entries.append(LogEntry(0, self.agent2.name, answer2))
        logger.info("Round 0: initial responses")
        logger.info("%s: %s", self.agent1.name, answer1)
        logger.info("%s: %s", self.agent2.name, answer2)

        prev1, prev2 = answer1, answer2
        agreed = False
        rounds_used = 0

        for round_num in range(1, self.max_rounds + 1):
            rounds_used = round_num
            if self.concurrent_rounds:
                left = functools.partial(
                    self.agent1.respond_to_other_agent, self.agent2.name, prev2, question
                )
                right = functools.partial(
                    self.agent2.respond_to_other_agent, self.agent1.name, prev1, question
                )
                reply1, reply2 = self._two(left, right)
            else:
                reply1 = self.agent1.respond_to_other_agent(self.agent2.name, prev2, question)
                reply2 = self.agent2.respond_to_other_agent(self.agent1.name, reply1.text, question)

            entries.append(LogEntry(round_num, self.agent1.name, reply1.text, reply1.agrees))
            entries.append(LogEntry(round_num, self.agent2.name, reply2.text, reply2.agrees))
            logger.info("Round %d: %s -> %s", round_num, reply1.text, reply2.text)
            logger.info(
                "Round %d verdicts: %s=%s, %s=%s",
                round_num,
                self.agent1.name,
                reply1.agrees,
                self.agent2.name,
                reply2.agrees,
            )

            if reply1.agrees and reply2.agrees:
                agreed = True
                logger.info("Agreement reached after %d round(s)", round_num)
                break

            prev1, prev2 = reply1.text, reply2.text

        final_answer = self._generate_consensus_answer(question, prev1, prev2)

        usage = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        for agent in (self.agent1, self.agent2, self.consensus_agent):
            for key, value in agent.usage().items():
                usage[key] += value

        self.last_result = DiscussionResult(
            question=question,
            final_answer=final_answer,
            agreed=agreed,
            rounds_used=rounds_used,
            entries=entries,
            usage=usage,
        )
        return self.last_result

    def get_discussion_log(self) -> list[str]:
        """Log of the most recent discussion (kept for compatibility with the 1.x API)."""
        return self.last_result.log if self.last_result else []

    def _generate_consensus_answer(self, question: str, answer1: str, answer2: str) -> str:
        prompt = (
            f"Question: {question}\n\n"
            f"{self.agent1.name} said:\n{quote(answer1)}\n\n"
            f"{self.agent2.name} said:\n{quote(answer2)}\n\n"
            "Provide a final, consolidated answer that captures the consensus. If the two "
            "differ, reconcile them into one coherent answer and note the disagreement."
        )
        final_answer = self.consensus_agent.generate(prompt, fresh=True)
        logger.info("Final consensus answer:\n%s", final_answer)
        return final_answer


def ask_question(
    question: str,
    model_name: str = DEFAULT_MODEL,
    ollama_url: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    num_ctx: int = DEFAULT_NUM_CTX,
    seed: int | None = DEFAULT_SEED,
    max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS,
    retries: int = DEFAULT_RETRIES,
    timeout: float | None = DEFAULT_TIMEOUT,
) -> DiscussionResult:
    """
    Ask a question and get a consensus answer from two debating agents.

    Returns a ``DiscussionResult``; ``str(result)`` is the final answer.

    Example:
        >>> result = ask_question("What is the capital of France?", model_name="mistral")
        >>> result.agreed
        True
    """
    discussion = TwinAgentDiscussion(
        model_name=model_name,
        ollama_url=ollama_url,
        temperature=temperature,
        max_rounds=max_rounds,
        num_ctx=num_ctx,
        seed=seed,
        max_history_turns=max_history_turns,
        retries=retries,
        timeout=timeout,
    )
    return discussion.get_answer(question)
