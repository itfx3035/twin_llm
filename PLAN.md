# Improvement plan — `twin_llm` (issues 1–7)

Working document. Status: `[ ]` pending, `[x]` done, `[-]` skipped (with reason).

**Status: complete.** All 7 items implemented; `ruff check` + `ruff format --check` clean,
21 tests pass, CLI verified. Notes on deviations at the bottom.

Scope agreed: the 7 prioritised items from the review. Presentation is moved out
of the library, the API becomes structured, and repo hygiene is added.

---

## 1. Reset state per question + move `print()` out of `twin_lib.py`

- [x] `Agent.reset()` clears per-agent history; `TwinAgentDiscussion.get_answer()`
      resets both agents and the log at the start (fixes cross-question leakage).
- [x] Replace all 16 `print()` calls with `logging` via `logger = logging.getLogger("twin_llm")`.
- [x] No truncated `[:100]` previews inside the library; full text in structured log entries.
- [x] Caller (`main.py` / examples) configures `logging.basicConfig(...)`.

## 2. Structured result + deterministic agreement parsing

- [x] `DiscussionResult` dataclass: `question`, `final_answer`, `agreed`,
      `rounds_used`, `entries` (structured), `log` (string list, kept for compat),
      `llm_calls`, `prompt_tokens`, `completion_tokens`; `__str__` returns `final_answer`
      so `print(result)` still works.
- [x] `ask_question()` and `get_answer()` return `DiscussionResult` (**breaking change**, v2.0.0).
- [x] Agents must end each reply with `AGREE: YES` / `AGREE: NO`; parsed with
      `parse_verdict()` (last match wins; natural language is treated as unclear instead of
      guessed, so it is never mis-scored as agreement).
- [x] The fallback verdict check runs at `temperature=0` (+ optional `seed`) for determinism.
- [x] Extra verification LLM call only as a fallback when no verdict marker is present
      (`check_agreement()` kept as public method) → 5 calls typical instead of 23 at `max_rounds=5`.
- [x] `to_dict()` for JSON output from the CLI.

## 3. Message history correctness

- [x] Exactly one system message at index 0 per agent; later turns are user/assistant only.
- [x] `num_ctx` configurable (`TWIN_NUM_CTX`, default 8192) instead of hard-coded `6140`.
- [x] History pruning: keep system + last `max_history_turns` turns (env `TWIN_MAX_HISTORY_TURNS`).
- [x] Use typed LangChain messages when `langchain_core` is available, fall back to
      `(role, content)` tuples so tests do not need LangChain installed.
- [x] Quote the other agent's text with delimiters so it is data, not instructions.

## 4. Repo hygiene, tests, error handling

- [x] `requirements.txt` (pinned), `pyproject.toml` (package metadata + dev extras + ruff config).
- [x] `.gitignore`, `LICENSE` (renamed from `LICENSE.txt`), `.github/workflows/ci.yml` (ruff + pytest).
- [x] Lazy import of `langchain_ollama` with a friendly `TwinLLMError` message; LLM injectable
      for tests (`Agent(..., llm=stub)`).
- [x] Retries with capped exponential backoff + `request_timeout`; clear message for
      connection errors / missing model (`ollama pull` hint).
- [x] Argument validation: `max_rounds >= 0`, `0 <= temperature <= 2`, `num_ctx > 0`,
      `max_history_turns >= 1`.
- [x] `tests/test_twin_lib.py` (pytest, stub LLM, no network): verdict parsing, single system
      message, history pruning, reset between questions, early agreement, max-rounds no-agreement,
      fallback verdict call, retry + failure, structured result fields, CLI exit codes.
- [x] Cost/latency visibility: LLM call count + usage tokens per agent, aggregated in the result.

## 5. `run.py` → real CLI

- [x] Rename `run.py` to `main.py` with `argparse` + `if __name__ == "__main__":`.
- [x] Flags: `--question/-q`, `--model`, `--url`, `--temperature`, `--max-rounds`, `--num-ctx`,
      `--seed`, `--quiet`, `--verbose`, `--json`, `--strict` (exit 2 when no consensus),
      `--max-history-turns`, `--retries`, `--timeout`, `--concurrent-rounds`, `--no-parallel`.
- [x] Defaults from env vars: `OLLAMA_URL`, `OLLAMA_MODEL`, `TWIN_TEMPERATURE`, `TWIN_MAX_ROUNDS`,
      `TWIN_NUM_CTX`, `TWIN_SEED`, `TWIN_RETRIES`, `TWIN_TIMEOUT`, `TWIN_MAX_HISTORY_TURNS`.
- [x] Replaced the arithmetic demo question (LLMs cannot compute `52460484.69331842` reliably)
      with a judgement question; added `examples/` scripts.
- [x] Exit codes: 0 success, 1 error, 2 no-consensus under `--strict`.

## 6. README fixes

- [x] Fixed `ollama serve 0.0.0.0:11434` → `OLLAMA_HOST=0.0.0.0:11434 ollama serve`.
- [x] Fixed the `http://{server_ip _address}:11434` typo; model names now consistent with the CLI default.
- [x] Corrected the Python version claim to 3.10+.
- [x] Added: TOC, project layout, env-var table, CLI usage, troubleshooting, cost/latency warning,
      limitations, testing instructions, changelog for the v2.0 breaking change, license section.

## 7. Genuine debate + optional concurrency

- [x] Distinct per-agent personas (`PERSONA_ALPHA` / `PERSONA_BETA`) so the twins are not
      identical echo chambers; per-agent persona/model via injected `Agent` objects.
- [x] Neutral third agent for consensus instead of poking `self.agent1.llm`.
- [x] Optional `seed` for reproducibility.
- [x] Optional concurrency: parallel initial responses, and `concurrent_rounds=True` where each
      agent answers the other's *previous-round* reply simultaneously (`ThreadPoolExecutor`).
      Default stays sequential to preserve original semantics.

---

## Deliverables

| File | Action | Status |
|---|---|---|
| `twin_lib.py` | rewrite (logging, structured result, parsing, pruning, retries, personas, DI) | done |
| `main.py` | new CLI (replaces `run.py`) | done |
| `run.py` | removed (CLI content moved to `main.py`) | done |
| `tests/test_twin_lib.py` | new (21 tests) | done |
| `examples/basic_usage.py` | new | done |
| `examples/per_agent_personas.py` | new | done |
| `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`, `.gitignore` | new | done |
| `.github/workflows/ci.yml` | new | done |
| `README.md` | rewritten | done |
| `LICENSE.txt` → `LICENSE` | renamed | done |
| `PLAN.md` | this plan | done |

## Notes / deviations

- `parse_verdict()` no longer guesses from stray "yes"/"no" words; unclear replies return `None`
  and trigger the low-temperature fallback check. This avoids false "agreement".
- Debate replies use the agent's `temperature`; only the fallback verdict check is forced to 0.0.
- `run.py` was deleted rather than `git mv`'d because `main.py` already existed when the rename ran;
  the content is fully replaced by the CLI either way.
- Nothing was committed — changes are staged/untracked in the working tree for review.

## Out of scope (deliberately)

- Rewriting the framework as an async service, DB persistence, or a web UI.
- Embedding-similarity convergence (deferred; deterministic verdict markers already fix the
  fragile parsing and most of the wasted calls).
- CI beyond lint + unit tests (no model-based integration tests, they need a running Ollama).

## Verification before completion

1. `python3 -m pytest -q` — **21 passed** (runs without Ollama and without `langchain_ollama`).
2. `ruff check .` — clean; `ruff format --check .` — clean.
3. `python3 -m py_compile` on every file — ok; `python3 main.py --help` works; error path prints a
   friendly message and exits 1.

Not done (deliberately): no end-to-end run against a live Ollama server from this environment.
