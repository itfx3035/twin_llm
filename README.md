# Twins: Twin-Agent Discussion System

Two LLM agents (Agent-Alpha and Agent-Beta) answer a question independently, debate
each other's reasoning round by round, and a neutral prompt then synthesises the
consensus answer. Runs entirely against a local [Ollama](https://ollama.ai) server.

- **Two autonomous agents** with distinct personas debate from different angles
- **Round-by-round discussion** with machine-readable agree/disagree verdicts
- **Consensus building** that reconciles the two perspectives
- **Offline-first** — no external API, no data leaves your machine
- **Configurable** model, temperature, round budget, context window, retries, seed

Use it when you want a second opinion on a judgement call. It is *not* a calculator
and not a fact-checker: agents can agree on a wrong answer, so the value is debate,
not verification.

## Contents

- [Quick start](#quick-start)
- [Command-line usage](#command-line-usage)
- [Library usage](#library-usage)
- [Configuration and environment variables](#configuration-and-environment-variables)
- [How it works](#how-it-works)
- [Cost and latency](#cost-and-latency)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)
- [Development and tests](#development-and-tests)
- [Changelog](#changelog)
- [License](#license)

## Quick start

### Prerequisites

1. **Ollama** installed and running (https://ollama.ai)
2. **Python 3.10+**
3. A pulled model:

```bash
ollama pull ministral-3:latest     # default used by the CLI
# or: ollama pull mistral / neural-chat / dolphin-mixtral
```

### Install

```bash
git clone https://github.com/itfx3035/twin_llm && cd twin_llm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python main.py --question "What are the pros and cons of AI in healthcare?"
```

## Command-line usage

```bash
python main.py -q "Should we adopt a managed Postgres or run our own?" \
  --model ministral-3:latest \
  --url http://localhost:11434 \
  --temperature 0.4 \
  --max-rounds 4 \
  --num-ctx 8192 \
  --seed 42 \
  --json
```

| Flag | Default | Purpose |
|---|---|---|
| `-q, --question` | sample judgement question | Question to discuss |
| `--model` | `ministral-3:latest` (or `OLLAMA_MODEL`) | Ollama model, must be pulled |
| `--url` | `http://localhost:11434` (or `OLLAMA_URL`) | Ollama server URL |
| `--temperature` | `0.7` | Debate creativity; the fallback verdict check runs at `0.0` |
| `--max-rounds` | `5` | Round budget (`0` skips the debate) |
| `--num-ctx` | `8192` | Context window passed to Ollama |
| `--max-history-turns` | `20` | Turns kept per agent before pruning |
| `--retries` | `3` | LLM call attempts, capped exponential backoff |
| `--timeout` | `120` | Per-request timeout (seconds) |
| `--seed` | unset | Reproducible runs |
| `--concurrent-rounds` | off | Both agents answer the previous round simultaneously |
| `--no-parallel` | off | Run the two initial answers sequentially |
| `--json` | off | Structured output (question, answer, agreed, rounds, usage, log) |
| `--strict` | off | Exit `2` when no consensus was reached |
| `--quiet` / `--verbose` | off | Only the final answer / debug logging |

Exit codes: `0` success, `1` error (server unreachable, model missing, bad args),
`2` no consensus under `--strict`.

## Library usage

```python
import logging
from twin_lib import ask_question

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

result = ask_question(
    "What makes a good software architecture?",
    model_name="ministral-3:latest",
    ollama_url="http://localhost:11434",
    temperature=0.5,
    max_rounds=3,
)

print(result.final_answer)  # str(result) is also the final answer
print(result.agreed, result.rounds_used, result.usage["llm_calls"])
for line in result.log:
    print(line)
```

The library logs instead of printing, so you decide what is shown. Round output
appears as `INFO` records on the `twin_llm` logger.

### Distinct personas / different models

Identical model + identical persona → the agents mostly agree with themselves. Give
them different personas (see `examples/per_agent_personas.py`):

```python
from twin_lib import Agent, TwinAgentDiscussion

alpha = Agent(
    name="OptimistAgent",
    model_name="ministral-3:latest",
    persona="An advocate who argues the strongest case for the proposal.",
)
beta = Agent(
    name="RedTeamAgent",
    model_name="ministral-3:latest",
    persona="A red-team reviewer who looks for weaknesses and never rubber-stamps.",
    temperature=0.3,
)

discussion = TwinAgentDiscussion(agent1=alpha, agent2=beta, max_rounds=4)
result = discussion.get_answer("Should a small team split a monolith into microservices?")
```

## Configuration and environment variables

Defaults are read from the environment, so a shell profile or `.env` file is enough:

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `ministral-3:latest` | Model name |
| `TWIN_TEMPERATURE` | `0.7` | Debate temperature (verdicts are `0.0`) |
| `TWIN_MAX_ROUNDS` | `5` | Round budget |
| `TWIN_NUM_CTX` | `8192` | Context window |
| `TWIN_MAX_HISTORY_TURNS` | `20` | History kept per agent |
| `TWIN_RETRIES` | `3` | Retry attempts per LLM call |
| `TWIN_TIMEOUT` | `120` | Request timeout (seconds) |
| `TWIN_SEED` | unset | Generation seed |

### Ollama on another machine

`ollama serve` does **not** take an address argument; use `OLLAMA_HOST`:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Then connect with `--url http://192.168.1.100:11434` or `export OLLAMA_URL=...`.

## How it works

1. Each agent answers the question independently (round 0).
2. Each round, the agents react to the other's previous reply and must end with a
   machine-readable verdict line (`AGREE: YES` / `AGREE: NO`).
3. The debate stops as soon as both verdicts agree, or the round budget is exhausted.
4. A neutral prompt reconciles the last two answers into the final consensus, noting
   any remaining disagreement.

Verdicts are parsed from the reply itself, so a typical run costs `2 + 2R + 1` LLM
calls instead of `2 + 2R + 2R + 1` — 13 at `max_rounds=5`, or as few as 5 when the
agents agree in round 1. Only when a model ignores the required `AGREE:` format is a
single extra low-temperature check made per round.

## Cost and latency

The round budget is the cost knob. On a 7–8B local model a 5-round discussion is
13+ inferences, which can take a couple of minutes on CPU. Lower `--max-rounds`
and `--num-ctx`, or use `--concurrent-rounds` when you have GPU headroom.
`--json` reports `usage.llm_calls`, `prompt_tokens` and `completion_tokens`.

## Limitations

- **Agreement is self-reported.** Agents say what they think, not what is true; they
  can converge on a wrong answer, and early agreement is common.
- **LLMs cannot compute reliably.** Do not use it for arithmetic, counts or exact
  figures — the consensus will happily average two wrong numbers. Add a tool call if
  you need exact arithmetic.
- **Parsing is heuristic.** If a model ignores the `AGREE:` format, a single fallback
  low-temperature call is made; if even that is unclear, the verdict is `None`
  (treated as disagreement).
- **Context truncation.** History is pruned to `--max-history-turns`; agents can lose
  earlier nuance, and Ollama silently truncates anything beyond `num_ctx`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ConnectionError` / connection refused | Start Ollama (`ollama serve`) and check `--url` |
| `404 model not found` | `ollama pull <model>`, then `ollama list` to confirm the exact name |
| Verdicts always `None` | Use a model that follows instructions; keep `--temperature` low |
| Always agrees in round 1 | Give the agents different personas (and models) — see the personas example |
| Answers drift as rounds grow | Raise `--num-ctx`, lower `--max-history-turns`, or lower `--max-rounds` |
| Non-reproducible runs | Set `--seed` |

## Project layout

```
twin_llm/
├── twin_lib.py               # library (agents, discussion, result)
├── main.py                   # CLI entry point
├── examples/
│   ├── basic_usage.py
│   └── per_agent_personas.py
├── tests/test_twin_lib.py    # unit tests with a stub LLM (no Ollama needed)
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── .github/workflows/ci.yml
```

## Development and tests

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format .
pytest
```

The tests use an injected stub LLM, so they run without Ollama or a GPU.

## Changelog

**v2.0.0** (breaking changes from v1.x)

- `ask_question()` and `get_answer()` now return a `DiscussionResult` object
  (`final_answer`, `agreed`, `rounds_used`, `log`, `usage`). Use `result.final_answer`
  or `str(result)` instead of the plain string.
- The library no longer prints; it logs through the `twin_llm` logger.
- Agent state is reset per question, so reusing a `TwinAgentDiscussion` is safe.
- One system message per agent (was re-appended every round), history pruning,
  configurable `num_ctx`, deterministic verdict parsing, retries + timeouts.
- Distinct per-agent personas, optional per-agent models, optional concurrency.
- Added CLI (`main.py`), environment variables, tests, CI, `requirements.txt`.

## License

MIT — see [LICENSE](LICENSE).
