#!/usr/bin/env python3
"""Command-line interface for the twin-agent discussion system."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from twin_lib import (
    DEFAULT_MAX_HISTORY_TURNS,
    DEFAULT_MAX_ROUNDS,
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_OLLAMA_URL,
    DEFAULT_RETRIES,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    TwinAgentDiscussion,
    TwinLLMError,
)

DEFAULT_QUESTION = (
    "What are the main trade-offs of running a local LLM instead of a hosted API for a small team?"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the argument parser; every default falls back to an env var."""
    parser = argparse.ArgumentParser(
        prog="twin-llm",
        description="Let two agents discuss a question and reach a consensus answer.",
        epilog=(
            "Environment defaults: OLLAMA_URL, OLLAMA_MODEL, TWIN_TEMPERATURE, TWIN_MAX_ROUNDS, "
            "TWIN_NUM_CTX, TWIN_SEED, TWIN_RETRIES, TWIN_TIMEOUT, TWIN_MAX_HISTORY_TURNS."
        ),
    )
    parser.add_argument("-q", "--question", default=DEFAULT_QUESTION, help="Question to discuss.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model (must be pulled).")
    parser.add_argument("--url", default=DEFAULT_OLLAMA_URL, help="Ollama server URL.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Debate creativity; the fallback verdict check runs at 0.0.",
    )
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    parser.add_argument("--max-history-turns", type=int, default=DEFAULT_MAX_HISTORY_TURNS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for reproducibility.")
    parser.add_argument(
        "--concurrent-rounds",
        action="store_true",
        help="Both agents answer the previous round simultaneously (faster).",
    )
    parser.add_argument(
        "--no-parallel",
        dest="parallel",
        action="store_false",
        help="Run the initial answers sequentially.",
    )
    parser.add_argument("--json", action="store_true", help="Print the structured result as JSON.")
    parser.add_argument(
        "--strict", action="store_true", help="Exit with code 2 if no consensus was reached."
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the final answer.")
    parser.add_argument("--verbose", action="store_true", help="Debug-level logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one discussion and print the result. Returns the process exit code."""
    args = parse_args(argv)

    level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")

    try:
        discussion = TwinAgentDiscussion(
            model_name=args.model,
            ollama_url=args.url,
            temperature=args.temperature,
            max_rounds=args.max_rounds,
            num_ctx=args.num_ctx,
            seed=args.seed,
            max_history_turns=args.max_history_turns,
            retries=args.retries,
            timeout=args.timeout,
            parallel=args.parallel,
            concurrent_rounds=args.concurrent_rounds,
        )
        result = discussion.get_answer(args.question)
    except (TwinLLMError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.final_answer)
        if not args.quiet:
            print(
                f"\nagreed={result.agreed} rounds={result.rounds_used} "
                f"llm_calls={result.usage.get('llm_calls', 0)}",
                file=sys.stderr,
            )

    return 2 if (args.strict and not result.agreed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
