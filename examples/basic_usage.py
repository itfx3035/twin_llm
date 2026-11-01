"""Minimal library usage: round-by-round output comes from the logger."""

from __future__ import annotations

import logging

from twin_lib import TwinAgentDiscussion

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

discussion = TwinAgentDiscussion(
    model_name="ministral-3:latest",
    ollama_url="http://localhost:11434",
    temperature=0.5,
    max_rounds=3,
)

result = discussion.get_answer("What makes a good software architecture?")

print("\nFinal answer:", result.final_answer)
print("Agreed:", result.agreed, "in", result.rounds_used, "round(s)")
print("LLM calls:", result.usage["llm_calls"])
for line in result.log:
    print(line)
