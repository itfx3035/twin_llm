"""Distinct personas and per-agent models — genuine disagreement, not echo chambers.

The same model with the same persona mostly agrees with itself, so give each agent
a different persona (or a different model) to get a real debate.
"""

from __future__ import annotations

import logging

from twin_lib import Agent, TwinAgentDiscussion

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

alpha = Agent(
    name="OptimistAgent",
    model_name="ministral-3:latest",
    persona="An advocate for the proposed approach who argues the strongest case for it.",
    temperature=0.8,
)
beta = Agent(
    name="RedTeamAgent",
    model_name="ministral-3:latest",
    persona="A red-team reviewer whose job is to find weaknesses and risks, never rubber-stamping.",
    temperature=0.3,
)

discussion = TwinAgentDiscussion(agent1=alpha, agent2=beta, max_rounds=4, concurrent_rounds=True)
result = discussion.get_answer("Should a small team migrate from a monolith to microservices?")

print(f"\nagreed={result.agreed} rounds={result.rounds_used} calls={result.usage['llm_calls']}")
print(result.final_answer)
