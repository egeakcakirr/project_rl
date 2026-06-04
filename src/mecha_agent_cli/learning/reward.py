"""Shaped reward derivation from AgentRunResult for the contextual bandit.

``episode_reward`` produces the episode-level reward in [-1, 1] reported on
each run (the benchmark ``mean_reward`` metric). ``reward_to_beta_update`` is
the bridge consumed by the bandit to convert a reward into a Beta-Bernoulli
pseudo-success in [0, 1].
"""

from __future__ import annotations

from mecha_agent_cli.agent.result import AgentRunResult
from mecha_agent_cli.config.schema import LearningConfig
from mecha_agent_cli.core.types import FailureType


def episode_reward(result: AgentRunResult, cfg: LearningConfig) -> float:
    """Return a shaped episode reward in ``[-1.0, 1.0]``."""
    report = result.validation_report
    success = result.status == "success"
    attempts = max(1, result.total_attempts)
    passed_checks = sum(1 for r in report.results if r.passed)

    base = cfg.success_reward if success else 0.0
    progress = 0.0 if success else cfg.progress_bonus * float(passed_checks)
    attempt_cost = cfg.attempt_penalty * float(attempts - 1)

    def failed(name: str) -> bool:
        return report.primary_failure is FailureType.SEMANTIC and any(
            r.name == name and not r.passed for r in report.results
        )

    extract_cost = cfg.extract_failure_penalty if failed("extract") else 0.0
    behavior_cost = cfg.behavior_failure_penalty if failed("behavior") else 0.0

    duration = float(result.duration_sec or 0.0)
    if cfg.latency_penalty > 0.0 and cfg.latency_horizon_sec > 0.0 and duration > 0.0:
        latency_cost = cfg.latency_penalty * min(1.0, duration / cfg.latency_horizon_sec)
    else:
        latency_cost = 0.0

    raw = base + progress - attempt_cost - extract_cost - behavior_cost - latency_cost
    return max(-1.0, min(1.0, raw))


def reward_to_beta_update(reward: float) -> float:
    """Map reward in ``[-1, 1]`` to a pseudo-success in ``[0, 1]``."""
    return max(0.0, min(1.0, (reward + 1.0) / 2.0))


__all__ = ["episode_reward", "reward_to_beta_update"]
