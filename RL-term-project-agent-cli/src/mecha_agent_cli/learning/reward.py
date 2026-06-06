"""Bounded reward helpers for the learning sidecar."""

from __future__ import annotations

from mecha_agent_cli.agent.result import AgentRunResult
from mecha_agent_cli.config.schema import LearningConfig


def episode_reward(result: AgentRunResult, cfg: LearningConfig) -> float:
    """Return a shaped episode reward in ``[-1.0, 1.0]``."""
    success = str(getattr(result, "status", "")).lower() == "success"
    attempts = max(1, _int_attr(result, "total_attempts", 1))
    report = getattr(result, "validation_report", None)

    success_reward = _float_attr(cfg, "success_reward", 1.0)
    failure_reward = _float_attr(cfg, "failure_reward", 0.0)
    failure_penalty = _float_attr(cfg, "failure_penalty", 0.0)
    attempt_penalty = _float_attr(cfg, "attempt_penalty", 0.0) * float(
        max(0, attempts - 1)
    )
    progress_bonus = 0.0
    if not success:
        progress_bonus = _float_attr(cfg, "progress_bonus", 0.0) * float(
            _passed_checks(report)
        )

    failed_checks = _failed_check_names(report)
    primary_failure = _primary_failure_name(report)
    runtime_failure = "runtime" in failed_checks or primary_failure == "runtime"
    behavior_failure = "behavior" in failed_checks
    extract_cost = (
        _float_attr(cfg, "extract_failure_penalty", 0.0)
        if "extract" in failed_checks
        else 0.0
    )
    behavior_cost = (
        _float_attr(cfg, "behavior_failure_penalty", 0.0)
        if behavior_failure
        else 0.0
    )
    runtime_cost = (
        _float_attr(cfg, "runtime_failure_penalty", 0.0)
        if runtime_failure
        else 0.0
    )
    latency_cost = _latency_penalty(result, cfg)

    raw = success_reward if success else failure_reward - failure_penalty
    raw += progress_bonus
    raw -= attempt_penalty + extract_cost + behavior_cost + runtime_cost + latency_cost
    if runtime_failure and not success:
        raw = min(raw, _float_attr(cfg, "runtime_failure_reward_cap", 0.25))
    return _clamp(raw, lower=-1.0, upper=1.0)


def reward_to_beta_update(reward: float) -> float:
    """Map reward in ``[-1, 1]`` to a pseudo-success in ``[0, 1]``."""
    return max(0.0, min(1.0, (reward + 1.0) / 2.0))


def _passed_checks(report: object) -> int:
    """Return the count of passed, non-skipped validation checks."""
    count = 0
    for item in getattr(report, "results", ()) or ():
        passed = bool(getattr(item, "passed", False))
        skipped = bool(getattr(item, "skipped", False))
        if passed and not skipped:
            count += 1
    return count


def _failed_check_names(report: object) -> set[str]:
    """Return lowercase names for failed validation checks."""
    names: set[str] = set()
    for item in getattr(report, "results", ()) or ():
        if bool(getattr(item, "passed", False)):
            continue
        name = str(getattr(item, "name", "")).strip().lower()
        if name:
            names.add(name)
    return names


def _primary_failure_name(report: object) -> str:
    """Return the lowercase primary failure label when available."""
    failure = getattr(report, "primary_failure", "")
    value = getattr(failure, "value", failure)
    return str(value).strip().lower()


def _latency_penalty(result: AgentRunResult, cfg: LearningConfig) -> float:
    """Return a bounded latency penalty when duration data is available."""
    duration = max(0.0, _float_attr(result, "duration_sec", 0.0))
    penalty = max(0.0, _float_attr(cfg, "latency_penalty", 0.0))
    horizon = _float_attr(cfg, "latency_horizon_sec", 0.0)
    if duration <= 0.0 or penalty <= 0.0 or horizon <= 0.0:
        return 0.0
    return penalty * min(1.0, duration / horizon)


def _float_attr(obj: object, name: str, default: float) -> float:
    """Return ``name`` coerced to float, or ``default`` when unavailable."""
    value = getattr(obj, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_attr(obj: object, name: str, default: int) -> int:
    """Return ``name`` coerced to int, or ``default`` when unavailable."""
    value = getattr(obj, name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, *, lower: float, upper: float) -> float:
    """Clamp ``value`` into the inclusive ``[lower, upper]`` interval."""
    return max(lower, min(upper, value))


__all__ = ["episode_reward", "reward_to_beta_update"]
