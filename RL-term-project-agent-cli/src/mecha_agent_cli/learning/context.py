"""Compact deterministic context helpers for the learning sidecar."""

from __future__ import annotations

import re


def task_family(user_request: str) -> str:
    """Return a broad, non-solution-specific task family label."""
    text = _normalise_request(user_request)
    if _has_any(text, ("sort", "sorted", "ordering", "merge sort", "quicksort")):
        return "sort"
    if _has_any(text, ("fibonacci", "fib", "dynamic programming", "memoization", "memoize")):
        return "fibonacci_dp"
    if _has_any(text, ("moving average", "signal", "filter", "smoothing", "convolution")):
        return "signal_filter"
    if _has_any(
        text,
        (
            "linear system",
            "matrix",
            "numerical",
            "integrate",
            "integration",
            "root finding",
            "eigen",
            "optimization",
        ),
    ):
        return "numerical"
    if _has_any(
        text,
        (
            "reinforcement learning",
            "q-learning",
            "q learning",
            "policy gradient",
            "bandit",
            "markov decision",
            "mdp",
        ),
    ):
        return "general_rl"
    return "generic"


def build_context_key(*, user_request: str, model_name: str, has_baseline: bool) -> str:
    """Build a deterministic placeholder context key."""
    family = task_family(user_request)
    mode = "edit" if has_baseline else "new"
    safe_model = (model_name or "unknown").strip().lower() or "unknown"
    return f"{family}|{safe_model}|{mode}"


def _normalise_request(user_request: str) -> str:
    """Return a lowercase request string with compact whitespace."""
    return re.sub(r"\s+", " ", user_request.strip().lower())


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    """Return whether any broad keyword appears in ``text``."""
    return any(needle in text for needle in needles)


__all__ = ["build_context_key", "task_family"]
