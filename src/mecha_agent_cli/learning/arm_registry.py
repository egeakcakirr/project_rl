"""Placeholder arm registry for course starter repositories.

This module intentionally provides only a minimal, import-safe action
registry. Students are expected to design and implement the real action
space as part of the learning assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mecha_agent_cli.config.schema import ModelProfile


@dataclass(frozen=True)
class Arm:
    """Minimal action descriptor used by placeholder learning modules."""

    arm_id: str
    profile_name: str
    overrides: dict[str, Any] = field(default_factory=dict[str, Any])
    description: str = ""

    def apply(self, base: ModelProfile) -> ModelProfile:
        """Apply arm-specific hyperparameter overrides to the base model profile safely."""
        if not self.overrides:
            return base
        
        current_config = dict(base.model_extra or {})
        current_config.update(self.overrides)
        
        return base.model_copy(update={"model_extra": current_config})


ARM_REGISTRY: tuple[Arm, ...] = (
    Arm(
        arm_id="direct.baseline",
        profile_name="direct",
        overrides={},
        description="Placeholder baseline arm.",
    ),
    Arm(
        arm_id="direct.cold_precise",
        profile_name="direct",
        overrides={"temperature": 0.0, "top_p": 0.8},
        description="Deterministic low temperature arm.",
    ),
    Arm(
        arm_id="direct.balanced",
        profile_name="direct",
        overrides={"temperature": 0.4, "top_p": 0.9},
        description="Balanced exploration profile.",
    ),
    Arm(
        arm_id="direct.long_stable",
        profile_name="direct",
        overrides={"temperature": 0.2, "num_predict": 2048, "repeat_penalty": 1.1},
        description="Extended token context stability.",
    ),
    Arm(
        arm_id="direct.repeat_guard",
        profile_name="direct",
        overrides={"temperature": 0.3, "repeat_penalty": 1.4, "frequency_penalty": 0.5},
        description="Strict penalty arm.",
    ),
)

_BY_ID: dict[str, Arm] = {arm.arm_id: arm for arm in ARM_REGISTRY}


def get_arm(arm_id: str) -> Arm:
    """Return the arm with id ``arm_id`` or raise :class:`KeyError`."""
    if arm_id not in _BY_ID:
        msg = f"Unknown arm_id: {arm_id!r}"
        raise KeyError(msg)
    return _BY_ID[arm_id]


def list_arm_ids() -> list[str]:
    """Return all registered arm ids in registration order."""
    return [arm.arm_id for arm in ARM_REGISTRY]


__all__ = ["ARM_REGISTRY", "Arm", "get_arm", "list_arm_ids"]
