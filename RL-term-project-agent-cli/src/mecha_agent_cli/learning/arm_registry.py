"""Safe model-profile arm registry for the learning sidecar."""

from __future__ import annotations

from dataclasses import dataclass, field

from mecha_agent_cli.config.schema import ModelProfile


def _empty_overrides() -> dict[str, object]:
    """Return a fresh empty override mapping for dataclass defaults."""
    return {}


@dataclass(frozen=True)
class Arm:
    """Small action descriptor for direct-generation profile variants."""

    arm_id: str
    profile_name: str
    overrides: dict[str, object] = field(default_factory=_empty_overrides)
    description: str = ""

    def apply(self, base: ModelProfile) -> ModelProfile:
        """Return a validated copy of ``base`` with this arm's overrides."""
        data = base.model_dump()
        supported = set(data)
        unknown = sorted(set(self.overrides) - supported)
        if unknown:
            fields = ", ".join(unknown)
            msg = f"Unsupported ModelProfile override(s) for {self.arm_id}: {fields}"
            raise ValueError(msg)
        data.update(self.overrides)
        return type(base).model_validate(data)


ARM_REGISTRY: tuple[Arm, ...] = (
    Arm(
        arm_id="direct.baseline",
        profile_name="direct",
        overrides={},
        description="Placeholder baseline arm.",
    ),
    Arm(
        arm_id="direct.cold",
        profile_name="direct",
        overrides={"temperature": 0.1, "top_p": 0.70},
        description="Lower-variance direct generation.",
    ),
    Arm(
        arm_id="direct.cool",
        profile_name="direct",
        overrides={"temperature": 0.2, "top_p": 0.85},
        description="Conservative sampling with modest diversity.",
    ),
    Arm(
        arm_id="direct.warm",
        profile_name="direct",
        overrides={"temperature": 0.6, "top_p": 0.90},
        description="Moderately exploratory direct generation.",
    ),
    Arm(
        arm_id="direct.hot",
        profile_name="direct",
        overrides={"temperature": 0.8, "top_p": 0.95},
        description="Higher-diversity direct generation.",
    ),
    Arm(
        arm_id="direct.tight_topk",
        profile_name="direct",
        overrides={"top_k": 20, "top_p": 0.75},
        description="Narrower token sampling.",
    ),
    Arm(
        arm_id="direct.broad_topk",
        profile_name="direct",
        overrides={"top_k": 80, "top_p": 0.95},
        description="Broader token sampling.",
    ),
    Arm(
        arm_id="direct.high_repeat",
        profile_name="direct",
        overrides={"repeat_penalty": 1.15},
        description="Slightly stronger repetition penalty.",
    ),
    Arm(
        arm_id="direct.no_think",
        profile_name="direct",
        overrides={"think": False},
        description="Disable model thinking mode when supported.",
    ),
    Arm(
        arm_id="direct.fixed_seed",
        profile_name="direct",
        overrides={"seed": 7},
        description="Deterministic sampler seed variant.",
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
