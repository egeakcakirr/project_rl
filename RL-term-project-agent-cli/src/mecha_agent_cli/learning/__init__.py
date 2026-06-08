"""Learning-sidecar package exports."""

from __future__ import annotations

from mecha_agent_cli.learning.arm_registry import ARM_REGISTRY, Arm, get_arm, list_arm_ids
from mecha_agent_cli.learning.bandit import ArmStat, BanditStore, ThompsonBandit
from mecha_agent_cli.learning.context import build_context_key, task_family
from mecha_agent_cli.learning.q_learning import QLearningPlaceholder, placeholder_status
from mecha_agent_cli.learning.reward import episode_reward, reward_to_beta_update

__all__ = [
    "ARM_REGISTRY",
    "Arm",
    "ArmStat",
    "BanditStore",
    "QLearningPlaceholder",
    "ThompsonBandit",
    "build_context_key",
    "episode_reward",
    "get_arm",
    "list_arm_ids",
    "placeholder_status",
    "reward_to_beta_update",
    "task_family",
]
