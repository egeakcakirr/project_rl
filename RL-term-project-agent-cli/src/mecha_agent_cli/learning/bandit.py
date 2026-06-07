"""Minimal contextual bandit implementation for learning-sidecar arms."""

from __future__ import annotations

import math
import random
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mecha_agent_cli.config.schema import LearningConfig
from mecha_agent_cli.learning.arm_registry import ARM_REGISTRY, Arm, get_arm
from mecha_agent_cli.learning.reward import reward_to_beta_update

_COLD_START_ARM_ORDER: tuple[str, ...] = (
    "direct.baseline",
    "direct.cold",
    "direct.cool",
    "direct.warm",
    "direct.high_repeat",
    "direct.no_think",
    "direct.tight_topk",
    "direct.fixed_seed",
    "direct.hot",
    "direct.broad_topk",
)
_RUNTIME_RECOVERY_ARM_ORDER: tuple[str, ...] = (
    "direct.baseline",
    "direct.cold",
    "direct.cool",
    "direct.high_repeat",
    "direct.no_think",
    "direct.warm",
    "direct.tight_topk",
    "direct.fixed_seed",
    "direct.hot",
    "direct.broad_topk",
)


@dataclass(frozen=True)
class ArmStat:
    """Minimal bookkeeping for one ``(context, arm)`` pair."""

    context_key: str
    arm_id: str
    alpha: float
    beta: float
    pulls: int
    cumulative_reward: float
    last_reward: float
    last_success: bool

    @property
    def mean(self) -> float:
        """Return the posterior mean for this arm statistic."""
        return self.alpha / (self.alpha + self.beta)


class BanditStore:
    """Bandit storage facade with in-memory and SQLite-backed modes."""

    def __init__(self, db_path: Path | None) -> None:
        self.db_path = db_path
        self._rows: dict[tuple[str, str], ArmStat] = {}
        if self.db_path is not None:
            self._ensure_schema()

    def fetch(self, context_key: str) -> dict[str, ArmStat]:
        """Return rows for ``context_key`` keyed by ``arm_id``."""
        if self.db_path is not None:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    """
                    SELECT context_key, arm_id, alpha, beta, pulls,
                           cumulative_reward, last_reward, last_success
                    FROM bandit_arms
                    WHERE context_key = ?
                    ORDER BY id
                    """,
                    (context_key,),
                ).fetchall()
            return {str(row["arm_id"]): _row_to_stat(row) for row in rows}

        out: dict[str, ArmStat] = {}
        for (ctx, arm_id), stat in self._rows.items():
            if ctx == context_key:
                out[arm_id] = stat
        return out

    def upsert(self, stat: ArmStat) -> None:
        """Insert or update one row."""
        if self.db_path is not None:
            updated_at = datetime.now(UTC).isoformat()
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    INSERT INTO bandit_arms (
                        context_key, arm_id, alpha, beta, pulls,
                        cumulative_reward, last_reward, last_success, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(context_key, arm_id) DO UPDATE SET
                        alpha = excluded.alpha,
                        beta = excluded.beta,
                        pulls = excluded.pulls,
                        cumulative_reward = excluded.cumulative_reward,
                        last_reward = excluded.last_reward,
                        last_success = excluded.last_success,
                        updated_at = excluded.updated_at
                    """,
                    (
                        stat.context_key,
                        stat.arm_id,
                        stat.alpha,
                        stat.beta,
                        stat.pulls,
                        stat.cumulative_reward,
                        stat.last_reward,
                        int(stat.last_success),
                        updated_at,
                    ),
                )
                conn.commit()
            return

        self._rows[(stat.context_key, stat.arm_id)] = stat

    def all_rows(self) -> list[ArmStat]:
        """Return all stored rows."""
        if self.db_path is not None:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    """
                    SELECT context_key, arm_id, alpha, beta, pulls,
                           cumulative_reward, last_reward, last_success
                    FROM bandit_arms
                    ORDER BY id
                    """
                ).fetchall()
            return [_row_to_stat(row) for row in rows]
        return list(self._rows.values())

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection for the configured bandit database."""
        if self.db_path is None:
            msg = "BanditStore has no SQLite db_path"
            raise RuntimeError(msg)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create the bandit table/index if the configured DB lacks them."""
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bandit_arms (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  context_key TEXT NOT NULL,
                  arm_id TEXT NOT NULL,
                  alpha REAL NOT NULL DEFAULT 1.0,
                  beta REAL NOT NULL DEFAULT 1.0,
                  pulls INTEGER NOT NULL DEFAULT 0,
                  cumulative_reward REAL NOT NULL DEFAULT 0.0,
                  last_reward REAL NOT NULL DEFAULT 0.0,
                  last_success INTEGER NOT NULL DEFAULT 0,
                  updated_at TEXT NOT NULL,
                  UNIQUE(context_key, arm_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bandit_arms_ctx "
                "ON bandit_arms(context_key)"
            )
            conn.commit()


class ThompsonBandit:
    """Small contextual bandit controller for safe direct-profile arms."""

    BASELINE_ARM_ID = "direct.baseline"

    def __init__(
        self,
        store: BanditStore,
        cfg: LearningConfig,
        *,
        arms: tuple[Arm, ...] = ARM_REGISTRY,
        rng: random.Random | None = None,
    ) -> None:
        self.store = store
        self.cfg = cfg
        self.arms = arms if arms else (get_arm(self.BASELINE_ARM_ID),)
        self._arm_ids = tuple(a.arm_id for a in self.arms)
        self._arms_by_id = {a.arm_id: a for a in self.arms}
        self.rng = rng or random.Random()

    # -- selection -------------------------------------------------------

    def select(self, context_key: str) -> Arm:
        """Select an arm for ``context_key`` according to configured mode."""
        if not bool(getattr(self.cfg, "enabled", True)):
            return self._baseline_arm()
        mode = str(getattr(self.cfg, "mode", "thompson")).strip().lower()
        if mode == "off":
            return self._baseline_arm()

        rows = self.store.fetch(context_key)
        under_pulled = self._under_pulled_arms(context_key=context_key, rows=rows)
        if under_pulled:
            return under_pulled[0]
        if mode == "greedy":
            return self._select_greedy(context_key=context_key, rows=rows)
        if mode == "ucb1":
            return self._select_ucb1(context_key=context_key, rows=rows)
        return self._select_thompson(context_key=context_key, rows=rows)

    # -- update ----------------------------------------------------------

    def update(
        self,
        *,
        context_key: str,
        arm_id: str,
        reward: float,
        success: bool,
    ) -> ArmStat:
        """Record a bounded posterior update and return the new stat."""
        prior = self.store.fetch(context_key).get(arm_id, _zero_stat(context_key, arm_id))
        pseudo_success = self._pseudo_success(reward=reward, success=success)
        decay = self._decay_factor()
        alpha_evidence = max(0.0, prior.alpha - 1.0) * decay
        beta_evidence = max(0.0, prior.beta - 1.0) * decay
        new = ArmStat(
            context_key=context_key,
            arm_id=arm_id,
            alpha=1.0 + alpha_evidence + pseudo_success,
            beta=1.0 + beta_evidence + (1.0 - pseudo_success),
            pulls=prior.pulls + 1,
            cumulative_reward=prior.cumulative_reward + reward,
            last_reward=reward,
            last_success=success,
        )
        self.store.upsert(new)
        return new

    def _baseline_arm(self) -> Arm:
        """Return the baseline arm, falling back to the global registry."""
        return self._arms_by_id.get(
            self.BASELINE_ARM_ID,
            get_arm(self.BASELINE_ARM_ID),
        )

    def _stat(
        self,
        *,
        context_key: str,
        arm_id: str,
        rows: dict[str, ArmStat],
    ) -> ArmStat:
        """Return current stats for ``arm_id`` in ``context_key``."""
        return rows.get(arm_id, _zero_stat(context_key, arm_id))

    def _under_pulled_arms(
        self,
        *,
        context_key: str,
        rows: dict[str, ArmStat],
    ) -> list[Arm]:
        """Return arms that have not yet reached the configured pull floor."""
        try:
            min_pulls = int(getattr(self.cfg, "min_pulls_before_exploit", 0))
        except (TypeError, ValueError):
            min_pulls = 0
        if min_pulls <= 0:
            return []
        under_pulled: list[Arm] = []
        for arm in self._ordered_arms(context_key):
            stat = self._stat(context_key=context_key, arm_id=arm.arm_id, rows=rows)
            if stat.pulls < min_pulls:
                under_pulled.append(arm)
        if not under_pulled:
            return []
        fewest_pulls = min(
            self._stat(context_key=context_key, arm_id=arm.arm_id, rows=rows).pulls
            for arm in under_pulled
        )
        return [
            arm
            for arm in under_pulled
            if self._stat(context_key=context_key, arm_id=arm.arm_id, rows=rows).pulls
            == fewest_pulls
        ]

    def _ordered_arms(self, context_key: str) -> list[Arm]:
        """Return arms in a conservative cold-start order."""
        preferred = (
            _RUNTIME_RECOVERY_ARM_ORDER
            if _context_has_previous_runtime_failure(context_key)
            else _COLD_START_ARM_ORDER
        )
        order = {arm_id: idx for idx, arm_id in enumerate(preferred)}
        fallback = len(order)
        return sorted(
            self.arms,
            key=lambda arm: (order.get(arm.arm_id, fallback), arm.arm_id),
        )

    def _select_thompson(self, *, context_key: str, rows: dict[str, ArmStat]) -> Arm:
        """Select by Beta posterior sampling with context-aware arm boosting.
        Arms that match structural features of the current task receive a
        prior boost to their sample, making them more likely to be selected
        before sufficient pull data is collected.
        """
        boost_map = _context_arm_boosts(context_key)
        best_arm = self._baseline_arm()
        best_sample = float("-inf")
        for arm in self.arms:
            stat = self._stat(context_key=context_key, arm_id=arm.arm_id, rows=rows)
            sample = self.rng.betavariate(
                max(stat.alpha, 1e-9),
                max(stat.beta, 1e-9),
            )
            sample += boost_map.get(arm.arm_id, 0.0)
            if sample > best_sample:
                best_arm = arm
                best_sample = sample
        return best_arm

    def _select_greedy(self, *, context_key: str, rows: dict[str, ArmStat]) -> Arm:
        """Select by posterior mean, with optional epsilon exploration."""
        epsilon = self._float_cfg("epsilon", default=0.0)
        if epsilon > 0.0 and self.rng.random() < min(1.0, epsilon):
            return self.rng.choice(self.arms)
        return max(
            self.arms,
            key=lambda arm: self._stat(context_key=context_key, arm_id=arm.arm_id, rows=rows).mean,
        )

    def _select_ucb1(self, *, context_key: str, rows: dict[str, ArmStat]) -> Arm:
        """Select by a simple UCB1 score over posterior means."""
        for arm in self.arms:
            if self._stat(context_key=context_key, arm_id=arm.arm_id, rows=rows).pulls <= 0:
                return arm
        total_pulls = sum(
            self._stat(context_key=context_key, arm_id=arm.arm_id, rows=rows).pulls
            for arm in self.arms
        )
        total = max(1, total_pulls)
        c = max(0.0, self._float_cfg("ucb_c", default=1.4))
        return max(
            self.arms,
            key=lambda arm: self._ucb_score(
                self._stat(context_key=context_key, arm_id=arm.arm_id, rows=rows),
                total_pulls=total,
                c=c,
            ),
        )

    def _ucb_score(self, stat: ArmStat, *, total_pulls: int, c: float) -> float:
        """Return a UCB score for one arm stat."""
        if stat.pulls <= 0:
            return float("inf")
        return stat.mean + c * math.sqrt(
            math.log(max(1, total_pulls)) / float(stat.pulls)
        )

    def _decay_factor(self) -> float:
        """Return a safe evidence decay factor from config."""
        return max(
            0.0,
            min(1.0, self._float_cfg("decay_factor", default=1.0)),
        )

    def _pseudo_success(self, *, reward: float, success: bool) -> float:
        """Return bounded posterior credit for one observation."""
        pseudo_success = reward_to_beta_update(reward)
        if success:
            return pseudo_success
        failure_cap = self._float_cfg("failure_pseudo_success_cap", default=0.45)
        return min(pseudo_success, max(0.0, min(0.49, failure_cap)))

    def _float_cfg(self, name: str, *, default: float) -> float:
        """Return one config value coerced to float."""
        value = getattr(self.cfg, name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


def _zero_stat(context_key: str, arm_id: str) -> ArmStat:
    """Return a default stat for an unseen ``(context_key, arm_id)``."""
    return ArmStat(
        context_key=context_key,
        arm_id=arm_id,
        alpha=1.0,
        beta=1.0,
        pulls=0,
        cumulative_reward=0.0,
        last_reward=0.0,
        last_success=False,
    )


def _row_to_stat(row: sqlite3.Row) -> ArmStat:
    """Convert a SQLite row to an ``ArmStat``."""
    return ArmStat(
        context_key=str(row["context_key"]),
        arm_id=str(row["arm_id"]),
        alpha=float(row["alpha"]),
        beta=float(row["beta"]),
        pulls=int(row["pulls"]),
        cumulative_reward=float(row["cumulative_reward"]),
        last_reward=float(row["last_reward"]),
        last_success=bool(row["last_success"]),
    )


def _context_has_previous_runtime_failure(context_key: str) -> bool:
    """Return whether the compact context reports a prior runtime failure."""
    return "|pf:runtime" in context_key.lower()

def _context_arm_boosts(context_key: str) -> dict[str, float]:
    """Return per-arm sample boosts derived from context key features.

    Boosts are small positive offsets added to Thompson samples before
    comparison. They do not override learned posteriors — they only
    give context-relevant arms a mild head-start during cold exploration.
    """
    boosts: dict[str, float] = {}
    key = context_key.lower()

    if "general_rl" in key or "actor_critic" in key:
        boosts["direct.long_stable"] = 0.15
        boosts["direct.balanced"] = 0.08

    if "csv" in key or "visual" in key:
        boosts["direct.long_stable"] = boosts.get("direct.long_stable", 0.0) + 0.08
        boosts["direct.balanced"] = boosts.get("direct.balanced", 0.0) + 0.05

    if "sparse_reward" in key or "boundary" in key:
        boosts["direct.cold_precise"] = 0.10

    if "timeout" in key or "stress" in key:
        boosts["direct.repeat_guard"] = 0.10

    if "|pf:runtime" in key:
        boosts["direct.cold"] = 0.12
        boosts["direct.cool"] = 0.08

    return boosts

__all__ = ["ArmStat", "BanditStore", "ThompsonBandit"]
