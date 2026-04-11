from __future__ import annotations

from models import Reward


class RewardEngine:
    """Deterministic incremental reward logic for step-by-step evaluation."""

    INVALID_ACTION_PENALTY = -0.15
    REPEATED_ACTION_PENALTY = -0.08
    OPTIONAL_ACTION_REWARD = 0.02

    def compute(
        self,
        *,
        total_reward: float,
        action_name: str,
        is_valid: bool,
        is_repeated: bool,
        required_actions: list[str],
    ) -> Reward:
        penalties: list[str] = []

        if not is_valid:
            delta = self.INVALID_ACTION_PENALTY
            penalties.append("invalid_action")
            reason = f"Invalid action: {action_name}"
        elif is_repeated:
            delta = self.REPEATED_ACTION_PENALTY
            penalties.append("repeated_action")
            reason = f"Repeated action: {action_name}"
        else:
            if action_name in required_actions:
                delta = 1.0 / max(1, len(required_actions))
                reason = f"Correct required action: {action_name}"
            else:
                delta = self.OPTIONAL_ACTION_REWARD
                reason = f"Valid optional action: {action_name}"

        new_total = round(max(0.0, total_reward + delta), 4)
        return Reward(delta=round(delta, 4), total=new_total, reason=reason, penalties=penalties)
