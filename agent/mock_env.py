"""Mock Gymnasium environment for fast RL training.

Uses the real RewardEngine and StateManager from the project engine layer,
but skips all I/O (no git clone, no file parsing).  This makes episodes run
in microseconds so the agent can train over thousands of episodes quickly.

Observation  : float32 vector of length 8
               [binary flag per required action (7)] + [step / max_steps (1)]
Action space : Discrete(7)  — one integer per required action
Reward       : same as real env  (+1/7 per unique required action, penalties
               for invalid / repeated steps)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.reward_engine import RewardEngine
from engine.state_manager import StateManager

REQUIRED_ACTIONS: list[str] = [
    "detect_language",
    "calculate_loc",
    "parse_dependencies",
    "compute_complexity",
    "security_scan",
    "recommend_modernization",
    "generate_report",
]

MAX_STEPS = 12
N_ACTIONS = len(REQUIRED_ACTIONS)
OBS_DIM = N_ACTIONS + 1  # binary flags + step fraction


def _encode_obs(completed: set[str], step: int, max_steps: int) -> np.ndarray:
    flags = np.array(
        [1.0 if a in completed else 0.0 for a in REQUIRED_ACTIONS],
        dtype=np.float32,
    )
    step_frac = np.array([step / max(1, max_steps)], dtype=np.float32)
    return np.concatenate([flags, step_frac])


class MockAppLensEnv(gym.Env):
    """Fast mock version of AppLensOpenEnv compatible with Gymnasium & SB3."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self._reward_engine = RewardEngine()
        self._state_manager = StateManager()

    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._state_manager.reset(
            task_id="mock_episode",
            task_level="analysis",
            app_data={"id": "mock_app"},
            required_actions=list(REQUIRED_ACTIONS),
            available_actions=list(REQUIRED_ACTIONS),
            max_steps=MAX_STEPS,
        )
        obs = _encode_obs(set(), 0, MAX_STEPS)
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_name = REQUIRED_ACTIONS[int(action)]
        state = self._state_manager.get_state()

        is_valid = action_name in state["available_actions"]
        is_repeated = self._state_manager.is_action_completed(action_name)

        reward = self._reward_engine.compute(
            total_reward=state["total_reward"],
            action_name=action_name,
            is_valid=is_valid,
            is_repeated=is_repeated,
            required_actions=state["required_actions"],
        )

        self._state_manager.increment_step()
        self._state_manager.add_reward(reward.delta)

        if is_repeated:
            self._state_manager.add_repeated_action()
        else:
            self._state_manager.add_completed_action(action_name)

        state = self._state_manager.get_state()
        completed = set(state["completed_actions"])
        all_done = all(a in completed for a in REQUIRED_ACTIONS)
        step_limit = state["step_count"] >= MAX_STEPS

        terminated = all_done or step_limit
        truncated = False
        if terminated:
            self._state_manager.mark_done()

        obs = _encode_obs(completed, state["step_count"], MAX_STEPS)
        info: dict = {
            "completed_actions": list(completed),
            "step_count": state["step_count"],
            "total_reward": state["total_reward"],
        }
        return obs, float(reward.delta), terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self) -> None:  # noqa: D401
        pass
