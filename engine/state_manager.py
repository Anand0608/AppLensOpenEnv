from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from models import Observation


class StateManager:
    """Stores and exposes mutable environment state."""

    def __init__(self) -> None:
        self._state: Dict[str, Any] = {}

    def reset(
        self,
        *,
        task_id: str,
        task_level: str,
        app_data: Dict[str, Any],
        required_actions: List[str],
        available_actions: List[str],
        max_steps: int,
        data_confidence: float = 0.0,
        fetch_reward: float = 0.0,
    ) -> None:
        self._state = {
            "task_id": task_id,
            "task_level": task_level,
            "app_data": deepcopy(app_data),
            "required_actions": list(required_actions),
            "available_actions": list(available_actions),
            "completed_actions": [],
            "results": {},
            "step_count": 0,
            "max_steps": max_steps,
            "done": False,
            "invalid_actions": 0,
            "repeated_actions": 0,
            "total_reward": round(fetch_reward, 4),
            "data_confidence": round(data_confidence, 4),
            "fetch_reward": round(fetch_reward, 4),
        }

    def get_state(self) -> Dict[str, Any]:
        return deepcopy(self._state)

    def increment_step(self) -> None:
        self._state["step_count"] += 1

    def mark_done(self) -> None:
        self._state["done"] = True

    def is_done(self) -> bool:
        return bool(self._state.get("done", False))

    def is_action_completed(self, action_name: str) -> bool:
        return action_name in self._state["completed_actions"]

    def add_completed_action(self, action_name: str) -> None:
        if action_name not in self._state["completed_actions"]:
            self._state["completed_actions"].append(action_name)

    def set_result(self, action_name: str, result: Dict[str, Any]) -> None:
        self._state["results"][action_name] = result

    def add_invalid_action(self) -> None:
        self._state["invalid_actions"] += 1

    def add_repeated_action(self) -> None:
        self._state["repeated_actions"] += 1

    def add_reward(self, delta: float) -> float:
        self._state["total_reward"] += delta
        self._state["total_reward"] = round(max(0.0, self._state["total_reward"]), 4)
        return self._state["total_reward"]

    def to_observation(self) -> Observation:
        app_data = self._state["app_data"]
        return Observation(
            task_id=self._state["task_id"],
            task_level=self._state["task_level"],
            app_id=app_data.get("id", "unknown_app"),
            step_count=self._state["step_count"],
            max_steps=self._state["max_steps"],
            required_actions=list(self._state["required_actions"]),
            completed_actions=list(self._state["completed_actions"]),
            available_actions=list(self._state["available_actions"]),
            results=deepcopy(self._state["results"]),
            data_confidence=self._state.get("data_confidence", 0.0),
            fetch_reward=self._state.get("fetch_reward", 0.0),
        )
