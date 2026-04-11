from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from engine.action_router import ActionRouter
from engine.reward_engine import RewardEngine
from engine.state_manager import StateManager
from models import Action, Observation, Reward
from scanner.confidence import compute_confidence, compute_fetch_reward
from scanner.repo_scanner import scan_from_url

# Full analysis pipeline in the correct dependency order.
_REQUIRED_ACTIONS = [
    "detect_language",
    "calculate_loc",
    "parse_dependencies",
    "compute_complexity",
    "security_scan",
    "recommend_modernization",
    "generate_report",
]


from openenv.core import Environment

class AppLensOpenEnv(Environment):
    """OpenEnv-compatible environment for app assessment and modernization analysis.

    Accepts a public Git repo URL, clones it, scans the source tree, and
    exposes the full analysis pipeline through reset() / step() / state().

    Class-level state is used so that openenv-core's per-request instantiation
    (simulation mode creates a new env for each /reset and /step HTTP call)
    still sees persistent state across calls.
    """

    # openenv-core must not create more than one concurrent session.
    SUPPORTS_CONCURRENT_SESSIONS: ClassVar[bool] = False

    # Shared across all instances — survives per-request instantiation.
    _action_router: ClassVar[ActionRouter] = ActionRouter()
    _reward_engine: ClassVar[RewardEngine] = RewardEngine()
    _state_manager: ClassVar[StateManager] = StateManager()
    _vulnerabilities: ClassVar[Optional[Dict[str, Any]]] = None
    _base_dir: ClassVar[Path] = Path(__file__).resolve().parent

    @classmethod
    def _ensure_vulnerabilities(cls) -> Dict[str, Any]:
        if cls._vulnerabilities is None:
            path = cls._base_dir / "data" / "vulnerabilities.json"
            with path.open("r", encoding="utf-8") as f:
                cls._vulnerabilities = json.load(f)
        return cls._vulnerabilities

    def __init__(self) -> None:
        super().__init__()
        # These remain as instance attribute aliases to the class-level objects.
        self.action_router = self.__class__._action_router
        self.reward_engine = self.__class__._reward_engine
        self.state_manager = self.__class__._state_manager
        self.vulnerabilities = self.__class__._ensure_vulnerabilities()
        self.base_dir = self.__class__._base_dir

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        repo_url: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Clone and scan a public Git repo, then return initial observation.

        Args:
            seed: Unused — accepted for openenv-core base-class compatibility.
            episode_id: Unused — accepted for openenv-core base-class compatibility.
            repo_url: Public HTTPS Git URL (optional; defaults to task preset).
        """
        import os
        task_name = os.getenv("TASK_NAME", "easy").lower()

        if task_name == "easy":
            default_repo = "https://github.com/pallets/itsdangerous.git"
            req_actions = ["detect_language", "calculate_loc"]
            task_level = "easy"
        elif task_name == "medium":
            default_repo = "https://github.com/pallets/click.git"
            req_actions = ["detect_language", "calculate_loc", "parse_dependencies", "compute_complexity"]
            task_level = "medium"
        else:  # hard
            default_repo = "https://github.com/pallets/flask.git"
            req_actions = _REQUIRED_ACTIONS.copy()
            task_level = "hard"

        target_repo_url = repo_url if repo_url else default_repo

        app_data = scan_from_url(target_repo_url)

        # Score how complete/rich the fetched data is.
        data_confidence = compute_confidence(app_data)
        fetch_reward = compute_fetch_reward(data_confidence)

        task_config = {
            "id": f"{task_name}_{app_data['id']}",
            "description": f"Analysis of {target_repo_url} - {task_level}",
            "required_actions": req_actions,
            "max_steps": len(req_actions) + 5,
        }
        self.state_manager.reset(
            task_id=task_config["id"],
            task_level=task_level,
            app_data=app_data,
            required_actions=task_config["required_actions"],
            available_actions=self.action_router.available_actions,
            max_steps=task_config["max_steps"],
            data_confidence=data_confidence,
            fetch_reward=fetch_reward,
        )
        obs = self.state_manager.to_observation()
        return obs.model_copy(update={"reward": fetch_reward, "done": False})

    def state(self) -> Observation:
        """Returns current observation-style state snapshot."""
        if not self.state_manager.get_state():
            return Observation()
        return self.state_manager.to_observation()

    def close(self) -> None:
        """Clean up resources. Required by openenv-core POST /reset handler."""
        pass

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Executes one action in the environment.

        Returns an Observation with ``reward`` (delta for this step) and
        ``done`` embedded, as required by openenv-core's HTTP server.
        """
        if not self.state_manager.get_state():
            raise RuntimeError("Environment not initialized. Call reset() before step().")

        if self.state_manager.is_done():
            obs = self.state_manager.to_observation()
            return obs.model_copy(update={
                "reward": 0.0,
                "done": True,
                "error": "Episode already finished",
            })

        state = self.state_manager.get_state()
        action_name = action.action.strip().lower()
        is_valid = action_name in state["available_actions"]
        is_repeated = self.state_manager.is_action_completed(action_name)

        reward = self.reward_engine.compute(
            total_reward=state["total_reward"],
            action_name=action_name,
            is_valid=is_valid,
            is_repeated=is_repeated,
            required_actions=state["required_actions"],
        )

        self.state_manager.increment_step()
        new_total = self.state_manager.add_reward(reward.delta)
        reward = Reward(delta=reward.delta, total=new_total, reason=reward.reason, penalties=reward.penalties)

        error_msg: Optional[str] = None

        if not is_valid:
            self.state_manager.add_invalid_action()
            error_msg = f"Unsupported action: {action_name}"
        elif is_repeated:
            self.state_manager.add_repeated_action()
            error_msg = f"Action already completed: {action_name}"
        else:
            current_state = self.state_manager.get_state()
            valid, result = self.action_router.run(
                action_name=action_name,
                app_data=current_state["app_data"],
                context=current_state["results"],
                vulnerabilities=self.vulnerabilities,
            )
            if valid:
                self.state_manager.set_result(action_name, result)
                self.state_manager.add_completed_action(action_name)

        current_state = self.state_manager.get_state()
        required_done = all(
            a in current_state["completed_actions"]
            for a in current_state["required_actions"]
        )
        step_limit_reached = current_state["step_count"] >= current_state["max_steps"]

        done = required_done or step_limit_reached
        if done:
            self.state_manager.mark_done()

        obs = self.state_manager.to_observation()
        return obs.model_copy(update={
            "reward": round(reward.delta, 4),
            "done": done,
            "error": error_msg,
        })

        """Clone and scan a public Git repo, then return initial observation.

        Args:
            seed: Unused — accepted for openenv-core base-class compatibility.
            episode_id: Unused — accepted for openenv-core base-class compatibility.
            repo_url: Public HTTPS Git URL (optional; defaults to task preset).
        """
        import os
        task_name = os.getenv("TASK_NAME", "easy").lower()

        if task_name == "easy":
            default_repo = "https://github.com/pallets/itsdangerous.git"
            req_actions = ["detect_language", "calculate_loc"]
            task_level = "easy"
        elif task_name == "medium":
            default_repo = "https://github.com/pallets/click.git"
            req_actions = ["detect_language", "calculate_loc", "parse_dependencies", "compute_complexity"]
            task_level = "medium"
        else:  # hard
            default_repo = "https://github.com/pallets/flask.git"
            req_actions = _REQUIRED_ACTIONS.copy()
            task_level = "hard"

        target_repo_url = repo_url if repo_url else default_repo

        app_data = scan_from_url(target_repo_url)

        # Score how complete/rich the fetched data is.
        data_confidence = compute_confidence(app_data)
        fetch_reward = compute_fetch_reward(data_confidence)

        self.task_config = {
            "id": f"{task_name}_{app_data['id']}",
            "description": f"Analysis of {target_repo_url} - {task_level}",
            "required_actions": req_actions,
            "max_steps": len(req_actions) + 5,
        }
        self.state_manager.reset(
            task_id=self.task_config["id"],
            task_level=task_level,
            app_data=app_data,
            required_actions=self.task_config["required_actions"],
            available_actions=self.action_router.available_actions,
            max_steps=self.task_config["max_steps"],
            data_confidence=data_confidence,
            fetch_reward=fetch_reward,
        )
        obs = self.state_manager.to_observation()
        return obs.model_copy(update={"reward": fetch_reward, "done": False})

    def state(self) -> Observation:
        """Returns current observation-style state snapshot."""
        if not self.state_manager.get_state():
            return Observation()
        return self.state_manager.to_observation()

    def close(self) -> None:
        """Clean up resources. Required by openenv-core POST /reset handler."""
        pass

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Executes one action in the environment.

        Returns an Observation with ``reward`` (delta for this step) and
        ``done`` embedded, as required by openenv-core's HTTP server.
        """
        if not self.state_manager.get_state():
            raise RuntimeError("Environment not initialized. Call reset() before step().")

        if self.state_manager.is_done():
            current_state = self.state_manager.get_state()
            obs = self.state_manager.to_observation()
            return obs.model_copy(update={
                "reward": 0.0,
                "done": True,
                "error": "Episode already finished",
            })

        state = self.state_manager.get_state()
        action_name = action.action.strip().lower()
        is_valid = action_name in state["available_actions"]
        is_repeated = self.state_manager.is_action_completed(action_name)

        reward = self.reward_engine.compute(
            total_reward=state["total_reward"],
            action_name=action_name,
            is_valid=is_valid,
            is_repeated=is_repeated,
            required_actions=state["required_actions"],
        )

        self.state_manager.increment_step()
        new_total = self.state_manager.add_reward(reward.delta)
        reward = Reward(delta=reward.delta, total=new_total, reason=reward.reason, penalties=reward.penalties)

        error_msg: Optional[str] = None

        if not is_valid:
            self.state_manager.add_invalid_action()
            error_msg = f"Unsupported action: {action_name}"
        elif is_repeated:
            self.state_manager.add_repeated_action()
            error_msg = f"Action already completed: {action_name}"
        else:
            current_state = self.state_manager.get_state()
            valid, result = self.action_router.run(
                action_name=action_name,
                app_data=current_state["app_data"],
                context=current_state["results"],
                vulnerabilities=self.vulnerabilities,
            )
            if valid:
                self.state_manager.set_result(action_name, result)
                self.state_manager.add_completed_action(action_name)

        current_state = self.state_manager.get_state()
        required_done = all(
            a in current_state["completed_actions"]
            for a in current_state["required_actions"]
        )
        step_limit_reached = current_state["step_count"] >= current_state["max_steps"]

        done = required_done or step_limit_reached
        if done:
            self.state_manager.mark_done()

        obs = self.state_manager.to_observation()
        return obs.model_copy(update={
            "reward": round(reward.delta, 4),
            "done": done,
            "error": error_msg,
        })

