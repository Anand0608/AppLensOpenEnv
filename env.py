from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

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


class AppLensOpenEnv:
    """OpenEnv-compatible environment for app assessment and modernization analysis.

    Accepts a public Git repo URL, clones it, scans the source tree, and
    exposes the full analysis pipeline through reset() / step() / state().
    """

    def __init__(self) -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.task_config: Dict[str, Any] = {}
        self.vulnerabilities = self._load_json(self.base_dir / "data" / "vulnerabilities.json")

        self.action_router = ActionRouter()
        self.reward_engine = RewardEngine()
        self.state_manager = StateManager()

    def _load_json(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def reset(self, repo_url: str) -> Observation:
        """Clone and scan a public Git repo, then return initial observation.

        Args:
            repo_url: Public HTTPS Git URL (GitHub, Azure Repos, GitLab, etc.).
        """
        app_data = scan_from_url(repo_url)

        # Score how complete/rich the fetched data is.
        data_confidence = compute_confidence(app_data)
        fetch_reward = compute_fetch_reward(data_confidence)

        self.task_config = {
            "id": f"repo_{app_data['id']}",
            "description": f"Full analysis of {repo_url}",
            "required_actions": list(_REQUIRED_ACTIONS),
            "max_steps": len(_REQUIRED_ACTIONS) + 5,
        }
        self.state_manager.reset(
            task_id=self.task_config["id"],
            task_level="analysis",
            app_data=app_data,
            required_actions=self.task_config["required_actions"],
            available_actions=self.action_router.available_actions,
            max_steps=self.task_config["max_steps"],
            data_confidence=data_confidence,
            fetch_reward=fetch_reward,
        )
        return self.state_manager.to_observation()

    def state(self) -> Observation:
        """Returns current observation-style state snapshot."""
        return self.state_manager.to_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Executes one action in the environment.

        Returns:
            (observation, reward, done, info)
        """
        if self.state_manager.get_state() == {}:
            raise RuntimeError("Environment not initialized. Call reset() before step().")

        if self.state_manager.is_done():
            current_state = self.state_manager.get_state()
            reward = Reward(
                delta=0.0,
                total=current_state["total_reward"],
                reason="Episode already finished",
                penalties=[],
            )
            return self.state_manager.to_observation(), reward, True, {"warning": "Episode already done"}

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

        info: Dict[str, Any] = {"action": action_name}

        if not is_valid:
            self.state_manager.add_invalid_action()
            info["error"] = f"Unsupported action: {action_name}"
        elif is_repeated:
            self.state_manager.add_repeated_action()
            info["warning"] = f"Action already completed: {action_name}"
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
                info["result"] = result

        current_state = self.state_manager.get_state()
        required_done = all(
            required_action in current_state["completed_actions"]
            for required_action in current_state["required_actions"]
        )
        step_limit_reached = current_state["step_count"] >= current_state["max_steps"]

        done = required_done or step_limit_reached
        if done:
            self.state_manager.mark_done()
            info["required_actions_completed"] = required_done
            info["step_limit_reached"] = step_limit_reached

        return self.state_manager.to_observation(), reward, done, info
