"""Run a trained PPO agent on a real Git repository.

The trained model (from agent/train.py) is loaded and used to decide the
action sequence.  The chosen actions are executed against the real
AppLensOpenEnv (which clones and analyses the repo), so every action
produces actual analysis results.

Usage
-----
python agent/run_agent.py https://github.com/pallets/flask.git
python agent/run_agent.py https://github.com/pallets/flask.git --model agent/models/ppo_applens
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from stable_baselines3 import PPO

from agent.mock_env import REQUIRED_ACTIONS, MAX_STEPS, _encode_obs
from env import AppLensOpenEnv
from models import Action

DEFAULT_MODEL_PATH = str(ROOT_DIR / "agent" / "models" / "ppo_applens")


def run_agent(repo_url: str, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    model_path = Path(model_path)
    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}.zip\n"
            "Run:  python agent/train.py  first."
        )

    print(f"\n  AppLens RL Agent — Live Inference")
    print(f"  Repository  : {repo_url}")
    print(f"  Model       : {model_path}.zip")
    print()

    model = PPO.load(str(model_path))

    real_env = AppLensOpenEnv()
    observation = real_env.reset(repo_url)

    print(f"  App ID          : {observation.app_id}")
    print(f"  Data confidence : {observation.data_confidence:.4f}  ({int(observation.data_confidence * 100)}%)")
    print(f"  Fetch reward    : +{observation.fetch_reward:.4f}  (seeded into total reward)")
    print()
    completed: set[str] = set()
    total_reward = 0.0

    for step in range(MAX_STEPS):
        obs_vec = _encode_obs(completed, step, MAX_STEPS)
        action_idx, _ = model.predict(obs_vec, deterministic=True)
        action_name = REQUIRED_ACTIONS[int(action_idx)]

        observation = real_env.step(Action(action=action_name))
        total_reward = observation.reward

        status = "✓" if not observation.error else observation.error
        print(f"  step {step + 1:>2}  [{action_name:<28}]  delta={observation.reward:+.4f}  {status}")

        if not observation.error:
            completed.add(action_name)

        if observation.done:
            break

    print(f"\n  Completed : {', '.join(sorted(completed))}")
    print(f"  Reward    : {round(total_reward, 4)}")
    print()
    return {
        "repo_url": repo_url,
        "completed_actions": sorted(completed),
        "total_reward": round(total_reward, 4),
        "results": observation.results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained PPO agent on a real repo.")
    parser.add_argument("repo_url", help="Public Git repository URL")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to trained model (no .zip)")
    args = parser.parse_args()
    run_agent(repo_url=args.repo_url, model_path=args.model)


if __name__ == "__main__":
    main()
