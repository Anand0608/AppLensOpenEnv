"""Train a PPO agent on the MockAppLensEnv using a HuggingFace feature extractor.

Usage
-----
python agent/train.py
python agent/train.py --timesteps 50000 --save-path agent/models/ppo_applens
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from agent.mock_env import MockAppLensEnv
from agent.hf_extractor import HFDistilBertExtractor


DEFAULT_TIMESTEPS = 30_000
DEFAULT_SAVE_PATH = str(ROOT_DIR / "agent" / "models" / "ppo_applens")


def make_env():
    env = MockAppLensEnv()
    env = Monitor(env)
    return env


def train(timesteps: int = DEFAULT_TIMESTEPS, save_path: str = DEFAULT_SAVE_PATH) -> PPO:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("Checking environment compatibility …")
    check_env(MockAppLensEnv(), warn=True)

    print(f"Creating vectorised environment (4 parallel workers) …")
    vec_env = make_vec_env(make_env, n_envs=4)

    eval_env = Monitor(MockAppLensEnv())
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path.parent / "best"),
        log_path=str(save_path.parent / "logs"),
        eval_freq=2_000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )

    policy_kwargs = {
        "features_extractor_class": HFDistilBertExtractor,
        "features_extractor_kwargs": {
            "features_dim": 128,
            "freeze_bert": True,
        },
        "net_arch": [128, 64],
    }

    print(f"Building PPO model with HuggingFace DistilBERT feature extractor …")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=None,
    )

    print(f"\nTraining for {timesteps:,} timesteps …")
    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)

    model.save(str(save_path))
    print(f"\nModel saved → {save_path}.zip")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO + HuggingFace agent on AppLens env.")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS, help="Training timesteps")
    parser.add_argument("--save-path", default=DEFAULT_SAVE_PATH, help="Path to save trained model")
    args = parser.parse_args()
    train(timesteps=args.timesteps, save_path=args.save_path)


if __name__ == "__main__":
    main()
