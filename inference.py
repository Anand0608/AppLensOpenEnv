"""
Baseline inference script for AppLensOpenEnv.

Runs an LLM (via OpenAI-compatible API) against the environment for all three
tasks (easy, medium, hard) and emits structured [START]/[STEP]/[END] logs.

Environment variables:
    API_BASE_URL   LLM endpoint         (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier     (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       API key / HF token
    TASK_NAME      If set, run only this task: easy | medium | hard

Output format (one block per task):
    [START] task=<name> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""
from __future__ import annotations

import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from env import AppLensOpenEnv
from models import Action

IMAGE_NAME = os.getenv("IMAGE_NAME")  # used when pulling from a Docker image
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy"

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("APP_LENS_BENCHMARK", "app_lens_openenv")
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.5  # normalized score in [0, 1]

TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI assessment agent operating within AppLensOpenEnv.
    You must execute a sequence of analysis actions on a given Git repository.
    The available actions are:
    - detect_language
    - calculate_loc
    - parse_dependencies
    - compute_complexity
    - security_scan
    - recommend_modernization
    - generate_report

    You must complete all required actions to succeed. Do not repeat actions that are already completed.
    Reply with exactly one action string — no quotes, no prefixes, just the action name.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, required_actions: List[str], completed_actions: List[str]) -> str:
    completed_str = ", ".join(completed_actions) if completed_actions else "none"
    return textwrap.dedent(
        f"""
        Step: {step}
        Required Actions: {', '.join(required_actions)}
        Completed Actions: {completed_str}

        Send your next action.
        """
    ).strip()


def _fallback_action(required_actions: List[str], completed_actions: List[str]) -> str:
    """Sequentially pick the first incomplete required action."""
    for act in required_actions:
        if act not in completed_actions:
            return act
    return "detect_language"


def get_model_action(
    client: OpenAI,
    step: int,
    required_actions: List[str],
    completed_actions: List[str],
) -> str:
    user_prompt = build_user_prompt(step, required_actions, completed_actions)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else _fallback_action(required_actions, completed_actions)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return _fallback_action(required_actions, completed_actions)


def _compute_score(obs) -> float:
    """Score strictly in (0.0, 1.0) = fraction of required actions completed.

    The hackathon grader requires the score to be strictly between 0 and 1
    (exclusive).  We clamp to [0.01, 0.99] so a perfect run returns 0.99
    and a zero-action run returns 0.01.
    """
    required = set(obs.required_actions)
    if not required:
        return 0.01
    completed_required = sum(1 for a in obs.completed_actions if a in required)
    raw = completed_required / len(required)
    # Strictly open interval (0, 1) — clamp away from exact 0.0 and 1.0.
    return round(max(0.01, min(0.99, raw)), 4)


def run_task(client: OpenAI, task_name: str) -> None:
    """Run a single task episode and emit [START]/[STEP]/[END] logs."""
    os.environ["TASK_NAME"] = task_name
    env = AppLensOpenEnv()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    obs = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()

        for step in range(1, obs.max_steps + 1):
            action_name = get_model_action(
                client, step, obs.required_actions, obs.completed_actions
            )
            obs = env.step(Action(action=action_name))

            delta = obs.reward
            done = obs.done
            error = obs.error

            rewards.append(delta)
            steps_taken = step

            log_step(step=step, action=action_name, reward=delta, done=done, error=error)

            if done:
                break

        # Score = fraction of required actions completed; strictly in [0.0, 1.0].
        score = _compute_score(obs) if obs is not None else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_name!r} raised: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    task_filter = os.getenv("TASK_NAME", "").strip()
    tasks_to_run = [task_filter] if task_filter in TASKS else TASKS

    for task_name in tasks_to_run:
        run_task(client, task_name)


if __name__ == "__main__":
    main()
