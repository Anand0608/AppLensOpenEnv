"""Analyse a live Git repository by URL.

Usage:
    python baseline/run_analysis.py <REPO_URL>

Example:
    python baseline/run_analysis.py https://github.com/pallets/flask.git
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on path.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env import AppLensOpenEnv
from models import Action


# Full analysis pipeline in the correct dependency order.
ACTION_SEQUENCE = [
    "detect_language",
    "calculate_loc",
    "parse_dependencies",
    "compute_complexity",
    "security_scan",
    "recommend_modernization",
    "generate_report",
]


def analyse_repo(repo_url: str) -> dict:
    env = AppLensOpenEnv()
    observation = env.reset(repo_url)

    print(f"  App ID          : {observation.app_id}")
    print(f"  Required        : {', '.join(observation.required_actions)}")
    print(f"  Max steps       : {observation.max_steps}")
    print(f"  Data confidence : {observation.data_confidence:.4f}  ({int(observation.data_confidence * 100)}%)")
    print(f"  Fetch reward    : +{observation.fetch_reward:.4f}  (seeded into total reward)")
    print()

    for action_name in ACTION_SEQUENCE:
        if observation.done:
            break
        action = Action(action=action_name)
        observation = env.step(action)
        status = "OK" if not observation.error else observation.error
        print(f"  step  {observation.step_count:>2}  {action_name:<28}  reward_delta={observation.reward:+.4f}  {status}")

    return {
        "app_id": observation.app_id,
        "steps": observation.step_count,
        "reward_total": observation.fetch_reward,
        "completed_actions": observation.completed_actions,
        "results": observation.results,
        "data_confidence": observation.data_confidence,
        "fetch_reward": observation.fetch_reward,
    }


def _pretty_section(title: str, data: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(json.dumps(data, indent=2, default=str))


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python baseline/run_analysis.py <REPO_URL>")
        print("Example: python baseline/run_analysis.py https://github.com/pallets/flask.git")
        sys.exit(1)

    repo_url = sys.argv[1]
    print(f"\n  AppLens OpenEnv — Live Repo Analysis")
    print(f"  URL: {repo_url}\n")
    print("  Cloning and scanning repository...")
    print()

    result = analyse_repo(repo_url)

    # Print each analysis section.
    results = result["results"]
    if "detect_language" in results:
        _pretty_section("Language Detection", results["detect_language"])
    if "calculate_loc" in results:
        _pretty_section("Lines of Code", results["calculate_loc"])
    if "parse_dependencies" in results:
        _pretty_section("Dependencies", results["parse_dependencies"])
    if "compute_complexity" in results:
        _pretty_section("Complexity Analysis", results["compute_complexity"])
    if "security_scan" in results:
        _pretty_section("Security Scan", results["security_scan"])
    if "recommend_modernization" in results:
        _pretty_section("Modernization Recommendations", results["recommend_modernization"])
    if "generate_report" in results:
        _pretty_section("Consolidated Report", results["generate_report"])

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total Steps     : {result['steps']}")
    print(f"  Data Confidence : {result['data_confidence']:.4f}  ({int(result['data_confidence'] * 100)}%)")
    print(f"  Fetch Reward    : +{result['fetch_reward']:.4f}")
    print(f"  Reward Total    : {result['reward_total']}")
    print(f"  Actions Done    : {', '.join(result['completed_actions'])}")
    print()


if __name__ == "__main__":
    main()
