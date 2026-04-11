from __future__ import annotations

from typing import Any, Dict


def compute_complexity(*, app_data: Dict[str, Any], context: Dict[str, Any], vulnerabilities: Dict[str, Any]) -> Dict[str, Any]:
    """Computes deterministic complexity score from LOC, dependencies, and legacy flag."""
    if "calculate_loc" in context:
        total_loc = int(context["calculate_loc"].get("total_loc", 0))
    else:
        total_loc = sum(int(file_info.get("loc", 0)) for file_info in app_data.get("files", []))

    if "parse_dependencies" in context:
        dependency_count = int(context["parse_dependencies"].get("count", 0))
    else:
        dependency_count = len(app_data.get("dependencies", []))

    legacy_flag = bool(app_data.get("legacy", False))
    score = min(100, int((total_loc / 20) + (dependency_count * 4) + (15 if legacy_flag else 0)))

    if score < 35:
        level = "low"
    elif score < 70:
        level = "medium"
    else:
        level = "high"

    return {
        "score": score,
        "level": level,
        "inputs": {
            "total_loc": total_loc,
            "dependency_count": dependency_count,
            "legacy": legacy_flag,
        },
    }
