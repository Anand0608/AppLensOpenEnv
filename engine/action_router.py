from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from analysis.code_analyzer import calculate_loc, detect_language, generate_report
from analysis.complexity import compute_complexity
from analysis.dependency_analyzer import parse_dependencies
from analysis.modernization import recommend_modernization
from analysis.security import security_scan


class ActionRouter:
    """Routes action names to deterministic analyzer functions."""

    def __init__(self) -> None:
        self._routes: Dict[str, Callable[..., Dict[str, Any]]] = {
            "detect_language": detect_language,
            "calculate_loc": calculate_loc,
            "parse_dependencies": parse_dependencies,
            "compute_complexity": compute_complexity,
            "security_scan": security_scan,
            "recommend_modernization": recommend_modernization,
            "generate_report": generate_report,
        }

    @property
    def available_actions(self) -> list[str]:
        return list(self._routes.keys())

    def run(
        self,
        action_name: str,
        app_data: Dict[str, Any],
        context: Dict[str, Any],
        vulnerabilities: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        if action_name not in self._routes:
            return False, {"error": f"Unsupported action: {action_name}"}

        action_fn = self._routes[action_name]
        result = action_fn(app_data=app_data, context=context, vulnerabilities=vulnerabilities)
        return True, result
