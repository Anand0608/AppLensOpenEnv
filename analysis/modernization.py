from __future__ import annotations

from typing import Any, Dict, List


def recommend_modernization(*, app_data: Dict[str, Any], context: Dict[str, Any], vulnerabilities: Dict[str, Any]) -> Dict[str, Any]:
    """Builds deterministic modernization recommendations."""
    language = context.get("detect_language", {}).get("language", app_data.get("language", "unknown")).lower()
    complexity = context.get("compute_complexity", {})
    security = context.get("security_scan", {})

    complexity_level = complexity.get("level", "unknown")
    vulnerability_count = int(security.get("vulnerability_count", 0))
    critical_count = int(security.get("critical_count", 0))
    legacy = bool(app_data.get("legacy", False))

    recommendations: List[str] = []

    if legacy:
        recommendations.append("Migrate to a supported runtime/framework version")
    if complexity_level == "high":
        recommendations.append("Refactor into bounded modules before platform migration")
    if vulnerability_count > 0:
        recommendations.append("Upgrade vulnerable dependencies and enable SCA checks")
    if not recommendations:
        recommendations.append("Maintain current architecture and add observability guardrails")

    recommendations.append("Adopt CI/CD with automated tests and static analysis")

    if critical_count > 0 or (legacy and complexity_level == "high"):
        priority = "high"
    elif legacy or vulnerability_count > 0:
        priority = "medium"
    else:
        priority = "low"

    target_stack_map = {
        "python": "Python 3.11 + FastAPI",
        "java": "Java 21 + Spring Boot 3",
        "csharp": ".NET 8",
        "javascript": "Node.js 22 + TypeScript",
    }
    target_stack = target_stack_map.get(language, "Modern supported runtime")

    effort_weeks = (len(recommendations) * 2) + (4 if complexity_level == "high" else 1)

    return {
        "priority": priority,
        "target_stack": target_stack,
        "effort_weeks": effort_weeks,
        "recommendations": recommendations,
    }
