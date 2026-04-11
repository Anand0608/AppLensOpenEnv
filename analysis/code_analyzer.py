from __future__ import annotations

from typing import Any, Dict


def detect_language(*, app_data: Dict[str, Any], context: Dict[str, Any], vulnerabilities: Dict[str, Any]) -> Dict[str, Any]:
    """Detects language directly from app metadata or file extensions."""
    language = app_data.get("language")
    if language:
        return {"language": language.lower()}

    extension_to_language = {
        ".py": "python",
        ".java": "java",
        ".cs": "csharp",
        ".js": "javascript",
        ".ts": "typescript",
    }
    files = app_data.get("files", [])
    for file_info in files:
        path = str(file_info.get("path", "")).lower()
        for ext, inferred_language in extension_to_language.items():
            if path.endswith(ext):
                return {"language": inferred_language}

    return {"language": "unknown"}


def calculate_loc(*, app_data: Dict[str, Any], context: Dict[str, Any], vulnerabilities: Dict[str, Any]) -> Dict[str, Any]:
    """Computes total lines of code from file metadata."""
    files = app_data.get("files", [])
    total_loc = sum(int(file_info.get("loc", 0)) for file_info in files)
    return {
        "total_loc": total_loc,
        "file_count": len(files),
    }


def generate_report(*, app_data: Dict[str, Any], context: Dict[str, Any], vulnerabilities: Dict[str, Any]) -> Dict[str, Any]:
    """Builds a consolidated report from previously executed analysis actions."""
    language = context.get("detect_language", {}).get("language", "unknown")
    loc = context.get("calculate_loc", {}).get("total_loc", 0)
    deps = context.get("parse_dependencies", {}).get("dependencies", [])
    complexity = context.get("compute_complexity", {})
    security = context.get("security_scan", {})
    modernization = context.get("recommend_modernization", {})

    report = {
        "app_name": app_data.get("name", "Unknown Application"),
        "language": language,
        "total_loc": loc,
        "dependency_count": len(deps),
        "complexity_level": complexity.get("level", "unknown"),
        "vulnerability_count": security.get("vulnerability_count", 0),
        "modernization_priority": modernization.get("priority", "unknown"),
        "summary": (
            f"{app_data.get('name', 'Application')} in {language} has {loc} LOC, "
            f"{len(deps)} dependencies, complexity={complexity.get('level', 'unknown')}, "
            f"vulnerabilities={security.get('vulnerability_count', 0)}."
        ),
    }
    return {"report": report}
