from __future__ import annotations

from typing import Any, Dict


def parse_dependencies(*, app_data: Dict[str, Any], context: Dict[str, Any], vulnerabilities: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes and returns dependency information."""
    dependencies = app_data.get("dependencies", [])
    normalized = sorted({str(dep).strip().lower() for dep in dependencies if str(dep).strip()})
    return {
        "dependencies": normalized,
        "count": len(normalized),
    }
