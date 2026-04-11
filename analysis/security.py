from __future__ import annotations

from typing import Any, Dict, List


def security_scan(*, app_data: Dict[str, Any], context: Dict[str, Any], vulnerabilities: Dict[str, Any]) -> Dict[str, Any]:
    """Scans dependencies against deterministic local vulnerability dataset."""
    if "parse_dependencies" in context:
        dependencies = context["parse_dependencies"].get("dependencies", [])
    else:
        dependencies = [str(dep).lower() for dep in app_data.get("dependencies", [])]

    vuln_map = vulnerabilities.get("vulnerable_dependencies", {})
    findings: List[Dict[str, str]] = []

    for dep in dependencies:
        entry = vuln_map.get(dep)
        if entry:
            findings.append(
                {
                    "dependency": dep,
                    "severity": entry.get("severity", "unknown"),
                    "cve": entry.get("cve", "N/A"),
                    "note": entry.get("note", "No detail provided"),
                }
            )

    findings.sort(key=lambda item: item["dependency"])
    critical_count = sum(1 for finding in findings if finding["severity"].lower() == "critical")

    return {
        "vulnerability_count": len(findings),
        "critical_count": critical_count,
        "findings": findings,
    }
