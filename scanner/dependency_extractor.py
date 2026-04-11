from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set
from xml.etree import ElementTree


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

def _parse_requirements_txt(path: Path) -> List[str]:
    """Extracts package names from requirements.txt / requirements*.txt."""
    deps: List[str] = []
    for line in _read_text(path).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # Strip version specifiers: e.g. "flask>=2.0" → "flask"
        name = re.split(r"[>=<!~;\[]", line, maxsplit=1)[0].strip()
        if name:
            deps.append(name.lower())
    return deps


def _parse_setup_cfg(path: Path) -> List[str]:
    """Best-effort extraction from setup.cfg install_requires."""
    deps: List[str] = []
    in_install = False
    for line in _read_text(path).splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("install_requires"):
            in_install = True
            continue
        if in_install:
            if stripped and not stripped.startswith("[") and not stripped.startswith("#"):
                name = re.split(r"[>=<!~;\[]", stripped, maxsplit=1)[0].strip()
                if name:
                    deps.append(name.lower())
            elif stripped.startswith("["):
                break
    return deps


def _parse_pyproject_toml(path: Path) -> List[str]:
    """Rough extraction of dependencies from pyproject.toml without a TOML parser."""
    deps: List[str] = []
    in_deps = False
    for line in _read_text(path).splitlines():
        stripped = line.strip()
        if "dependencies" in stripped and "=" in stripped:
            in_deps = True
            continue
        if in_deps:
            if stripped.startswith("]"):
                in_deps = False
                continue
            match = re.match(r'["\']([a-zA-Z0-9_\-]+)', stripped)
            if match:
                deps.append(match.group(1).lower())
    return deps


# ---------------------------------------------------------------------------
# JavaScript / TypeScript
# ---------------------------------------------------------------------------

def _parse_package_json(path: Path) -> List[str]:
    deps: List[str] = []
    try:
        data: Dict[str, Any] = json.loads(_read_text(path))
    except (json.JSONDecodeError, ValueError):
        return deps
    for section_key in ("dependencies", "devDependencies"):
        section = data.get(section_key, {})
        if isinstance(section, dict):
            deps.extend(name.lower() for name in section)
    return deps


# ---------------------------------------------------------------------------
# Java / Kotlin (Maven pom.xml)
# ---------------------------------------------------------------------------

_MAVEN_NS = {"m": "http://maven.apache.org/POM/4.0.0"}


def _parse_pom_xml(path: Path) -> List[str]:
    deps: List[str] = []
    try:
        tree = ElementTree.parse(path)  # noqa: S314 — trusted local file
    except Exception:
        return deps

    root = tree.getroot()
    # Handle both namespaced and non-namespaced POMs.
    for dep in root.iter():
        tag = dep.tag.split("}")[-1] if "}" in dep.tag else dep.tag
        if tag == "dependency":
            artifact = None
            for child in dep:
                child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if child_tag == "artifactId" and child.text:
                    artifact = child.text.strip().lower()
            if artifact:
                deps.append(artifact)
    return deps


def _parse_build_gradle(path: Path) -> List[str]:
    """Rough regex extraction from build.gradle / build.gradle.kts."""
    deps: List[str] = []
    text = _read_text(path)
    # Matches patterns like:  implementation 'group:artifact:version'
    for match in re.finditer(r"['\"]([^'\"]+?):([^'\"]+?):([^'\"]*?)['\"]", text):
        artifact = match.group(2).strip().lower()
        if artifact:
            deps.append(artifact)
    return deps


# ---------------------------------------------------------------------------
# .NET (C# / VB)
# ---------------------------------------------------------------------------

def _parse_csproj(path: Path) -> List[str]:
    deps: List[str] = []
    try:
        tree = ElementTree.parse(path)  # noqa: S314
    except Exception:
        return deps
    for ref in tree.iter():
        tag = ref.tag.split("}")[-1] if "}" in ref.tag else ref.tag
        if tag == "PackageReference":
            name = ref.get("Include") or ref.get("include")
            if name:
                deps.append(name.strip().lower())
    return deps


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------

def _parse_go_mod(path: Path) -> List[str]:
    deps: List[str] = []
    in_require = False
    for line in _read_text(path).splitlines():
        stripped = line.strip()
        if stripped.startswith("require ("):
            in_require = True
            continue
        if in_require:
            if stripped == ")":
                in_require = False
                continue
            parts = stripped.split()
            if parts:
                deps.append(parts[0].lower())
        elif stripped.startswith("require "):
            parts = stripped.split()
            if len(parts) >= 2:
                deps.append(parts[1].lower())
    return deps


# ---------------------------------------------------------------------------
# Ruby
# ---------------------------------------------------------------------------

def _parse_gemfile(path: Path) -> List[str]:
    deps: List[str] = []
    for line in _read_text(path).splitlines():
        match = re.match(r"^\s*gem\s+['\"]([^'\"]+)['\"]", line)
        if match:
            deps.append(match.group(1).strip().lower())
    return deps


# ---------------------------------------------------------------------------
# PHP
# ---------------------------------------------------------------------------

def _parse_composer_json(path: Path) -> List[str]:
    deps: List[str] = []
    try:
        data: Dict[str, Any] = json.loads(_read_text(path))
    except (json.JSONDecodeError, ValueError):
        return deps
    for section_key in ("require", "require-dev"):
        section = data.get(section_key, {})
        if isinstance(section, dict):
            deps.extend(name.lower() for name in section if name.lower() != "php")
    return deps


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

# Map of filename (or globbed name) → parser function.
_PARSERS: Dict[str, Any] = {
    "requirements.txt": _parse_requirements_txt,
    "setup.cfg": _parse_setup_cfg,
    "pyproject.toml": _parse_pyproject_toml,
    "package.json": _parse_package_json,
    "pom.xml": _parse_pom_xml,
    "build.gradle": _parse_build_gradle,
    "build.gradle.kts": _parse_build_gradle,
    "go.mod": _parse_go_mod,
    "Gemfile": _parse_gemfile,
    "composer.json": _parse_composer_json,
}

# .csproj / .vbproj use suffix matching.
_SUFFIX_PARSERS: Dict[str, Any] = {
    ".csproj": _parse_csproj,
    ".vbproj": _parse_csproj,
    ".fsproj": _parse_csproj,
}


def extract_dependencies(repo_root: Path) -> List[str]:
    """Walks the repo and extracts dependency names from all known manifest files."""
    all_deps: Set[str] = set()

    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        # Skip vendored / build dirs.
        parts = path.relative_to(repo_root).parts
        skip_dirs = {".git", "node_modules", "__pycache__", "vendor", "build", "dist", "target", ".gradle"}
        if any(part in skip_dirs for part in parts):
            continue

        name = path.name
        if name in _PARSERS:
            all_deps.update(_PARSERS[name](path))
            continue

        suffix = path.suffix.lower()
        if suffix in _SUFFIX_PARSERS:
            all_deps.update(_SUFFIX_PARSERS[suffix](path))

    # Also pick up requirements-*.txt pattern.
    for req_file in sorted(repo_root.rglob("requirements*.txt")):
        parts = req_file.relative_to(repo_root).parts
        skip_dirs_req = {".git", "node_modules", "vendor"}
        if any(part in skip_dirs_req for part in parts):
            continue
        all_deps.update(_parse_requirements_txt(req_file))

    return sorted(all_deps)
