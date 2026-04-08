from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from scanner.dependency_extractor import extract_dependencies


# File extensions considered as source code, mapped to language names.
EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    ".py": "python",
    ".java": "java",
    ".cs": "csharp",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".rb": "ruby",
    ".php": "php",
    ".cpp": "cpp",
    ".c": "c",
    ".rs": "rust",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".vb": "vb",
}

# Directories to skip during file walk.
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "bin", "obj", "build", "dist", "target", ".idea", ".vs", ".vscode",
    "packages", ".gradle", ".mvn", "vendor",
}

# Max files to include in the metadata (keeps JSON manageable).
MAX_FILES = 500

# Allowed URL schemes
_ALLOWED_SCHEMES = {"https", "http"}

# Regex to loosely validate a Git remote URL (https only for safety).
_GIT_URL_PATTERN = re.compile(
    r"^https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+\.git$"
    r"|^https?://(?:github\.com|dev\.azure\.com|gitlab\.com|bitbucket\.org)/[^\s]+$"
)


def _validate_repo_url(url: str) -> None:
    """Basic validation to allow only public HTTPS git URLs."""
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"Only HTTPS URLs are supported. Got scheme: {parsed.scheme!r}")
    if not parsed.netloc:
        raise ValueError(f"Invalid URL — no host found: {url!r}")


def _derive_app_name(url: str) -> str:
    """Extracts a human-friendly app name from the repo URL."""
    path = urlparse(url).path.rstrip("/")
    name = path.rsplit("/", 1)[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name or "unknown_app"


def clone_repo(url: str, dest: Path, *, depth: int = 1) -> None:
    """Shallow-clones a public Git repo into *dest*."""
    _validate_repo_url(url)
    subprocess.run(
        ["git", "clone", "--depth", str(depth), "--single-branch", url, str(dest)],
        check=True,
        capture_output=True,
        timeout=120,
    )


def _count_lines(path: Path) -> int:
    """Counts non-blank lines in a text file (best-effort)."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return sum(1 for line in text.splitlines() if line.strip())
    except Exception:
        return 0


def _detect_legacy_markers(repo_root: Path) -> bool:
    """Heuristic: flag legacy if known legacy config files are present."""
    legacy_markers = [
        "web.config",          # classic ASP.NET / IIS
        "global.asax",         # ASP.NET WebForms
        "struts-config.xml",   # Apache Struts (legacy Java)
        "build.xml",           # Ant-based Java (usually legacy)
        "setup.cfg",           # may indicate old Python packaging
    ]
    for marker in legacy_markers:
        if list(repo_root.rglob(marker)):
            return True
    return False


def scan_repo(repo_root: Path) -> Dict[str, Any]:
    """Walks a cloned repo and produces the app-data dict the environment expects.

    Returns a dict with keys: id, name, language, legacy, dependencies, files.
    """
    lang_loc: Dict[str, int] = {}
    file_records: List[Dict[str, Any]] = []

    for file_path in sorted(repo_root.rglob("*")):
        if not file_path.is_file():
            continue
        # Skip ignored directories.
        parts = file_path.relative_to(repo_root).parts
        if any(part in SKIP_DIRS for part in parts):
            continue

        ext = file_path.suffix.lower()
        lang = EXTENSION_TO_LANGUAGE.get(ext)
        if lang is None:
            continue

        loc = _count_lines(file_path)
        lang_loc[lang] = lang_loc.get(lang, 0) + loc

        if len(file_records) < MAX_FILES:
            file_records.append({
                "path": str(file_path.relative_to(repo_root)),
                "loc": loc,
            })

    # Primary language = the one with the most LOC.
    primary_language = max(lang_loc, key=lang_loc.get) if lang_loc else "unknown"

    dependencies = extract_dependencies(repo_root)
    legacy = _detect_legacy_markers(repo_root)

    return {
        "id": repo_root.name,
        "name": repo_root.name,
        "language": primary_language,
        "legacy": legacy,
        "dependencies": dependencies,
        "files": file_records,
    }


def scan_from_url(url: str, *, keep_clone: bool = False) -> Dict[str, Any]:
    """Clone a public repo URL, scan it, and return app-data dict.

    Args:
        url: Public HTTPS Git URL.
        keep_clone: If True the cloned directory is *not* deleted (useful for
                    debugging).  The path is returned under key ``_clone_path``.
    """
    _validate_repo_url(url)
    tmp_dir = Path(tempfile.mkdtemp(prefix="applens_"))
    clone_dest = tmp_dir / _derive_app_name(url)

    try:
        clone_repo(url, clone_dest)
        app_data = scan_repo(clone_dest)
        app_data["name"] = _derive_app_name(url)
        app_data["id"] = app_data["name"]
        app_data["_source_url"] = url
        if keep_clone:
            app_data["_clone_path"] = str(clone_dest)
        return app_data
    finally:
        if not keep_clone:
            shutil.rmtree(tmp_dir, ignore_errors=True)
