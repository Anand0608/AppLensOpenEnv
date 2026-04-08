"""Data-quality confidence scorer for scanned repositories.

Produces a float in [0.0, 1.0] that reflects how complete and rich
the app_data returned by scan_from_url() is.

Scoring breakdown (total = 1.0)
--------------------------------
  language_identified   0.20  — primary language is not "unknown"
  files_found           0.25  — at least one source file was discovered
  loc_volume            0.20  — scaled by file count (saturates at 50+ files)
  dependencies_found    0.25  — at least one dependency was extracted
  url_present           0.10  — source URL was recorded in app_data

The score is used in env.reset() to:
  • Populate Observation.data_confidence
  • Seed the initial Reward.total with a fetch_reward ∝ confidence
    so the RL agent is immediately rewarded for high-quality data.
"""
from __future__ import annotations

from typing import Any

# Weight of each dimension.
_WEIGHTS: dict[str, float] = {
    "language_identified": 0.20,
    "files_found":         0.25,
    "loc_volume":          0.20,
    "dependencies_found":  0.25,
    "url_present":         0.10,
}

_LOC_SATURATION_FILES = 50  # file count at which loc_volume score maxes out


def score_breakdown(app_data: dict[str, Any]) -> dict[str, float]:
    """Return per-dimension scores for a scanned app_data dict.

    Each value in the returned dict is in [0.0, 1.0].
    """
    language: str = app_data.get("language", "unknown") or "unknown"
    files: list = app_data.get("files", []) or []
    dependencies: list = app_data.get("dependencies", []) or []
    source_url: str = app_data.get("_source_url", "") or ""

    file_count = len(files)
    total_loc = sum(f.get("loc", 0) for f in files)

    # language_identified
    lang_score = 1.0 if language not in ("unknown", "", None) else 0.0

    # files_found — binary: at least one file
    files_score = 1.0 if file_count > 0 else 0.0

    # loc_volume — scaled: 0 → 0.0, ≥ saturation → 1.0
    loc_score = min(1.0, file_count / _LOC_SATURATION_FILES) if total_loc > 0 else 0.0

    # dependencies_found — binary: at least one dependency
    deps_score = 1.0 if len(dependencies) > 0 else 0.0

    # url_present — binary: non-empty source URL recorded
    url_score = 1.0 if source_url else 0.0

    return {
        "language_identified": lang_score,
        "files_found":         files_score,
        "loc_volume":          loc_score,
        "dependencies_found":  deps_score,
        "url_present":         url_score,
    }


def compute_confidence(app_data: dict[str, Any]) -> float:
    """Return a single weighted confidence score in [0.0, 1.0].

    Parameters
    ----------
    app_data:
        The dict returned by scanner.repo_scanner.scan_from_url().
    """
    breakdown = score_breakdown(app_data)
    total = sum(
        score * _WEIGHTS[dim]
        for dim, score in breakdown.items()
    )
    return round(min(1.0, max(0.0, total)), 4)


def compute_fetch_reward(confidence: float, max_reward: float = 0.25) -> float:
    """Translate data confidence into an initial fetch reward.

    The fetch reward is added to total_reward at reset() time to signal
    that successfully retrieving high-quality repo data is itself valuable.

    Parameters
    ----------
    confidence:
        Score in [0.0, 1.0] from compute_confidence().
    max_reward:
        Upper bound on the fetch reward (default 0.25).
    """
    return round(confidence * max_reward, 4)
