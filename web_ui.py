from __future__ import annotations

import json
import os
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, render_template, request

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.workflow import (
    LATEST_ANALYSIS_FILE,
    REPORTS_DIR,
    build_document,
    ensure_dirs,
    run_analysis,
    send_mail,
    write_json,
)

app = Flask(__name__)

EMAIL_PATTERN = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
DEFAULT_REPO_URL = 'https://github.com/pallets/flask.git'


def _safe_name_from_url(repo_url: str) -> str:
    cleaned = repo_url.rstrip("/")
    tail = cleaned.split("/")[-1] if cleaned else "repo"
    return tail.replace(".git", "") or "repo"


def _send_report_in_background(email: str, analysis_data: dict[str, Any]) -> None:
    try:
        ensure_dirs()
        app_id = analysis_data.get("app_id") or _safe_name_from_url(analysis_data.get("repo_url", ""))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        unique_json = REPORTS_DIR / f"{app_id}-analysis-{timestamp}.json"
        write_json(analysis_data, unique_json)
        report_path = build_document(input_json=unique_json)
        send_mail(recipient=email, report_path=report_path)
    except Exception as exc:
        print(f"[web_ui] Background email delivery failed: {exc}")


def _build_summary(data: dict[str, Any]) -> dict[str, Any]:
    results = data.get("results", {})

    language = results.get("detect_language", {}).get("language", "unknown")
    loc = results.get("calculate_loc", {}).get("total_loc", 0)
    file_count = results.get("calculate_loc", {}).get("file_count", 0)
    dep_count = results.get("parse_dependencies", {}).get("count", 0)

    complexity = results.get("compute_complexity", {})
    security = results.get("security_scan", {})
    modernization = results.get("recommend_modernization", {})

    return {
        "repo_url": data.get("repo_url", ""),
        "app_id": data.get("app_id", ""),
        "language": language,
        "loc": loc,
        "file_count": file_count,
        "dep_count": dep_count,
        "complexity_level": complexity.get("level", "N/A"),
        "complexity_score": complexity.get("score", "N/A"),
        "vulnerability_count": security.get("vulnerability_count", 0),
        "critical_count": security.get("critical_count", 0),
        "priority": modernization.get("priority", "N/A"),
        "target_stack": modernization.get("target_stack", "N/A"),
        "effort_weeks": modernization.get("effort_weeks", "N/A"),
        "recommendations": modernization.get("recommendations", []),
    }


@app.get("/")
def index():
    return render_template(
        "index.html",
        error=None,
        message=None,
        result=None,
        email="",
        repo_url=DEFAULT_REPO_URL,
        send_report=False,
    )


@app.post("/analyze")
def analyze():
    email = request.form.get("email", "").strip()
    repo_url = request.form.get("repo_url", "").strip()
    send_report = request.form.get("send_report") == "on"

    if not EMAIL_PATTERN.match(email):
        return render_template(
            "index.html",
            error="Please enter a valid email address.",
            message=None,
            result=None,
            email=email,
            repo_url=repo_url or DEFAULT_REPO_URL,
            send_report=send_report,
        )

    if not repo_url:
        return render_template(
            "index.html",
            error="Please enter a repository URL.",
            message=None,
            result=None,
            email=email,
            repo_url=repo_url,
            send_report=send_report,
        )

    try:
        ensure_dirs()
        analysis_data = run_analysis(repo_url)
        write_json(analysis_data, LATEST_ANALYSIS_FILE)

        if send_report:
            worker = threading.Thread(
                target=_send_report_in_background,
                args=(email, json.loads(json.dumps(analysis_data, default=str))),
                daemon=True,
            )
            worker.start()
            message = (
                f"Analysis complete. Results are shown below. "
                f"Report email to {email} is being processed in the background."
            )
        else:
            message = "Analysis complete. Results are shown below."

        return render_template(
            "index.html",
            error=None,
            message=message,
            result=_build_summary(analysis_data),
            email=email,
            repo_url=repo_url,
            send_report=send_report,
        )
    except Exception as exc:
        return render_template(
            "index.html",
            error=f"Failed to process request: {exc}",
            message=None,
            result=None,
            email=email,
            repo_url=repo_url,
            send_report=send_report,
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
