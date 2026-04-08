from __future__ import annotations

import argparse
import json
import os
import smtplib
import sys
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from env import AppLensOpenEnv
from models import Action

ACTION_SEQUENCE = [
    "detect_language",
    "calculate_loc",
    "parse_dependencies",
    "compute_complexity",
    "security_scan",
    "recommend_modernization",
    "generate_report",
]

DEFAULT_RL_MODEL_PATH = ROOT_DIR / "agent" / "models" / "ppo_applens"


def _rl_model_available() -> bool:
    return DEFAULT_RL_MODEL_PATH.with_suffix(".zip").exists()


def _load_rl_model():
    """Load the trained PPO model. Returns None if unavailable."""
    try:
        from stable_baselines3 import PPO
        return PPO.load(str(DEFAULT_RL_MODEL_PATH))
    except Exception:
        return None

ANALYSIS_DIR = ROOT_DIR / "analysis"
ARTIFACTS_DIR = ANALYSIS_DIR / "artifacts"
REPORTS_DIR = ANALYSIS_DIR / "reports"
LATEST_ANALYSIS_FILE = ARTIFACTS_DIR / "latest-analysis.json"
LATEST_REPORT_POINTER = ARTIFACTS_DIR / "latest-report-path.txt"


def ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_analysis(repo_url: str) -> dict:
    import numpy as np

    env = AppLensOpenEnv()
    observation = env.reset(repo_url)

    using_rl = _rl_model_available()
    model = _load_rl_model() if using_rl else None

    if using_rl and model:
        print(f"  Mode     : RL agent  (model: {DEFAULT_RL_MODEL_PATH}.zip)")
    else:
        print(f"  Mode     : Fixed sequence (no trained model found at {DEFAULT_RL_MODEL_PATH}.zip)")

    print(f"  Data confidence : {observation.data_confidence:.4f}  ({int(observation.data_confidence * 100)}%)")
    print(f"  Fetch reward    : +{observation.fetch_reward:.4f}")
    print()

    done = False
    last_reward_total = 0.0
    completed: set[str] = set()

    if using_rl and model:
        # RL agent selects each action dynamically.
        from agent.mock_env import REQUIRED_ACTIONS as RL_ACTIONS, MAX_STEPS, _encode_obs
        for step in range(MAX_STEPS):
            obs_vec = _encode_obs(completed, step, MAX_STEPS)
            action_idx, _ = model.predict(obs_vec, deterministic=True)
            action_name = RL_ACTIONS[int(action_idx)]
            action = Action(action=action_name)
            observation, reward, done, _info = env.step(action)
            last_reward_total = reward.total
            if "result" in _info:
                completed.add(action_name)
            if done:
                break
    else:
        # Fallback: deterministic fixed sequence.
        for action_name in ACTION_SEQUENCE:
            if done:
                break
            action = Action(action=action_name)
            observation, reward, done, _info = env.step(action)
            last_reward_total = reward.total
            if "result" in _info:
                completed.add(action_name)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_url": repo_url,
        "app_id": observation.app_id,
        "steps": observation.step_count,
        "max_steps": observation.max_steps,
        "required_actions": observation.required_actions,
        "completed_actions": observation.completed_actions,
        "reward_total": round(last_reward_total, 4),
        "results": observation.results,
        "data_confidence": observation.data_confidence,
        "fetch_reward": observation.fetch_reward,
        "mode": "rl_agent" if (using_rl and model) else "fixed_sequence",
    }


def write_json(data: dict, output_path: Path) -> Path:
    output_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return output_path


def _collect_report_data(data: dict[str, Any]) -> dict[str, Any]:
    results = data.get("results", {})
    language = results.get("detect_language", {}).get("language", "unknown")
    loc = results.get("calculate_loc", {}).get("total_loc", 0)
    file_count = results.get("calculate_loc", {}).get("file_count", 0)
    deps = results.get("parse_dependencies", {}).get("dependencies", [])
    dep_count = results.get("parse_dependencies", {}).get("count", len(deps))
    complexity = results.get("compute_complexity", {})
    sec = results.get("security_scan", {})
    modernization = results.get("recommend_modernization", {})
    consolidated = results.get("generate_report", {}).get("report", {})

    return {
        "results": results,
        "language": language,
        "loc": loc,
        "file_count": file_count,
        "deps": deps,
        "dep_count": dep_count,
        "complexity": complexity,
        "security": sec,
        "modernization": modernization,
        "consolidated": consolidated,
    }


def _render_report_markdown(data: dict) -> str:
    report_data = _collect_report_data(data)
    language = report_data["language"]
    loc = report_data["loc"]
    file_count = report_data["file_count"]
    deps = report_data["deps"]
    dep_count = report_data["dep_count"]
    complexity = report_data["complexity"]
    sec = report_data["security"]
    modernization = report_data["modernization"]
    consolidated = report_data["consolidated"]

    findings = sec.get("findings", [])
    finding_lines = []
    for finding in findings:
        finding_lines.append(
            f"- {finding.get('dependency', 'unknown')}: {finding.get('severity', 'unknown')} ({finding.get('cve', 'N/A')}) - {finding.get('note', '')}"
        )

    recommendations = modernization.get("recommendations", [])
    recommendation_lines = [f"- {item}" for item in recommendations]

    dependency_lines = [f"- {name}" for name in deps]

    report = [
        "# AppLens OpenEnv Analysis Report",
        "",
        f"**Generated on (UTC):** {data.get('generated_at_utc', 'N/A')}",
        f"**Analyzed repository:** {data.get('repo_url', 'N/A')}",
        f"**Application ID:** {data.get('app_id', 'N/A')}",
        "",
        "## Run Status",
        f"- Completed actions: {len(data.get('completed_actions', []))} / {len(data.get('required_actions', []))}",
        f"- Total steps: {data.get('steps', 0)}",
        f"- Reward total: {data.get('reward_total', 0)}",
        "",
        "## Executive Summary",
        f"- Language: {language}",
        f"- LOC: {loc} across {file_count} files",
        f"- Dependencies: {dep_count}",
        f"- Complexity: {complexity.get('level', 'N/A')} (score: {complexity.get('score', 'N/A')})",
        f"- Security findings: {sec.get('vulnerability_count', 0)} (critical: {sec.get('critical_count', 0)})",
        f"- Modernization priority: {modernization.get('priority', 'N/A')}",
        f"- Target stack: {modernization.get('target_stack', 'N/A')}",
        f"- Estimated effort: {modernization.get('effort_weeks', 'N/A')} weeks",
        "",
        "## Dependencies",
        *(dependency_lines or ["- None"]),
        "",
        "## Security Findings",
        *(finding_lines or ["- No vulnerabilities found"]),
        "",
        "## Modernization Recommendations",
        *(recommendation_lines or ["- No recommendations returned"]),
        "",
        "## Consolidated Summary",
        f"- {consolidated.get('summary', 'N/A')}",
    ]
    return "\n".join(report) + "\n"


def _build_pdf_report(data: dict[str, Any], output_pdf: Path) -> Path:
    report_data = _collect_report_data(data)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#111827"),
        fontSize=20,
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        textColor=colors.HexColor("#4B5563"),
        fontSize=9,
        leading=13,
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1F2937"),
        fontSize=12,
        spaceBefore=16,
        spaceAfter=6,
        borderPadding=0,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1F2937"),
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#6B7280"),
        leading=11,
    )

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )

    story: list[Any] = []
    header_band = Table([["APPLICATION ASSESSMENT REPORT"]], colWidths=[6.9 * inch])
    header_band.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1F2937")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(header_band)
    story.append(Spacer(1, 12))
    story.append(Paragraph("AppLens OpenEnv Assessment Report", title_style))
    story.append(
        Paragraph(
            (
                f"Generated: {data.get('generated_at_utc', 'N/A')}<br/>"
                f"Repository: {data.get('repo_url', 'N/A')}<br/>"
                f"Application ID: {data.get('app_id', 'N/A')}"
            ),
            subtitle_style,
        )
    )
    story.append(Spacer(1, 14))

    metrics_table = Table(
        [
            ["Language", "LOC", "Dependencies", "Complexity", "Vulnerabilities", "Priority"],
            [
                report_data["language"],
                f"{report_data['loc']:,}",
                str(report_data["dep_count"]),
                f"{report_data['complexity'].get('level', 'N/A')} ({report_data['complexity'].get('score', 'N/A')})",
                str(report_data["security"].get("vulnerability_count", 0)),
                str(report_data["modernization"].get("priority", "N/A")).title(),
            ],
        ],
        colWidths=[0.9 * inch, 0.8 * inch, 1.0 * inch, 1.35 * inch, 1.1 * inch, 0.95 * inch],
    )
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#374151")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F9FAFB")),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#9CA3AF")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(metrics_table)
    story.append(Spacer(1, 8))
    story.append(Paragraph("Summary metrics derived from the latest completed analysis run.", caption_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Executive Summary", section_style))
    summary_text = (
        f"This repository is primarily a <b>{report_data['language']}</b> codebase with <b>{report_data['loc']:,}</b> "
        f"lines of code across <b>{report_data['file_count']}</b> files. The application shows "
        f"<b>{report_data['complexity'].get('level', 'N/A')}</b> complexity and currently has "
        f"<b>{report_data['security'].get('vulnerability_count', 0)}</b> known vulnerability findings. "
        f"Modernization priority is assessed as <b>{str(report_data['modernization'].get('priority', 'N/A')).title()}</b>, "
        f"with a recommended target stack of <b>{report_data['modernization'].get('target_stack', 'N/A')}</b>."
    )
    story.append(Paragraph(summary_text, body_style))

    story.append(Paragraph("Key Observations", section_style))
    observations = [
        f"Completed {len(data.get('completed_actions', []))} of {len(data.get('required_actions', []))} required actions in {data.get('steps', 0)} steps.",
        f"Security scan identified {report_data['security'].get('vulnerability_count', 0)} known issues, including {report_data['security'].get('critical_count', 0)} critical findings.",
        f"The recommended modernization direction is {report_data['modernization'].get('target_stack', 'N/A')} with an estimated effort of {report_data['modernization'].get('effort_weeks', 'N/A')} weeks.",
    ]
    for item in observations:
        story.append(Paragraph(f"• {item}", body_style))

    story.append(Paragraph("Top Recommendations", section_style))
    recommendations = report_data["modernization"].get("recommendations", []) or ["No recommendations returned"]
    for item in recommendations:
        story.append(Paragraph(f"• {item}", body_style))

    story.append(Paragraph("Security Findings", section_style))
    findings = report_data["security"].get("findings", [])
    if findings:
        findings_rows = [["Dependency", "Severity", "CVE", "Note"]]
        for finding in findings:
            findings_rows.append(
                [
                    finding.get("dependency", "unknown"),
                    str(finding.get("severity", "unknown")).title(),
                    finding.get("cve", "N/A"),
                    finding.get("note", ""),
                ]
            )
        findings_table = Table(findings_rows, colWidths=[1.1 * inch, 0.9 * inch, 1.2 * inch, 2.75 * inch])
        findings_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4B5563")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#FAFAFA")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(findings_table)
    else:
        story.append(Paragraph("No known vulnerabilities were found in the scanned dependency set.", body_style))

    story.append(Paragraph("Dependency Snapshot", section_style))
    dependency_preview = ", ".join(report_data["deps"][:15])
    if len(report_data["deps"]) > 15:
        dependency_preview += ", ..."
    story.append(Paragraph(dependency_preview or "No dependencies detected", body_style))

    story.append(Paragraph("Consolidated Summary", section_style))
    story.append(Paragraph(report_data["consolidated"].get("summary", "N/A"), body_style))

    doc.build(story)
    return output_pdf


def build_document(input_json: Path, output_md: Path | None = None) -> Path:
    data = json.loads(input_json.read_text(encoding="utf-8"))
    if output_md is None:
        app_id = data.get("app_id", "app")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_md = REPORTS_DIR / f"{app_id}-analysis-{timestamp}.pdf"

    if output_md.suffix.lower() == ".md":
        content = _render_report_markdown(data)
        output_md.write_text(content, encoding="utf-8")
        output_path = output_md
    else:
        output_path = _build_pdf_report(data, output_md.with_suffix(".pdf"))

    LATEST_REPORT_POINTER.write_text(str(output_path), encoding="utf-8")
    return output_path


def _build_email_bodies(report_path: Path, subject: str, data: dict[str, Any]) -> tuple[str, str]:
    report_data = _collect_report_data(data)
    plain_text = (
                "Hello,\n\n"
                "The AppLens OpenEnv assessment has completed successfully.\n\n"
        f"Repository: {data.get('repo_url', 'N/A')}\n"
        f"Language: {report_data['language']}\n"
        f"LOC: {report_data['loc']:,}\n"
        f"Dependencies: {report_data['dep_count']}\n"
        f"Complexity: {report_data['complexity'].get('level', 'N/A')}\n"
        f"Vulnerabilities: {report_data['security'].get('vulnerability_count', 0)}\n"
        f"Priority: {report_data['modernization'].get('priority', 'N/A')}\n\n"
        f"Attached report: {report_path.name}\n\n"
                "Regards,\nAppLens OpenEnv\n"
    )

    html_body = f"""
    <html>
            <body style=\"margin:0;padding:24px;background:#f3f4f6;font-family:Segoe UI,Arial,sans-serif;color:#111827;\">
                <table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\" width=\"100%\" style=\"max-width:720px;margin:0 auto;background:#ffffff;border:1px solid #d1d5db;\">
                    <tr>
                        <td style=\"background:#1f2937;color:#ffffff;padding:16px 24px;font-size:12px;letter-spacing:1.2px;text-transform:uppercase;font-weight:600;\">AppLens OpenEnv</td>
                    </tr>
                    <tr>
                        <td style=\"padding:28px 24px 12px 24px;\">
                            <div style=\"font-size:26px;line-height:1.2;font-weight:700;color:#111827;margin-bottom:6px;\">Assessment Report</div>
                            <div style=\"font-size:14px;color:#4b5563;\">{subject}</div>
                        </td>
                    </tr>
                    <tr>
                        <td style=\"padding:0 24px 12px 24px;\">
                            <p style=\"font-size:14px;line-height:1.7;margin:0;color:#374151;\">The repository assessment has completed successfully. Please find the attached PDF report for detailed findings and recommendations.</p>
                        </td>
                    </tr>
                    <tr>
                        <td style=\"padding:4px 24px 16px 24px;\">
                            <table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\" width=\"100%\" style=\"border-collapse:collapse;border:1px solid #d1d5db;\">
                                <tr style=\"background:#f9fafb;\">
                                    <td style=\"padding:10px 12px;border-bottom:1px solid #d1d5db;font-size:12px;font-weight:600;color:#374151;\">Metric</td>
                                    <td style=\"padding:10px 12px;border-bottom:1px solid #d1d5db;font-size:12px;font-weight:600;color:#374151;\">Value</td>
                                </tr>
                                <tr><td style=\"padding:10px 12px;border-bottom:1px solid #e5e7eb;font-size:13px;color:#374151;\">Language</td><td style=\"padding:10px 12px;border-bottom:1px solid #e5e7eb;font-size:13px;color:#111827;\">{report_data['language']}</td></tr>
                                <tr><td style=\"padding:10px 12px;border-bottom:1px solid #e5e7eb;font-size:13px;color:#374151;\">Lines of code</td><td style=\"padding:10px 12px;border-bottom:1px solid #e5e7eb;font-size:13px;color:#111827;\">{report_data['loc']:,}</td></tr>
                                <tr><td style=\"padding:10px 12px;border-bottom:1px solid #e5e7eb;font-size:13px;color:#374151;\">Dependencies</td><td style=\"padding:10px 12px;border-bottom:1px solid #e5e7eb;font-size:13px;color:#111827;\">{report_data['dep_count']}</td></tr>
                                <tr><td style=\"padding:10px 12px;border-bottom:1px solid #e5e7eb;font-size:13px;color:#374151;\">Complexity</td><td style=\"padding:10px 12px;border-bottom:1px solid #e5e7eb;font-size:13px;color:#111827;\">{report_data['complexity'].get('level', 'N/A')}</td></tr>
                                <tr><td style=\"padding:10px 12px;border-bottom:1px solid #e5e7eb;font-size:13px;color:#374151;\">Vulnerability findings</td><td style=\"padding:10px 12px;border-bottom:1px solid #e5e7eb;font-size:13px;color:#111827;\">{report_data['security'].get('vulnerability_count', 0)}</td></tr>
                                <tr><td style=\"padding:10px 12px;font-size:13px;color:#374151;\">Modernization priority</td><td style=\"padding:10px 12px;font-size:13px;color:#111827;\">{str(report_data['modernization'].get('priority', 'N/A')).title()}</td></tr>
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td style=\"padding:0 24px 18px 24px;\">
                            <div style=\"font-size:12px;font-weight:600;color:#4b5563;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;\">Executive summary</div>
                            <div style=\"font-size:14px;line-height:1.7;color:#374151;\">{report_data['consolidated'].get('summary', 'Assessment completed successfully.')}</div>
                        </td>
                    </tr>
                    <tr>
                        <td style=\"padding:16px 24px;background:#f9fafb;border-top:1px solid #e5e7eb;font-size:12px;color:#6b7280;\">
                            Attachment: <strong style=\"color:#111827;\">{report_path.name}</strong>
                        </td>
                    </tr>
                </table>
            </body>
        </html>
    """
    return plain_text, html_body


def send_mail(
    recipient: str,
    report_path: Path,
    subject: str | None = None,
    smtp_user: str | None = None,
    smtp_password: str | None = None,
) -> None:
    smtp_server = os.getenv("GMAIL_SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("GMAIL_SMTP_PORT", "587"))
    smtp_user = smtp_user or os.getenv("GMAIL_SMTP_USER")
    smtp_password = smtp_password or os.getenv("GMAIL_SMTP_APP_PASSWORD")

    if not smtp_user or not smtp_password:
        raise RuntimeError(
            "Missing SMTP credentials. Provide --smtp-user and --smtp-password, or set GMAIL_SMTP_USER and GMAIL_SMTP_APP_PASSWORD."
        )

    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")

    if subject is None:
        subject = f"AppLens OpenEnv Report - {report_path.stem}"

    data = json.loads(LATEST_ANALYSIS_FILE.read_text(encoding="utf-8")) if LATEST_ANALYSIS_FILE.exists() else {"results": {}}
    plain_text, html_body = _build_email_bodies(report_path=report_path, subject=subject, data=data)

    msg = EmailMessage()
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(plain_text)
    msg.add_alternative(html_body, subtype="html")

    subtype = "pdf" if report_path.suffix.lower() == ".pdf" else "octet-stream"
    msg.add_attachment(
        report_path.read_bytes(),
        maintype="application",
        subtype=subtype,
        filename=report_path.name,
    )

    smtp_timeout = float(os.getenv("GMAIL_SMTP_TIMEOUT_SECONDS", "20"))

    with smtplib.SMTP(smtp_server, smtp_port, timeout=smtp_timeout) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(smtp_user, smtp_password)
        smtp.send_message(msg)


def get_latest_report_path() -> Path:
    if not LATEST_REPORT_POINTER.exists():
        raise FileNotFoundError(
            "No latest report pointer found. Run the 'document' stage first."
        )
    return Path(LATEST_REPORT_POINTER.read_text(encoding="utf-8").strip())


def main() -> None:
    ensure_dirs()

    parser = argparse.ArgumentParser(
        description="AppLens workflow: analyze -> document -> send"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Run analysis and save JSON artifact")
    analyze_parser.add_argument("repo_url", help="Repository URL to analyze")
    analyze_parser.add_argument(
        "--output-json",
        default=str(LATEST_ANALYSIS_FILE),
        help="Output JSON path",
    )

    doc_parser = subparsers.add_parser("document", help="Create PDF report from analysis JSON")
    doc_parser.add_argument(
        "--input-json",
        default=str(LATEST_ANALYSIS_FILE),
        help="Input analysis JSON path",
    )
    doc_parser.add_argument(
        "--output-doc",
        default=None,
        help="Output report path (.pdf by default, .md optional)",
    )

    send_parser = subparsers.add_parser("send", help="Send the report by email")
    send_parser.add_argument("--to", required=True, help="Recipient email")
    send_parser.add_argument("--report", default=None, help="Report path (defaults to latest generated)")
    send_parser.add_argument("--subject", default=None, help="Email subject")
    send_parser.add_argument("--smtp-user", default=None, help="SMTP username (optional if env var is set)")
    send_parser.add_argument("--smtp-password", default=None, help="SMTP app password (optional if env var is set)")

    all_parser = subparsers.add_parser("all", help="Run analyze, document, and send in one command")
    all_parser.add_argument("repo_url", help="Repository URL to analyze")
    all_parser.add_argument("--to", required=True, help="Recipient email")
    all_parser.add_argument("--subject", default=None, help="Email subject")
    all_parser.add_argument("--smtp-user", default=None, help="SMTP username (optional if env var is set)")
    all_parser.add_argument("--smtp-password", default=None, help="SMTP app password (optional if env var is set)")

    args = parser.parse_args()

    if args.command == "analyze":
        output_json = Path(args.output_json)
        result = run_analysis(args.repo_url)
        write_json(result, output_json)
        print(f"Analysis completed. JSON saved to: {output_json}")
        return

    if args.command == "document":
        input_json = Path(args.input_json)
        output_doc = Path(args.output_doc) if args.output_doc else None
        report_path = build_document(input_json=input_json, output_md=output_doc)
        print(f"Document created: {report_path}")
        return

    if args.command == "send":
        report_path = Path(args.report) if args.report else get_latest_report_path()
        send_mail(
            recipient=args.to,
            report_path=report_path,
            subject=args.subject,
            smtp_user=args.smtp_user,
            smtp_password=args.smtp_password,
        )
        print(f"Email sent to {args.to} with attachment: {report_path}")
        return

    if args.command == "all":
        result = run_analysis(args.repo_url)
        write_json(result, LATEST_ANALYSIS_FILE)
        report_path = build_document(input_json=LATEST_ANALYSIS_FILE)
        send_mail(
            recipient=args.to,
            report_path=report_path,
            subject=args.subject,
            smtp_user=args.smtp_user,
            smtp_password=args.smtp_password,
        )
        print("Workflow completed: analysis + document + email")
        print(f"JSON: {LATEST_ANALYSIS_FILE}")
        print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
