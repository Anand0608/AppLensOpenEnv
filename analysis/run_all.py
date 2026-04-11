from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.workflow import (
    LATEST_ANALYSIS_FILE,
    build_document,
    run_analysis,
    send_mail,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command workflow: run analysis, create report document, and send by email."
    )
    parser.add_argument("repo_url", help="Repository URL to analyze")
    parser.add_argument("recipient", help="Recipient email address")
    parser.add_argument("--subject", default=None, help="Optional email subject")
    parser.add_argument("--smtp-user", default=None, help="SMTP username (optional if env var is set)")
    parser.add_argument("--smtp-password", default=None, help="SMTP app password (optional if env var is set)")
    args = parser.parse_args()

    result = run_analysis(args.repo_url)
    write_json(result, LATEST_ANALYSIS_FILE)
    report_path = build_document(input_json=LATEST_ANALYSIS_FILE)
    send_mail(
        recipient=args.recipient,
        report_path=report_path,
        subject=args.subject,
        smtp_user=args.smtp_user,
        smtp_password=args.smtp_password,
    )

    print("Completed all stages successfully:")
    print(f"- Analysis JSON: {LATEST_ANALYSIS_FILE}")
    print(f"- Report document: {report_path}")
    print(f"- Email sent to: {args.recipient}")


if __name__ == "__main__":
    main()
