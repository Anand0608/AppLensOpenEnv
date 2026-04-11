import argparse
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


def build_message(sender: str, recipient: str, subject: str, body: str, attachments: list[Path]) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    for file_path in attachments:
        data = file_path.read_bytes()
        msg.add_attachment(
            data,
            maintype="application",
            subtype="octet-stream",
            filename=file_path.name,
        )

    return msg


def main() -> None:
    parser = argparse.ArgumentParser(description="Send AppLens report via Gmail SMTP.")
    parser.add_argument("--to", required=True, help="Recipient email address")
    parser.add_argument(
        "--subject",
        default="AppLens OpenEnv Report - Repository Analysis",
        help="Email subject",
    )
    parser.add_argument(
        "--body-file",
        default="analysis/flask-analysis-email-draft.txt",
        help="Path to text file used as email body",
    )
    parser.add_argument(
        "--attach",
        action="append",
        default=["analysis/flask-analysis-summary.md"],
        help="Attachment path (repeat --attach for multiple files)",
    )
    args = parser.parse_args()

    smtp_server = os.getenv("GMAIL_SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("GMAIL_SMTP_PORT", "587"))
    smtp_user = os.getenv("GMAIL_SMTP_USER")
    smtp_app_password = os.getenv("GMAIL_SMTP_APP_PASSWORD")

    if not smtp_user or not smtp_app_password:
        raise RuntimeError(
            "Missing Gmail credentials. Set GMAIL_SMTP_USER and GMAIL_SMTP_APP_PASSWORD environment variables."
        )

    body_path = Path(args.body_file)
    if not body_path.exists():
        raise FileNotFoundError(f"Body file not found: {body_path}")

    body = body_path.read_text(encoding="utf-8")
    attachment_paths = [Path(item) for item in args.attach]
    for path in attachment_paths:
        if not path.exists():
            raise FileNotFoundError(f"Attachment not found: {path}")

    message = build_message(
        sender=smtp_user,
        recipient=args.to,
        subject=args.subject,
        body=body,
        attachments=attachment_paths,
    )

    with smtplib.SMTP(smtp_server, smtp_port) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(smtp_user, smtp_app_password)
        smtp.send_message(message)

    print(f"Email sent to {args.to}")


if __name__ == "__main__":
    main()
