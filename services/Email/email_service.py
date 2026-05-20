"""Email service - env-only configuration."""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def get_email_config():
    """Load email config from environment variables only."""
    sender = os.environ.get("EMAIL_SENDER")
    recipient = os.environ.get("EMAIL_RECIPIENT")
    app_password = os.environ.get("EMAIL_APP_PASSWORD")
    
    if not all([sender, recipient, app_password]):
        raise EnvironmentError(
            "EMAIL_SENDER, EMAIL_RECIPIENT, and EMAIL_APP_PASSWORD must be set in environment"
        )
    
    return sender, recipient, app_password


def send_email(subject: str, body: str, html: bool = False) -> bool:
    """Send an email using env-configured credentials."""
    try:
        sender, recipient, app_password = get_email_config()
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipient
        
        if html:
            msg.attach(MIMEText(body, "html"))
        else:
            msg.attach(MIMEText(body, "plain"))
        
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, app_password)
            server.sendmail(sender, recipient, msg.as_string())
        return True
    except Exception as e:
        print(f"Email send failed: {e}")
        return False
