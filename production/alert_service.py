"""Alert service for production trading pipeline.

Sends notifications via email (SMTP) or webhook (Slack/Discord)
when critical events occur: drawdowns, model drift, data failures.
"""

import json
import logging
import os
import smtplib
import time
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)


class AlertService:
    """Sends alerts via email and/or webhook with cooldown."""

    def __init__(
        self,
        smtp_host: str = None,
        smtp_port: int = 587,
        smtp_user: str = None,
        smtp_password: str = None,
        alert_email: str = None,
        webhook_url: str = None,
        cooldown_seconds: int = 3600,
    ):
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST", "")
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user or os.getenv("SMTP_USER", "")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD", "")
        self.alert_email = alert_email or os.getenv("ALERT_EMAIL", "")
        self.webhook_url = webhook_url or os.getenv("ALERT_WEBHOOK_URL", "")
        self.cooldown_seconds = cooldown_seconds
        self._last_alert_time: Dict[str, float] = {}

    def _should_send(self, alert_type: str) -> bool:
        """Check cooldown to avoid alert flooding."""
        now = time.time()
        last = self._last_alert_time.get(alert_type, 0)
        if now - last < self.cooldown_seconds:
            logger.debug(f"Alert '{alert_type}' suppressed (cooldown)")
            return False
        return True

    def _record_sent(self, alert_type: str):
        self._last_alert_time[alert_type] = time.time()

    def send_email(self, subject: str, body: str) -> bool:
        """Send alert via SMTP email."""
        if not self.smtp_host or not self.alert_email:
            logger.warning("Email not configured, skipping email alert")
            return False
        try:
            msg = MIMEText(body, "plain")
            msg["Subject"] = f"[Energy Trading] {subject}"
            msg["From"] = self.smtp_user
            msg["To"] = self.alert_email

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(f"Email alert sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_webhook(self, subject: str, body: str) -> bool:
        """Send alert via webhook (Slack/Discord compatible)."""
        if not self.webhook_url:
            logger.warning("Webhook not configured, skipping webhook alert")
            return False
        try:
            payload = json.dumps({
                "text": f"*{subject}*\n{body}",
                "username": "Energy Trading Bot",
            }).encode("utf-8")

            req = Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urlopen(req, timeout=10)
            logger.info(f"Webhook alert sent: {subject}")
            return True
        except URLError as e:
            logger.error(f"Failed to send webhook: {e}")
            return False

    def send_alert(self, alert_type: str, subject: str, body: str):
        """Send alert via all configured channels with cooldown."""
        if not self._should_send(alert_type):
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_body = f"[{timestamp}] {body}"

        sent = False
        if self.webhook_url:
            sent = self.send_webhook(subject, full_body) or sent
        if self.smtp_host and self.alert_email:
            sent = self.send_email(subject, full_body) or sent

        if not sent:
            # Fallback: log the alert
            logger.warning(f"ALERT ({alert_type}): {subject} - {body}")

        self._record_sent(alert_type)

    # Convenience methods for common alerts
    def alert_drawdown(self, current_dd: float, threshold: float):
        self.send_alert(
            "drawdown",
            f"Drawdown Alert: EUR {current_dd:,.2f}",
            f"Current drawdown ({current_dd:,.2f} EUR) exceeds threshold ({threshold:,.2f} EUR).\n"
            "Consider reducing position size or pausing trading.",
        )

    def alert_model_drift(self, metric_name: str, current: float, baseline: float):
        self.send_alert(
            "model_drift",
            f"Model Drift: {metric_name} degraded",
            f"{metric_name} current={current:.4f} vs baseline={baseline:.4f} "
            f"(ratio={current/baseline:.2f}x).\nModel retraining recommended.",
        )

    def alert_data_failure(self, source: str, error: str):
        self.send_alert(
            f"data_failure_{source}",
            f"Data Fetch Failed: {source}",
            f"Failed to fetch data from {source}.\nError: {error}\n"
            "Using cached data as fallback.",
        )

    def alert_pipeline_error(self, step: str, error: str):
        self.send_alert(
            "pipeline_error",
            f"Pipeline Error: {step}",
            f"Pipeline step '{step}' failed.\nError: {error}",
        )
