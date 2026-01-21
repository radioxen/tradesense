"""Human-in-the-loop approval system.

Provides notification and approval mechanisms for trading decisions.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable
import uuid
import json

from src.orchestrator.contracts import ExecutiveDecision, Action
from src.utils.logging import get_logger


logger = get_logger(__name__)


class ApprovalStatus(str, Enum):
    """Approval status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    AUTO_APPROVED = "auto_approved"


@dataclass
class ApprovalRequest:
    """Approval request for a trading decision."""
    id: str
    decision: ExecutiveDecision
    created_at: datetime
    expires_at: datetime
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: str | None = None
    notes: str | None = None
    responded_at: datetime | None = None


class NotificationChannel(ABC):
    """Abstract notification channel."""

    @abstractmethod
    async def send(
        self,
        request: ApprovalRequest,
        message: str,
    ) -> bool:
        """Send notification.

        Args:
            request: Approval request.
            message: Formatted message.

        Returns:
            True if sent successfully.
        """
        pass

    @abstractmethod
    async def check_response(
        self,
        request_id: str,
    ) -> ApprovalStatus | None:
        """Check for response to a request.

        Args:
            request_id: Request ID.

        Returns:
            Approval status if responded, None otherwise.
        """
        pass


class SlackNotifier(NotificationChannel):
    """Slack notification channel."""

    def __init__(
        self,
        webhook_url: str | None = None,
        bot_token: str | None = None,
        channel: str = "#trading-alerts",
    ):
        """Initialize Slack notifier.

        Args:
            webhook_url: Slack webhook URL.
            bot_token: Slack bot token.
            channel: Channel to post to.
        """
        self.webhook_url = webhook_url
        self.bot_token = bot_token
        self.channel = channel
        self._responses: dict[str, ApprovalStatus] = {}

    async def send(
        self,
        request: ApprovalRequest,
        message: str,
    ) -> bool:
        """Send Slack notification."""
        if not self.webhook_url and not self.bot_token:
            logger.warning("Slack not configured, logging message only")
            logger.info(f"[SLACK] {message}")
            return True

        try:
            import aiohttp

            payload = {
                "channel": self.channel,
                "text": message,
                "blocks": self._build_blocks(request),
            }

            if self.webhook_url:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.webhook_url, json=payload) as resp:
                        return resp.status == 200
            else:
                # Use bot token
                async with aiohttp.ClientSession() as session:
                    headers = {"Authorization": f"Bearer {self.bot_token}"}
                    async with session.post(
                        "https://slack.com/api/chat.postMessage",
                        headers=headers,
                        json=payload,
                    ) as resp:
                        data = await resp.json()
                        return data.get("ok", False)

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    async def check_response(
        self,
        request_id: str,
    ) -> ApprovalStatus | None:
        """Check for Slack response."""
        return self._responses.get(request_id)

    def register_response(
        self,
        request_id: str,
        status: ApprovalStatus,
    ):
        """Register response (called from Slack callback)."""
        self._responses[request_id] = status

    def _build_blocks(self, request: ApprovalRequest) -> list[dict]:
        """Build Slack blocks for rich message."""
        decision = request.decision
        emoji = "📈" if decision.action == Action.BUY else "📉" if decision.action == Action.SELL else "⏸️"

        return [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Trading Decision: {decision.symbol}",
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Action:*\n{decision.action.value}"},
                    {"type": "mrkdwn", "text": f"*Size:*\n{decision.size_pct:.1%}"},
                    {"type": "mrkdwn", "text": f"*Confidence:*\n{decision.confidence:.0%}"},
                    {"type": "mrkdwn", "text": f"*Expires:*\n{request.expires_at.strftime('%H:%M:%S')}"},
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Rationale:*\n{decision.rationale}"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✅ Approve"},
                        "style": "primary",
                        "action_id": f"approve_{request.id}",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "❌ Reject"},
                        "style": "danger",
                        "action_id": f"reject_{request.id}",
                    },
                ]
            }
        ]


class TelegramNotifier(NotificationChannel):
    """Telegram notification channel."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ):
        """Initialize Telegram notifier.

        Args:
            bot_token: Telegram bot token.
            chat_id: Chat ID to send to.
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._responses: dict[str, ApprovalStatus] = {}

    async def send(
        self,
        request: ApprovalRequest,
        message: str,
    ) -> bool:
        """Send Telegram notification."""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram not configured, logging message only")
            logger.info(f"[TELEGRAM] {message}")
            return True

        try:
            import aiohttp

            decision = request.decision
            emoji = "📈" if decision.action == Action.BUY else "📉"

            text = f"""
{emoji} *Trading Decision: {decision.symbol}*

*Action:* {decision.action.value}
*Size:* {decision.size_pct:.1%}
*Confidence:* {decision.confidence:.0%}

*Rationale:*
{decision.rationale}

Reply with /approve_{request.id[:8]} or /reject_{request.id[:8]}
Expires: {request.expires_at.strftime('%H:%M:%S')}
"""

            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    return resp.status == 200

        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False

    async def check_response(
        self,
        request_id: str,
    ) -> ApprovalStatus | None:
        """Check for Telegram response."""
        return self._responses.get(request_id)

    def register_response(
        self,
        request_id: str,
        status: ApprovalStatus,
    ):
        """Register response (called from Telegram webhook)."""
        self._responses[request_id] = status


class ConsoleNotifier(NotificationChannel):
    """Console notification for testing."""

    def __init__(self):
        self._responses: dict[str, ApprovalStatus] = {}
        self._pending: dict[str, ApprovalRequest] = {}

    async def send(
        self,
        request: ApprovalRequest,
        message: str,
    ) -> bool:
        """Print to console."""
        decision = request.decision
        print("\n" + "=" * 60)
        print(f"🔔 TRADING DECISION REQUIRES APPROVAL")
        print("=" * 60)
        print(f"Symbol: {decision.symbol}")
        print(f"Action: {decision.action.value}")
        print(f"Size: {decision.size_pct:.1%}")
        print(f"Confidence: {decision.confidence:.0%}")
        print(f"Rationale: {decision.rationale}")
        print(f"Request ID: {request.id[:8]}")
        print(f"Expires: {request.expires_at}")
        print("=" * 60 + "\n")

        self._pending[request.id] = request
        return True

    async def check_response(
        self,
        request_id: str,
    ) -> ApprovalStatus | None:
        """Check console response."""
        return self._responses.get(request_id)

    def approve(self, request_id: str):
        """Approve a request."""
        for rid in self._pending:
            if rid.startswith(request_id):
                self._responses[rid] = ApprovalStatus.APPROVED
                return

    def reject(self, request_id: str):
        """Reject a request."""
        for rid in self._pending:
            if rid.startswith(request_id):
                self._responses[rid] = ApprovalStatus.REJECTED
                return


class HILApprovalManager:
    """Human-in-the-loop approval manager.

    Manages approval requests and notifications.
    """

    def __init__(
        self,
        channels: list[NotificationChannel] | None = None,
        timeout_seconds: int = 300,  # 5 minutes
        auto_approve_threshold: float = 0.8,  # Auto-approve above this confidence
        auto_approve_enabled: bool = False,
    ):
        """Initialize approval manager.

        Args:
            channels: Notification channels.
            timeout_seconds: Timeout for approval.
            auto_approve_threshold: Confidence threshold for auto-approval.
            auto_approve_enabled: Enable auto-approval.
        """
        self.channels = channels or [ConsoleNotifier()]
        self.timeout_seconds = timeout_seconds
        self.auto_approve_threshold = auto_approve_threshold
        self.auto_approve_enabled = auto_approve_enabled

        self._pending: dict[str, ApprovalRequest] = {}
        self._history: list[ApprovalRequest] = []

    async def request_approval(
        self,
        decision: ExecutiveDecision,
    ) -> ApprovalRequest:
        """Request approval for a decision.

        Args:
            decision: Executive decision.

        Returns:
            Approval request.
        """
        request = ApprovalRequest(
            id=str(uuid.uuid4()),
            decision=decision,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=self.timeout_seconds),
        )

        # Check auto-approval
        if self.auto_approve_enabled and decision.confidence >= self.auto_approve_threshold:
            request.status = ApprovalStatus.AUTO_APPROVED
            request.responded_at = datetime.now()
            logger.info(f"Auto-approved decision for {decision.symbol} (conf={decision.confidence:.0%})")
            self._history.append(request)
            return request

        # Send notifications
        message = self._format_message(request)
        for channel in self.channels:
            try:
                await channel.send(request, message)
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")

        self._pending[request.id] = request
        return request

    async def wait_for_approval(
        self,
        request: ApprovalRequest,
        poll_interval: float = 5.0,
    ) -> ApprovalStatus:
        """Wait for approval response.

        Args:
            request: Approval request.
            poll_interval: Seconds between polls.

        Returns:
            Final approval status.
        """
        if request.status != ApprovalStatus.PENDING:
            return request.status

        while datetime.now() < request.expires_at:
            # Check each channel for response
            for channel in self.channels:
                status = await channel.check_response(request.id)
                if status:
                    request.status = status
                    request.responded_at = datetime.now()
                    self._pending.pop(request.id, None)
                    self._history.append(request)
                    logger.info(f"Approval {status.value} for {request.decision.symbol}")
                    return status

            await asyncio.sleep(poll_interval)

        # Timeout
        request.status = ApprovalStatus.TIMEOUT
        request.responded_at = datetime.now()
        self._pending.pop(request.id, None)
        self._history.append(request)
        logger.warning(f"Approval timeout for {request.decision.symbol}")
        return ApprovalStatus.TIMEOUT

    def _format_message(self, request: ApprovalRequest) -> str:
        """Format notification message."""
        decision = request.decision
        return f"""
Trading Decision: {decision.action.value} {decision.symbol}
Size: {decision.size_pct:.1%} of equity
Confidence: {decision.confidence:.0%}
Rationale: {decision.rationale}
Request ID: {request.id[:8]}
Expires: {request.expires_at.strftime('%Y-%m-%d %H:%M:%S')}
"""

    def get_pending(self) -> list[ApprovalRequest]:
        """Get pending requests."""
        return list(self._pending.values())

    def get_history(self, limit: int = 100) -> list[ApprovalRequest]:
        """Get approval history."""
        return self._history[-limit:]
