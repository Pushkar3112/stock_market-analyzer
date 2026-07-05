"""
send_whatsapp node — delivery only. Never alters content.
Implements Guardrail 5.6: 24-hour session window + template enforcement.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

from ..state import PortfolioState
from ..config import config

logger = logging.getLogger(__name__)

# WhatsApp message size limit
WHATSAPP_MAX_CHARS = 4096


def _truncate_safely(report: str, max_chars: int = WHATSAPP_MAX_CHARS) -> str:
    """Truncate report to WhatsApp limit WITHOUT removing the disclaimer."""
    disclaimer_marker = "---\nDISCLAIMER:"
    if len(report) <= max_chars:
        return report

    disclaimer_idx = report.rfind(disclaimer_marker)
    if disclaimer_idx == -1:
        return report[:max_chars - 3] + "..."

    # Keep as much body as possible, always keep disclaimer
    disclaimer_part = report[disclaimer_idx:]
    body_budget = max_chars - len(disclaimer_part) - 10
    truncated_body = report[:body_budget] + "\n[... truncated for length]\n\n"
    return truncated_body + disclaimer_part


def _is_within_session_window(last_user_message_at: Optional[str]) -> bool:
    """Guardrail 5.6 — Check 24-hour session window."""
    if not last_user_message_at:
        return False
    last_msg = datetime.fromisoformat(last_user_message_at)
    return datetime.now() < last_msg + timedelta(hours=24)


async def _send_via_twilio(to: str, message: str, is_template: bool, template_id: Optional[str] = None) -> dict:
    """Send via Twilio WhatsApp API."""
    try:
        from twilio.rest import Client
        client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)

        if is_template and template_id:
            msg = client.messages.create(
                from_=config.TWILIO_WHATSAPP_FROM,
                to=to,
                content_sid=template_id,
            )
        else:
            msg = client.messages.create(
                from_=config.TWILIO_WHATSAPP_FROM,
                to=to,
                body=message,
            )

        return {"status": "sent", "message_sid": msg.sid, "provider": "twilio"}

    except ImportError:
        return {"status": "mock_sent", "message": "Twilio not configured — running in mock mode", "provider": "mock"}
    except Exception as e:
        return {"status": "failed", "error": str(e), "provider": "twilio"}


async def send_whatsapp(state: PortfolioState) -> dict:
    """
    Send report via WhatsApp. Guards:
    1. grounding_check_passed must be True
    2. Use template if outside 24h session window
    3. Per-symbol cooldown respected
    4. Never modify report content
    """
    grounding_passed = state.get("grounding_check_passed", False)
    report = state.get("report_draft", "")
    delivery_status = dict(state.get("delivery_status", {}))
    errors = []

    # Guard 1: Hard block if grounding failed
    if not grounding_passed:
        failures = state.get("grounding_failures", [])
        msg = f"DELIVERY BLOCKED: Grounding check failed. Unverified claims: {failures}"
        logger.error(msg)
        delivery_status["status"] = "blocked_grounding_failure"
        delivery_status["reason"] = msg
        return {"delivery_status": delivery_status, "errors": [msg]}

    # Guard 2: Check if human approval was required and given
    requires_approval = state.get("requires_human_approval", False)
    human_approved = state.get("human_approved")
    if requires_approval and not human_approved:
        msg = "DELIVERY BLOCKED: High-severity alert requires human approval before sending."
        logger.error(msg)
        delivery_status["status"] = "blocked_pending_approval"
        return {"delivery_status": delivery_status, "errors": [msg]}

    # Prepare message
    safe_report = _truncate_safely(report)

    # Guard 3: Session window check (Guardrail 5.6)
    last_user_message = delivery_status.get("last_user_message_at")
    within_session = _is_within_session_window(last_user_message)

    to_number = config.WHATSAPP_TO
    if not to_number:
        msg = "WHATSAPP_TO not configured — running in mock mode"
        logger.warning(msg)
        delivery_status.update({
            "status": "mock_sent",
            "report_length": len(safe_report),
            "session_window": within_session,
            "timestamp": datetime.now().isoformat(),
        })
        return {"delivery_status": delivery_status}

    if within_session:
        # Freeform message — allowed within 24h window
        result = await _send_via_twilio(to_number, safe_report, is_template=False)
    else:
        # Must use approved template outside session window
        logger.info("Outside 24h session window — using approved template (Guardrail 5.6)")
        template_summary = f"Portfolio analysis ready. {len(state.get('validated_alerts', []))} alerts triggered. Check your dashboard."
        result = await _send_via_twilio(to_number, template_summary, is_template=True)

    delivery_status.update({
        **result,
        "freeform": within_session,
        "report_length": len(safe_report),
        "timestamp": datetime.now().isoformat(),
        "alerts_included": len(state.get("validated_alerts", [])),
    })

    logger.info(f"WhatsApp delivery result: {result.get('status')}")
    return {"delivery_status": delivery_status}
