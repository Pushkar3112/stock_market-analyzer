"""
guardrail_validate node — the safety gate. Validates all alerts before delivery.
Implements all 7 guardrails from the spec.
"""
import logging
from datetime import datetime, timedelta
from typing import Any

from ..state import PortfolioState, Alert
from ..config import config

logger = logging.getLogger(__name__)


def _check_symbol_grounding(alert: Alert, tool_call_registry: dict) -> tuple[bool, str]:
    """Guardrail 5.1 — Was validate_symbol called for this symbol?"""
    symbol = alert["symbol"]
    if symbol == "PORTFOLIO":
        return True, ""

    for call_id, call_data in tool_call_registry.items():
        if (call_data.get("tool") == "validate_symbol"
                and call_data.get("args", {}).get("symbol", "").upper() == symbol.upper()
                and call_data.get("output", {}).get("valid") is True):
            return True, ""

    return False, f"Symbol '{symbol}' was not validated via validate_symbol before use."


def _check_numeric_grounding(alert: Alert, tool_call_registry: dict) -> tuple[bool, str]:
    """Guardrail 5.2 — Does every number in grounded_facts trace to a tool_call_id?"""
    source_calls = alert.get("source_tool_calls", [])
    if not source_calls:
        return False, "No source_tool_calls listed — numeric claims unverifiable."

    missing = [c for c in source_calls if c not in tool_call_registry]
    if missing:
        return False, f"source_tool_calls {missing} not found in tool_call_registry."

    return True, ""


def _check_severity_classification(alert: Alert, portfolio_metrics: dict) -> tuple[bool, str]:
    """Guardrail 5.3 — Is severity correctly classified?"""
    metrics = portfolio_metrics.get("metrics", {})
    var_95 = abs(metrics.get("var_95_pct", 0))
    requires_high = portfolio_metrics.get("requires_high_severity", False)

    if alert["alert_type"] == "portfolio_risk" and requires_high and alert["severity"] != "high":
        return False, f"Portfolio risk alert should be severity=high (VaR={var_95:.2f}%), got '{alert['severity']}'."

    return True, ""


def _check_rate_limit(alert: Alert, cooldown_state: dict, cooldown_hours: int) -> tuple[bool, str]:
    """Guardrail 5.4 — Per-symbol cooldown check."""
    symbol = alert["symbol"]
    alert_type = alert["alert_type"]
    key = f"{symbol}_{alert_type}"

    last_sent = cooldown_state.get(key)
    if last_sent:
        last_sent_dt = datetime.fromisoformat(last_sent)
        cooldown_until = last_sent_dt + timedelta(hours=cooldown_hours)
        if datetime.now() < cooldown_until:
            return False, (
                f"Rate limited: '{symbol}' {alert_type} alert already sent at {last_sent}. "
                f"Cooldown until {cooldown_until.isoformat()}."
            )

    return True, ""


def _check_concentration_risk(alert: Alert, portfolio_metrics: dict) -> tuple[bool, str]:
    """Guardrail 5.7 — Would acting on this breach concentration limits?"""
    if alert["alert_type"] != "new_opportunity":
        return True, ""  # only applies to new opportunity alerts

    sector_exposure = portfolio_metrics.get("sector_exposure", {})
    opportunity_sector = alert.get("grounded_facts", {}).get("sector", "Unknown")
    current_sector_weight = sector_exposure.get(opportunity_sector, 0)

    if current_sector_weight >= config.SECTOR_CONCENTRATION_LIMIT:
        return False, (
            f"Adding '{alert['symbol']}' would exceed sector concentration limit "
            f"({opportunity_sector} already at {current_sector_weight:.1f}%)."
        )

    return True, ""


async def guardrail_validate(state: PortfolioState) -> dict:
    """
    Run all 7 guardrail checks on candidate_alerts.
    Anything failing any check goes to blocked_alerts with explicit reason.
    """
    candidate_alerts = state.get("candidate_alerts", [])
    screened_opportunities = state.get("screened_opportunities", [])
    tool_call_registry = state.get("tool_call_registry", {})
    portfolio_metrics = state.get("portfolio_metrics", {})
    cooldown_state = state.get("delivery_status", {}).get("cooldown_registry", {})

    validated_alerts: list[Alert] = []
    blocked_alerts: list[dict] = []
    requires_human_approval = False

    logger.info(f"Guardrail validation: {len(candidate_alerts)} candidate alerts")

    for alert in candidate_alerts:
        rejection_reasons = []

        # Run all checks
        checks = [
            _check_symbol_grounding(alert, tool_call_registry),
            _check_numeric_grounding(alert, tool_call_registry),
            _check_severity_classification(alert, portfolio_metrics),
            _check_rate_limit(alert, cooldown_state, config.ALERT_COOLDOWN_HOURS),
            _check_concentration_risk(alert, portfolio_metrics),
        ]

        for passed, reason in checks:
            if not passed:
                rejection_reasons.append(reason)

        if rejection_reasons:
            blocked_alerts.append({
                "alert": alert,
                "reasons": rejection_reasons,
                "blocked_at": datetime.now().isoformat(),
            })
            logger.warning(f"BLOCKED alert [{alert['symbol']}]: {rejection_reasons}")
        else:
            validated_alerts.append(alert)
            # Check if human approval required (Guardrail 5.3)
            if alert["severity"] == "high":
                requires_human_approval = True
                logger.info(f"HIGH severity alert for {alert['symbol']} — human approval required")

    catch_rate = len(blocked_alerts) / (len(validated_alerts) + len(blocked_alerts)) if candidate_alerts else 0
    logger.info(
        f"Guardrail result: {len(validated_alerts)} validated, "
        f"{len(blocked_alerts)} blocked (catch rate: {catch_rate:.1%})"
    )

    return {
        "validated_alerts": validated_alerts,
        "blocked_alerts": blocked_alerts,
        "requires_human_approval": requires_human_approval,
    }
