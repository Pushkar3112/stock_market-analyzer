"""
generate_candidate_alerts node — creates alerts from technical indicators + portfolio metrics.
Every alert must cite source_tool_calls — no invented numbers.
"""
import uuid
import logging
from datetime import datetime
from typing import Literal

from ..state import PortfolioState, Alert
from ..config import config

logger = logging.getLogger(__name__)


def _determine_severity(
    alert_type: str,
    symbol: str,
    data: dict,
    portfolio_metrics: dict,
) -> Literal["low", "medium", "high"]:
    """Determine alert severity from thresholds — never LLM opinion."""
    metrics = portfolio_metrics.get("metrics", {})
    var_95 = abs(metrics.get("var_95_pct", 0))
    sector_exp = portfolio_metrics.get("sector_exposure", {})
    max_sector = max(sector_exp.values()) if sector_exp else 0

    if alert_type == "portfolio_risk":
        if var_95 > config.VAR_THRESHOLD_PCT:
            return "high"
        if max_sector > config.SECTOR_CONCENTRATION_LIMIT:
            return "high"
        return "medium"

    if alert_type == "price_move":
        change_pct = abs(data.get("change_pct", 0) or 0)
        if change_pct >= config.HIGH_SEVERITY_PRICE_MOVE_PCT:
            return "high"
        if change_pct >= 4.0:
            return "medium"
        return "low"

    if alert_type == "technical_signal":
        rsi = data.get("rsi", 50)
        if rsi is not None and (rsi > 80 or rsi < 20):
            return "high"
        if rsi is not None and (rsi > 70 or rsi < 30):
            return "medium"
        return "low"

    return "low"


async def generate_candidate_alerts(state: PortfolioState) -> dict:
    """
    Generate candidate alerts from:
    1. Large price moves (> threshold)
    2. Technical signals (RSI extremes, MACD crossovers, Bollinger breaks)
    3. Portfolio-level risk breaches

    Each alert cites source_tool_calls for grounding verification.
    """
    raw_market_data = state.get("raw_market_data", {})
    technical_indicators = state.get("technical_indicators", {})
    portfolio_metrics = state.get("portfolio_metrics", {})
    candidate_alerts: list[Alert] = []
    errors = []

    # --- Alert type 1: Price moves ---
    for symbol, data in raw_market_data.items():
        equity = data.get("equity_details", {})
        change_pct = equity.get("change_pct")

        if change_pct is None:
            continue

        if abs(change_pct) >= 3.0:  # only alert on meaningful moves
            severity = _determine_severity("price_move", symbol, {"change_pct": change_pct}, portfolio_metrics)
            alert: Alert = {
                "symbol": symbol,
                "alert_type": "price_move",
                "severity": severity,
                "grounded_facts": {
                    "current_price": equity.get("current_price"),
                    "previous_close": equity.get("previous_close"),
                    "change_pct": change_pct,
                },
                "rationale": (
                    f"{symbol} moved {change_pct:+.2f}% today "
                    f"(from {equity.get('previous_close')} to {equity.get('current_price')})."
                ),
                "source_tool_calls": [data.get("equity_tool_call_id", "unknown")],
            }
            candidate_alerts.append(alert)
            logger.info(f"Price move alert: {symbol} {change_pct:+.2f}% [{severity}]")

    # --- Alert type 2: Technical signals ---
    for symbol, ind_data in technical_indicators.items():
        indicators = ind_data.get("indicators", {})
        tool_call_id = ind_data.get("tool_call_id", "unknown")
        current_price = ind_data.get("current_price")

        rsi_data = indicators.get("RSI", {})
        rsi_val = rsi_data.get("value")
        macd_data = indicators.get("MACD", {})
        bb_data = indicators.get("Bollinger_Bands", {})

        # RSI extremes
        if rsi_val is not None and (rsi_val > 70 or rsi_val < 30):
            severity = _determine_severity("technical_signal", symbol, {"rsi": rsi_val}, portfolio_metrics)
            label = "overbought" if rsi_val > 70 else "oversold"
            alert = {
                "symbol": symbol,
                "alert_type": "technical_signal",
                "severity": severity,
                "grounded_facts": {
                    "rsi": rsi_val,
                    "current_price": current_price,
                    "interpretation": rsi_data.get("interpretation"),
                },
                "rationale": f"{symbol} RSI at {rsi_val:.1f} — {label}. {rsi_data.get('interpretation', '')}",
                "source_tool_calls": [tool_call_id],
            }
            candidate_alerts.append(alert)

        # MACD crossover signal
        if macd_data:
            hist = macd_data.get("histogram", 0)
            interpretation = macd_data.get("interpretation")
            if abs(hist) > 0:  # any MACD divergence
                alert = {
                    "symbol": symbol,
                    "alert_type": "technical_signal",
                    "severity": "low",
                    "grounded_facts": {
                        "macd_line": macd_data.get("macd_line"),
                        "signal_line": macd_data.get("signal_line"),
                        "histogram": hist,
                        "current_price": current_price,
                    },
                    "rationale": (
                        f"{symbol} MACD {interpretation}: histogram at {hist:.4f}. "
                        f"MACD={macd_data.get('macd_line'):.4f}, Signal={macd_data.get('signal_line'):.4f}."
                    ),
                    "source_tool_calls": [tool_call_id],
                }
                candidate_alerts.append(alert)

    # --- Alert type 3: Portfolio-level risk ---
    severity_flags = portfolio_metrics.get("severity_flags", [])
    if severity_flags:
        pm_tool_call_id = portfolio_metrics.get("tool_call_id", "unknown")
        metrics = portfolio_metrics.get("metrics", {})
        severity = "high" if portfolio_metrics.get("requires_high_severity") else "medium"
        alert = {
            "symbol": "PORTFOLIO",
            "alert_type": "portfolio_risk",
            "severity": severity,
            "grounded_facts": {
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "annualized_volatility_pct": metrics.get("annualized_volatility_pct"),
                "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                "var_95_pct": metrics.get("var_95_pct"),
                "sector_exposure": portfolio_metrics.get("sector_exposure", {}),
            },
            "rationale": "Portfolio risk breach: " + "; ".join(severity_flags),
            "source_tool_calls": [pm_tool_call_id],
        }
        candidate_alerts.append(alert)
        logger.info(f"Portfolio risk alert [{severity}]: {'; '.join(severity_flags)}")

    logger.info(f"Generated {len(candidate_alerts)} candidate alerts")
    return {
        "candidate_alerts": candidate_alerts,
        "errors": errors,
    }
