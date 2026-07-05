"""
generate_report node — writes the final portfolio report.
Uses LLM only for prose — all numbers must come from state.
Always appends SEBI regulatory disclaimer.
"""
import logging
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from ..state import PortfolioState
from ..config import config
from ..prompts import REPORT_GENERATION_PROMPT

logger = logging.getLogger(__name__)

llm = ChatGroq(
    model=config.GROQ_MODEL,
    api_key=config.GROQ_API_KEY,
    temperature=0.2,
    max_tokens=3000,
)


async def generate_report(state: PortfolioState) -> dict:
    """
    Generate the final analysis report.
    Structure: Portfolio Summary → Alerts → Opportunities → Disclaimer
    """
    validated_alerts = state.get("validated_alerts", [])
    portfolio_metrics = state.get("portfolio_metrics", {})
    screened_opportunities = state.get("screened_opportunities", [])
    blocked_alerts = state.get("blocked_alerts", [])

    metrics = portfolio_metrics.get("metrics", {})
    sector_exposure = portfolio_metrics.get("sector_exposure", {})
    holdings = portfolio_metrics.get("holdings", [])

    # Build structured context for LLM — only verified numbers
    context = f"""
PORTFOLIO ANALYSIS REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M IST')}

## Portfolio Metrics (source: tool_call_id={portfolio_metrics.get('tool_call_id', 'N/A')})
- Portfolio Value: ₹{portfolio_metrics.get('portfolio_value_inr', 'N/A'):,}
- Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}
- Annualized Return: {metrics.get('annualized_return_pct', 'N/A')}%
- Annualized Volatility: {metrics.get('annualized_volatility_pct', 'N/A')}%
- Max Drawdown: {metrics.get('max_drawdown_pct', 'N/A')}%
- VaR (95%, daily): {metrics.get('var_95_pct', 'N/A')}%

## Sector Exposure
{chr(10).join(f'- {k}: {v}%' for k, v in sector_exposure.items())}

## Holdings P&L
{chr(10).join(f"- {h['symbol']}: ₹{h.get('current_price', 'N/A')} (P&L: {h.get('pnl_pct', 'N/A')}%)" for h in holdings)}

## Validated Alerts ({len(validated_alerts)} total)
{chr(10).join(f"- [{a['severity'].upper()}] {a['symbol']}: {a['rationale']}" for a in validated_alerts) or 'No alerts triggered this cycle.'}

## Blocked Alerts ({len(blocked_alerts)} blocked by guardrails)
{chr(10).join(f"- {b['alert']['symbol']}: {', '.join(b['reasons'])}" for b in blocked_alerts) or 'None blocked.'}

## Factor-Screened Opportunities (top 5)
{chr(10).join(f"- {o['symbol']} (score: {o.get('composite_score', 'N/A')}): {o.get('explanation', 'No explanation')}" for o in screened_opportunities[:5]) or 'No opportunities identified.'}
"""

    prompt = (
        f"Write a professional portfolio analysis report based on the following verified data. "
        f"Do not invent or estimate any numbers — use only what is provided below.\n\n{context}"
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content=REPORT_GENERATION_PROMPT),
            HumanMessage(content=prompt),
        ])
        report_body = response.content
    except Exception as e:
        logger.error(f"Report generation LLM call failed: {e}")
        report_body = f"[Report generation failed: {e}]\n\nRaw data:\n{context}"

    # Guardrail 5.5 — ALWAYS append disclaimer, never leave to LLM discretion
    full_report = report_body.rstrip() + config.REGULATORY_DISCLAIMER

    logger.info(f"Report generated ({len(full_report)} chars)")
    return {"report_draft": full_report}
