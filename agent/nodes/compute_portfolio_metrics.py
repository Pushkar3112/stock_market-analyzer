"""
compute_portfolio_metrics node — Sharpe, volatility, max drawdown, VaR(95%), sector concentration.
Pure math. Never LLM-generated.
"""
import uuid
import logging
from datetime import datetime

from ..state import PortfolioState
from ...mcp_server.tools.portfolio_metrics import compute_portfolio_metrics as _compute_metrics

logger = logging.getLogger(__name__)


async def compute_portfolio_metrics_node(state: PortfolioState) -> dict:
    """
    Compute all portfolio-level risk metrics.
    Results are stored in portfolio_metrics with tool_call_id for grounding verification.
    """
    portfolio = state.get("portfolio", [])
    raw_market_data = state.get("raw_market_data", {})
    tool_call_registry = dict(state.get("tool_call_registry", {}))
    errors = []

    # Only include validated holdings
    validated_holdings = [h for h in portfolio if h["symbol"].upper() in raw_market_data]

    if not validated_holdings:
        return {
            "portfolio_metrics": {"error": "No validated holdings to compute metrics for"},
            "errors": ["No validated holdings available for portfolio metrics"],
        }

    call_id = f"portfolio_metrics_{uuid.uuid4().hex[:8]}"
    try:
        metrics = await _compute_metrics(validated_holdings, period_days=365)
        tool_call_registry[call_id] = {
            "tool": "compute_portfolio_metrics",
            "args": {"holdings": validated_holdings},
            "output": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        metrics["tool_call_id"] = call_id
        logger.info(f"Portfolio metrics computed [call_id={call_id}]: Sharpe={metrics.get('metrics', {}).get('sharpe_ratio', 'N/A')}")

    except Exception as e:
        errors.append(f"Portfolio metrics computation failed: {str(e)}")
        metrics = {"error": str(e)}

    return {
        "portfolio_metrics": metrics,
        "tool_call_registry": tool_call_registry,
        "errors": errors,
    }
