"""
compute_technical_indicators node — deterministic indicator calculation.
Never LLM-estimated. Uses pandas-ta via the MCP tool.
"""
import uuid
import logging
from datetime import datetime

from ..state import PortfolioState
from mcp_server.tools.technical_indicators import compute_technical_indicators as _compute

logger = logging.getLogger(__name__)


async def compute_technical_indicators_node(state: PortfolioState) -> dict:
    """
    Compute RSI, MACD, SMA_50, SMA_200, Bollinger Bands, EMA_20 for each portfolio symbol.
    Every value maps to a tool_call_id in the registry.
    """
    raw_market_data = state.get("raw_market_data", {})
    tool_call_registry = dict(state.get("tool_call_registry", {}))
    technical_indicators = {}
    errors = []

    for symbol in raw_market_data:
        call_id = f"tech_indicators_{symbol}_{uuid.uuid4().hex[:8]}"
        try:
            result = await _compute(symbol, indicators=["ALL"], period_days=365)
            tool_call_registry[call_id] = {
                "tool": "compute_technical_indicators",
                "args": {"symbol": symbol, "indicators": ["ALL"]},
                "output": result,
                "timestamp": datetime.now().isoformat(),
            }
            technical_indicators[symbol] = {
                **result,
                "tool_call_id": call_id,
            }
            logger.info(f"Computed indicators for {symbol} [call_id={call_id}]")
        except Exception as e:
            errors.append(f"Technical indicator computation failed for '{symbol}': {str(e)}")
            logger.error(f"Indicator error for {symbol}: {e}")

    return {
        "technical_indicators": technical_indicators,
        "tool_call_registry": tool_call_registry,
        "errors": errors,
    }
