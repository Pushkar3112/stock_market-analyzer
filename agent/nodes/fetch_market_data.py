"""
fetch_market_data node — validates all symbols and fetches equity details.
First node in the graph. Populates raw_market_data and tool_call_registry.
"""
import uuid
import logging
from datetime import datetime

from ..state import PortfolioState
from ...mcp_server.tools.validate_symbol import validate_symbol
from ...mcp_server.tools.market_data import get_equity_details

logger = logging.getLogger(__name__)


async def fetch_market_data(state: PortfolioState) -> dict:
    """
    1. Validate all portfolio symbols — halt if any fail.
    2. Fetch current equity details for each valid symbol.
    3. Populate raw_market_data and tool_call_registry.
    """
    portfolio = state.get("portfolio", [])
    raw_market_data = {}
    tool_call_registry = dict(state.get("tool_call_registry", {}))
    errors = []

    logger.info(f"Fetching market data for {len(portfolio)} holdings")

    for holding in portfolio:
        symbol = holding["symbol"].upper()

        # --- Guardrail 5.1: Symbol validation MUST run first ---
        validation_id = f"validate_{symbol}_{uuid.uuid4().hex[:8]}"
        try:
            validation = await validate_symbol(symbol)
            tool_call_registry[validation_id] = {
                "tool": "validate_symbol",
                "args": {"symbol": symbol},
                "output": validation,
                "timestamp": datetime.now().isoformat(),
            }

            if not validation["valid"]:
                errors.append(f"Symbol '{symbol}' failed validation: {validation['message']}")
                logger.warning(f"Rejected symbol: {symbol} — {validation['message']}")
                continue

        except Exception as e:
            errors.append(f"Symbol validation error for '{symbol}': {str(e)}")
            continue

        # --- Fetch equity details ---
        equity_id = f"equity_{symbol}_{uuid.uuid4().hex[:8]}"
        try:
            equity_data = await get_equity_details(symbol)
            tool_call_registry[equity_id] = {
                "tool": "get_equity_details",
                "args": {"symbol": symbol},
                "output": equity_data,
                "timestamp": datetime.now().isoformat(),
            }
            raw_market_data[symbol] = {
                "equity_details": equity_data,
                "equity_tool_call_id": equity_id,
                "validation_tool_call_id": validation_id,
            }

        except Exception as e:
            errors.append(f"Failed to fetch equity data for '{symbol}': {str(e)}")
            logger.error(f"Equity fetch error for {symbol}: {e}")

    logger.info(f"Fetched data for {len(raw_market_data)}/{len(portfolio)} symbols")

    return {
        "raw_market_data": raw_market_data,
        "tool_call_registry": tool_call_registry,
        "errors": errors,
    }
