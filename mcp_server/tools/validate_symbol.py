"""Symbol validation tool — guardrail dependency for all other tools."""
import logging
from .market_data import get_all_stock_symbols

logger = logging.getLogger(__name__)

# Cache the symbols list to avoid repeated fetches
_symbol_cache: set[str] | None = None


async def _get_symbol_set() -> set[str]:
    global _symbol_cache
    if _symbol_cache is None:
        result = await get_all_stock_symbols()
        _symbol_cache = {s.upper() for s in result.get("symbols", [])}
    return _symbol_cache


async def validate_symbol(symbol: str) -> dict:
    """
    Validate that a symbol exists on NSE.
    This MUST be called before any other tool for any ticker.
    Returns: {valid: bool, symbol: str, canonical: str, message: str}
    """
    if not symbol or not isinstance(symbol, str):
        return {
            "valid": False,
            "symbol": symbol,
            "canonical": None,
            "message": "Symbol must be a non-empty string.",
        }

    cleaned = symbol.upper().strip().replace(".NS", "").replace(".BO", "")

    try:
        known_symbols = await _get_symbol_set()

        if cleaned in known_symbols:
            return {
                "valid": True,
                "symbol": cleaned,
                "canonical": cleaned,
                "message": f"Symbol '{cleaned}' is valid on NSE.",
            }

        # Fuzzy match: check if user provided a partial name
        close_matches = [s for s in known_symbols if cleaned in s or s in cleaned]
        if close_matches:
            return {
                "valid": False,
                "symbol": symbol,
                "canonical": None,
                "message": (
                    f"Symbol '{cleaned}' not found. Did you mean one of: {close_matches[:3]}? "
                    "Resubmit with the correct symbol."
                ),
            }

        return {
            "valid": False,
            "symbol": symbol,
            "canonical": None,
            "message": f"Symbol '{cleaned}' does not exist on NSE. Hallucinated ticker rejected.",
        }

    except Exception as e:
        logger.error(f"validate_symbol error: {e}")
        return {
            "valid": False,
            "symbol": symbol,
            "canonical": None,
            "message": f"Validation error: {str(e)}",
        }
