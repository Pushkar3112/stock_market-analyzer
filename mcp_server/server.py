"""
MCP Server entry point for NSE India Stock Market Analyst.
Run with: python -m mcp_server.server

All tools are pure Python — no TypeScript or Node.js required.
Data sources: nsepython, yfinance, jugaad-data
"""

import asyncio
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .tools.market_data import (
    get_equity_details,
    get_equity_historical_data,
    get_all_stock_symbols,
    get_indices,
)
from .tools.validate_symbol import validate_symbol
from .tools.technical_indicators import compute_technical_indicators
from .tools.portfolio_metrics import compute_portfolio_metrics
from .tools.factor_screen import factor_screen
from .tools.backtest import backtest_alert_rule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Server("nse-stock-analyst")


# ── Tool definitions ──────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_equity_details",
            description="Get current equity details for an NSE symbol (price, volume, market cap, 52w high/low).",
            inputSchema={
                "type": "object",
                "properties": {"symbol": {"type": "string", "description": "NSE ticker symbol e.g. RELIANCE"}},
                "required": ["symbol"],
            },
        ),
        Tool(
            name="get_equity_historical_data",
            description="Get OHLCV historical data for an NSE symbol between two dates.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["symbol", "start_date", "end_date"],
            },
        ),
        Tool(
            name="get_all_stock_symbols",
            description="Get list of all valid NSE equity symbols.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_indices",
            description="Get current values for NSE indices (NIFTY 50, NIFTY BANK, etc.).",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="validate_symbol",
            description="Validate that a symbol exists on NSE. MUST be called before any other tool for any symbol. Returns {valid: bool, symbol: str, message: str}.",
            inputSchema={
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        ),
        Tool(
            name="compute_technical_indicators",
            description=(
                "Compute technical indicators for an NSE symbol. "
                "Returns RSI, MACD, SMA_50, SMA_200, Bollinger Bands, EMA_20. "
                "All values are deterministically calculated from historical OHLCV — never LLM-estimated."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "indicators": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["RSI", "MACD", "SMA_50", "SMA_200", "BB", "EMA_20", "ALL"]},
                        "description": "List of indicators to compute. Use ['ALL'] for all.",
                    },
                    "period_days": {"type": "integer", "default": 365, "description": "Lookback period in days"},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="compute_portfolio_metrics",
            description=(
                "Compute portfolio-level risk metrics: Sharpe ratio, annualized volatility, "
                "max drawdown, VaR(95%), sector concentration. "
                "Pure math from historical data — fully reproducible."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "holdings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string"},
                                "quantity": {"type": "number"},
                                "avg_price": {"type": "number"},
                            },
                            "required": ["symbol", "quantity", "avg_price"],
                        },
                    },
                    "period_days": {"type": "integer", "default": 365},
                },
                "required": ["holdings"],
            },
        ),
        Tool(
            name="factor_screen",
            description=(
                "Screen the NSE universe using quantitative factors (value, momentum, quality). "
                "Returns candidates with factor scores. NOT LLM opinion — pure quantitative filtering."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "universe": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of NSE symbols to screen. Pass [] to screen full Nifty 500.",
                    },
                    "factors": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["value", "momentum", "quality", "low_volatility"]},
                        "description": "Factors to apply",
                    },
                    "top_n": {"type": "integer", "default": 10, "description": "Number of top candidates to return"},
                },
                "required": ["factors"],
            },
        ),
        Tool(
            name="backtest_alert_rule",
            description=(
                "Replay a signal rule against historical data for a symbol and date range. "
                "Returns hit_rate (% correct direction), trades, and Sharpe of a rule-based strategy."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "rule": {
                        "type": "object",
                        "description": "Signal rule e.g. {type: 'rsi_oversold', threshold: 30, hold_days: 5}",
                    },
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["symbol", "rule", "start_date", "end_date"],
            },
        ),
    ]


# ── Tool call dispatcher ───────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    import json

    try:
        if name == "get_equity_details":
            result = await get_equity_details(arguments["symbol"])
        elif name == "get_equity_historical_data":
            result = await get_equity_historical_data(
                arguments["symbol"],
                arguments["start_date"],
                arguments["end_date"],
            )
        elif name == "get_all_stock_symbols":
            result = await get_all_stock_symbols()
        elif name == "get_indices":
            result = await get_indices()
        elif name == "validate_symbol":
            result = await validate_symbol(arguments["symbol"])
        elif name == "compute_technical_indicators":
            result = await compute_technical_indicators(
                arguments["symbol"],
                arguments.get("indicators", ["ALL"]),
                arguments.get("period_days", 365),
            )
        elif name == "compute_portfolio_metrics":
            result = await compute_portfolio_metrics(
                arguments["holdings"],
                arguments.get("period_days", 365),
            )
        elif name == "factor_screen":
            result = await factor_screen(
                arguments.get("universe", []),
                arguments["factors"],
                arguments.get("top_n", 10),
            )
        elif name == "backtest_alert_rule":
            result = await backtest_alert_rule(
                arguments["symbol"],
                arguments["rule"],
                arguments["start_date"],
                arguments["end_date"],
            )
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, default=str, indent=2))]

    except Exception as e:
        logger.error(f"Tool '{name}' failed: {e}", exc_info=True)
        return [TextContent(type="text", text=json.dumps({"error": str(e), "tool": name}))]


# ── Entry point ────────────────────────────────────────────────────────────────

async def main():
    logger.info("Starting NSE Stock Analyst MCP Server (Python)...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
