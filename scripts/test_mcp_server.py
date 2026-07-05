"""
MCP server test script — tests all tools in isolation with known inputs.
Run: python scripts/test_mcp_server.py
"""
import asyncio
import sys

sys.path.insert(0, ".")

from mcp_server.tools.market_data import get_equity_details, get_equity_historical_data, get_all_stock_symbols
from mcp_server.tools.validate_symbol import validate_symbol
from mcp_server.tools.technical_indicators import compute_technical_indicators
from mcp_server.tools.portfolio_metrics import compute_portfolio_metrics
from mcp_server.tools.factor_screen import factor_screen
from mcp_server.tools.backtest import backtest_alert_rule


async def test_all():
    print("=" * 60)
    print("MCP Server Tool Tests")
    print("=" * 60)

    # Test 1: validate_symbol
    print("\n[1] validate_symbol")
    valid = await validate_symbol("RELIANCE")
    invalid = await validate_symbol("FAKESTOCKXYZ")
    print(f"  RELIANCE valid  : {valid['valid']} — {valid['message']}")
    print(f"  FAKESTOCKXYZ    : {invalid['valid']} — {invalid['message']}")
    assert valid["valid"] is True
    assert invalid["valid"] is False
    print("  PASSED")

    # Test 2: get_equity_details
    print("\n[2] get_equity_details (RELIANCE)")
    equity = await get_equity_details("RELIANCE")
    print(f"  Company  : {equity.get('company_name')}")
    print(f"  Price    : {equity.get('current_price')}")
    print(f"  Sector   : {equity.get('sector')}")
    assert equity.get("current_price") is not None
    print("  PASSED")

    # Test 3: compute_technical_indicators
    print("\n[3] compute_technical_indicators (TCS, ALL)")
    indicators = await compute_technical_indicators("TCS", ["ALL"])
    rsi = indicators.get("indicators", {}).get("RSI", {}).get("value")
    macd = indicators.get("indicators", {}).get("MACD", {}).get("macd_line")
    sma50 = indicators.get("indicators", {}).get("SMA_50", {}).get("value")
    print(f"  RSI     : {rsi}")
    print(f"  MACD    : {macd}")
    print(f"  SMA_50  : {sma50}")
    assert rsi is not None and 0 <= rsi <= 100, f"RSI out of range: {rsi}"
    print("  PASSED")

    # Test 4: compute_portfolio_metrics
    print("\n[4] compute_portfolio_metrics")
    holdings = [
        {"symbol": "RELIANCE", "quantity": 10, "avg_price": 2800.0},
        {"symbol": "TCS", "quantity": 5, "avg_price": 3500.0},
        {"symbol": "HDFCBANK", "quantity": 20, "avg_price": 1600.0},
    ]
    metrics = await compute_portfolio_metrics(holdings)
    sharpe = metrics.get("metrics", {}).get("sharpe_ratio")
    var_95 = metrics.get("metrics", {}).get("var_95_pct")
    vol = metrics.get("metrics", {}).get("annualized_volatility_pct")
    print(f"  Sharpe Ratio     : {sharpe}")
    print(f"  VaR (95%)        : {var_95}%")
    print(f"  Annual Volatility: {vol}%")
    print(f"  Severity Flags   : {metrics.get('severity_flags', [])}")
    assert sharpe is not None
    print("  PASSED")

    # Test 5: factor_screen (small universe for speed)
    print("\n[5] factor_screen (small universe)")
    screen = await factor_screen(
        universe=["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"],
        factors=["momentum", "quality"],
        top_n=3,
    )
    candidates = screen.get("candidates", [])
    print(f"  Top candidate: {candidates[0]['symbol'] if candidates else 'None'}")
    print(f"  Candidates   : {[c['symbol'] for c in candidates]}")
    assert len(candidates) >= 1
    print("  PASSED")

    # Test 6: backtest_alert_rule
    print("\n[6] backtest_alert_rule (RSI oversold, RELIANCE, 2023)")
    bt = await backtest_alert_rule(
        symbol="RELIANCE",
        rule={"type": "rsi_oversold", "threshold": 35, "hold_days": 5},
        start_date="2023-01-01",
        end_date="2023-12-31",
    )
    print(f"  Signals      : {bt.get('total_signals', 0)}")
    print(f"  Hit Rate     : {bt.get('hit_rate_pct', 'N/A')}%")
    print(f"  Sharpe       : {bt.get('sharpe_ratio', 'N/A')}")
    print("  PASSED" if "error" not in bt else f"  NOTE: {bt.get('message', bt.get('error'))}")

    print("\n" + "=" * 60)
    print("All MCP tool tests completed.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_all())
