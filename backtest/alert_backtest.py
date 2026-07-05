"""
Alert hit-rate backtest — replay all triggered alert types for a symbol
and measure what % of signals led to price moving in the predicted direction.
Run: python -m backtest.alert_backtest
"""
import asyncio
import json
import sys
import logging
from datetime import datetime

sys.path.insert(0, ".")
from mcp_server.tools.backtest import backtest_alert_rule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configurations: symbol + rule + expected direction
BACKTEST_CONFIGS = [
    {
        "symbol": "RELIANCE",
        "rule": {"type": "rsi_oversold", "threshold": 35, "hold_days": 5},
        "description": "RSI oversold bounce — RELIANCE",
    },
    {
        "symbol": "TCS",
        "rule": {"type": "macd_crossover", "hold_days": 10},
        "description": "MACD bullish crossover — TCS",
    },
    {
        "symbol": "HDFCBANK",
        "rule": {"type": "golden_cross", "hold_days": 20},
        "description": "Golden Cross (SMA50 > SMA200) — HDFCBANK",
    },
    {
        "symbol": "INFY",
        "rule": {"type": "bb_breakout", "direction": "lower", "hold_days": 5},
        "description": "Bollinger Band lower breakout — INFY",
    },
    {
        "symbol": "ICICIBANK",
        "rule": {"type": "rsi_overbought", "threshold": 68, "hold_days": 5},
        "description": "RSI overbought reversal — ICICIBANK",
    },
]


async def run_all_backtests(
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
) -> dict:
    """Run all alert rule backtests and aggregate hit-rates."""
    results = []

    for cfg in BACKTEST_CONFIGS:
        logger.info(f"Backtesting: {cfg['description']}")
        try:
            result = await backtest_alert_rule(
                symbol=cfg["symbol"],
                rule=cfg["rule"],
                start_date=start_date,
                end_date=end_date,
            )
            results.append({
                "description": cfg["description"],
                "symbol": cfg["symbol"],
                "rule_type": cfg["rule"]["type"],
                "total_signals": result.get("total_signals", 0),
                "hit_rate_pct": result.get("hit_rate_pct"),
                "sharpe_ratio": result.get("sharpe_ratio"),
                "avg_return_per_trade_pct": result.get("avg_return_per_trade_pct"),
                "max_drawdown_pct": result.get("max_drawdown_pct"),
                "error": result.get("error") or result.get("message"),
            })
        except Exception as e:
            results.append({
                "description": cfg["description"],
                "symbol": cfg["symbol"],
                "error": str(e),
            })

    # Aggregate
    valid = [r for r in results if r.get("hit_rate_pct") is not None]
    avg_hit_rate = sum(r["hit_rate_pct"] for r in valid) / len(valid) if valid else None
    avg_sharpe = sum(r["sharpe_ratio"] for r in valid if r.get("sharpe_ratio")) / len(valid) if valid else None

    summary = {
        "backtest_period": {"start": start_date, "end": end_date},
        "total_rules_tested": len(BACKTEST_CONFIGS),
        "rules_with_data": len(valid),
        "avg_hit_rate_pct": round(avg_hit_rate, 2) if avg_hit_rate else None,
        "avg_sharpe_ratio": round(avg_sharpe, 4) if avg_sharpe else None,
        "individual_results": results,
    }

    # Print table
    print("\n" + "=" * 70)
    print("ALERT HIT-RATE BACKTEST RESULTS")
    print("=" * 70)
    print(f"{'Description':<40} {'Signals':>8} {'Hit Rate':>10} {'Sharpe':>8}")
    print("-" * 70)
    for r in results:
        signals = r.get("total_signals", 0) or "N/A"
        hit_rate = f"{r['hit_rate_pct']:.1f}%" if r.get("hit_rate_pct") is not None else "N/A"
        sharpe = f"{r['sharpe_ratio']:.3f}" if r.get("sharpe_ratio") is not None else "N/A"
        print(f"{r['description']:<40} {str(signals):>8} {hit_rate:>10} {sharpe:>8}")

    print("-" * 70)
    print(f"{'AVERAGE':.<40} {'':>8} {f'{avg_hit_rate:.1f}%' if avg_hit_rate else 'N/A':>10} {f'{avg_sharpe:.3f}' if avg_sharpe else 'N/A':>8}")
    print("=" * 70)

    # Save results
    import os
    os.makedirs("backtest", exist_ok=True)
    with open("backtest/alert_backtest_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to backtest/alert_backtest_results.json")

    return summary


if __name__ == "__main__":
    import os
    result = asyncio.run(run_all_backtests(
        start_date=os.getenv("BACKTEST_START_DATE", "2023-01-01"),
        end_date=os.getenv("BACKTEST_END_DATE", "2024-12-31"),
    ))
