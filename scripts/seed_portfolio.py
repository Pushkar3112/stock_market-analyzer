"""
Seed portfolio script — loads a sample portfolio and runs a quick analysis.
Run: python scripts/seed_portfolio.py
"""
import asyncio
import sys
sys.path.insert(0, ".")

SAMPLE_PORTFOLIOS = {
    "nifty_blue_chips": [
        {"symbol": "RELIANCE",  "quantity": 10, "avg_price": 2800.0},
        {"symbol": "TCS",       "quantity": 5,  "avg_price": 3500.0},
        {"symbol": "HDFCBANK",  "quantity": 20, "avg_price": 1600.0},
        {"symbol": "INFY",      "quantity": 15, "avg_price": 1400.0},
        {"symbol": "ICICIBANK", "quantity": 25, "avg_price": 950.0},
    ],
    "diversified_mid": [
        {"symbol": "TITAN",      "quantity": 8,  "avg_price": 3200.0},
        {"symbol": "BAJFINANCE", "quantity": 3,  "avg_price": 7000.0},
        {"symbol": "SUNPHARMA",  "quantity": 12, "avg_price": 1100.0},
        {"symbol": "WIPRO",      "quantity": 30, "avg_price": 450.0},
        {"symbol": "AXISBANK",   "quantity": 20, "avg_price": 1000.0},
    ],
}


async def main(portfolio_name: str = "nifty_blue_chips"):
    from agent.graph import run_portfolio_analysis

    portfolio = SAMPLE_PORTFOLIOS.get(portfolio_name)
    if not portfolio:
        print(f"Unknown portfolio: {portfolio_name}. Options: {list(SAMPLE_PORTFOLIOS.keys())}")
        return

    print(f"\nRunning analysis for portfolio: '{portfolio_name}'")
    print(f"Holdings: {[h['symbol'] for h in portfolio]}\n")

    result = await run_portfolio_analysis(
        portfolio=portfolio,
        trigger_type="manual_query",
        thread_id=f"seed_{portfolio_name}",
    )

    print("\n--- FINAL REPORT ---")
    print(result.get("report_draft", "No report generated"))


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "nifty_blue_chips"
    asyncio.run(main(name))
