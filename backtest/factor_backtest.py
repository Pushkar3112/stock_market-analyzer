"""
Backtesting module — factor-screen basket vs Nifty 50.
Produces the headline Sharpe ratio and max drawdown chart for resume/portfolio.
Run: python -m backtest.factor_backtest
"""
import asyncio
import sys
import logging
from datetime import datetime
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf

sys.path.insert(0, ".")
from mcp_server.tools.factor_screen import factor_screen

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.065  # 6.5% annual
BENCHMARK_SYMBOL = "^NSEI"  # Nifty 50


async def run_factor_backtest(
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    top_n: int = 15,
    rebalance_months: int = 3,
) -> dict:
    """
    Backtest the factor-screened portfolio vs Nifty 50.
    Rebalances every N months using fresh factor scores.

    Returns:
        sharpe_ratio, max_drawdown, cumulative_returns, benchmark_returns, trades
    """
    logger.info(f"Running factor backtest: {start_date} to {end_date}")

    # ── Generate rebalance dates ──
    dates = pd.date_range(start=start_date, end=end_date, freq=f"{rebalance_months}MS")

    all_portfolio_returns = []
    all_benchmark_returns = []
    all_dates = []
    rebalance_log = []

    for i, rebal_date in enumerate(dates[:-1]):
        period_start = rebal_date.strftime("%Y-%m-%d")
        period_end = dates[i + 1].strftime("%Y-%m-%d")

        logger.info(f"Rebalancing period {i+1}/{len(dates)-1}: {period_start} → {period_end}")

        # ── Factor screen at rebalance date ──
        try:
            screen_result = await factor_screen(
                universe=[],
                factors=["momentum", "quality", "low_volatility"],
                top_n=top_n,
            )
            candidates = screen_result.get("candidates", [])
            if not candidates:
                logger.warning(f"No candidates in period {period_start} — skipping")
                continue

            symbols = [c["symbol"] for c in candidates]

        except Exception as e:
            logger.error(f"Factor screen failed for {period_start}: {e}")
            continue

        # ── Download price data for the period ──
        yf_symbols = [f"{s}.NS" for s in symbols]
        try:
            raw = yf.download(
                yf_symbols + [BENCHMARK_SYMBOL],
                start=period_start,
                end=period_end,
                auto_adjust=True,
                progress=False,
            )

            if raw.empty:
                continue

            prices = raw["Close"].copy()
            prices.columns = [c.replace(".NS", "") for c in prices.columns]

            benchmark_col = BENCHMARK_SYMBOL.replace("^", "")
            benchmark_prices = prices.get(benchmark_col) or prices.get(BENCHMARK_SYMBOL)

            portfolio_symbols = [s for s in symbols if s in prices.columns]
            if len(portfolio_symbols) < 3:
                logger.warning(f"Too few symbols with data in {period_start}: {portfolio_symbols}")
                continue

            # Equal-weight portfolio
            port_prices = prices[portfolio_symbols].dropna(how="all")
            port_daily = port_prices.pct_change().dropna()
            port_returns = port_daily.mean(axis=1)  # equal weight

            # Benchmark returns
            if benchmark_prices is not None and not benchmark_prices.empty:
                bench_daily = benchmark_prices.pct_change().dropna()
                # Align dates
                common_dates = port_returns.index.intersection(bench_daily.index)
                port_returns = port_returns.loc[common_dates]
                bench_returns = bench_daily.loc[common_dates]
            else:
                bench_returns = pd.Series(0, index=port_returns.index)

            all_portfolio_returns.append(port_returns)
            all_benchmark_returns.append(bench_returns)
            rebalance_log.append({
                "period": period_start,
                "symbols": portfolio_symbols,
                "top_candidate": symbols[0] if symbols else None,
            })

        except Exception as e:
            logger.error(f"Price fetch failed for {period_start}: {e}")
            continue

    if not all_portfolio_returns:
        return {"error": "No valid periods for backtesting"}

    # ── Combine all periods ──
    combined_portfolio = pd.concat(all_portfolio_returns)
    combined_benchmark = pd.concat(all_benchmark_returns)
    common_idx = combined_portfolio.index.intersection(combined_benchmark.index)
    combined_portfolio = combined_portfolio[common_idx]
    combined_benchmark = combined_benchmark[common_idx]

    # ── Compute metrics ──
    def compute_metrics(returns: pd.Series, label: str) -> dict:
        ann_ret = float(returns.mean() * 252)
        ann_vol = float(returns.std() * np.sqrt(252))
        sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = float(drawdown.min())

        return {
            "label": label,
            "annualized_return_pct": round(ann_ret * 100, 2),
            "annualized_volatility_pct": round(ann_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "cumulative_return_pct": round((float((1 + returns).prod()) - 1) * 100, 2),
        }

    portfolio_metrics = compute_metrics(combined_portfolio, "Factor Strategy")
    benchmark_metrics = compute_metrics(combined_benchmark, "Nifty 50")

    logger.info(f"\nPortfolio: {portfolio_metrics}")
    logger.info(f"Benchmark: {benchmark_metrics}")

    # ── Generate chart ──
    chart_path = _generate_chart(combined_portfolio, combined_benchmark, portfolio_metrics, benchmark_metrics)

    result = {
        "backtest_period": {"start": start_date, "end": end_date},
        "rebalance_frequency_months": rebalance_months,
        "portfolio_strategy": portfolio_metrics,
        "benchmark_nifty_50": benchmark_metrics,
        "alpha": round(portfolio_metrics["annualized_return_pct"] - benchmark_metrics["annualized_return_pct"], 2),
        "rebalance_log": rebalance_log,
        "chart_path": chart_path,
        "note": "Past performance does not guarantee future results.",
    }

    # Save results
    with open("backtest/results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"Backtest complete. Chart saved to {chart_path}")
    return result


def _generate_chart(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    portfolio_metrics: dict,
    benchmark_metrics: dict,
) -> str:
    """Generate the headline cumulative return chart vs Nifty 50."""
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0f0f1a")
    ax1.set_facecolor("#0f0f1a")
    ax2.set_facecolor("#0f0f1a")

    # Cumulative returns
    port_cum = (1 + portfolio_returns).cumprod() - 1
    bench_cum = (1 + benchmark_returns).cumprod() - 1

    ax1.plot(port_cum.index, port_cum * 100, color="#00d4ff", linewidth=2.5, label="Factor Strategy")
    ax1.plot(bench_cum.index, bench_cum * 100, color="#ff6b35", linewidth=2.0, linestyle="--", label="Nifty 50")
    ax1.fill_between(port_cum.index, bench_cum * 100, port_cum * 100,
                     where=(port_cum > bench_cum), alpha=0.15, color="#00d4ff", label="Outperformance")

    # Annotations
    annotation_text = (
        f"Factor Strategy   Sharpe: {portfolio_metrics['sharpe_ratio']:.2f}  |  "
        f"Return: {portfolio_metrics['cumulative_return_pct']:+.1f}%  |  "
        f"Max DD: {portfolio_metrics['max_drawdown_pct']:.1f}%\n"
        f"Nifty 50              Sharpe: {benchmark_metrics['sharpe_ratio']:.2f}  |  "
        f"Return: {benchmark_metrics['cumulative_return_pct']:+.1f}%  |  "
        f"Max DD: {benchmark_metrics['max_drawdown_pct']:.1f}%"
    )
    ax1.text(0.01, 0.97, annotation_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment="top", color="white",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", alpha=0.8))

    ax1.set_title("Factor-Screened Strategy vs Nifty 50 — Backtested Returns",
                  color="white", fontsize=14, fontweight="bold", pad=15)
    ax1.set_ylabel("Cumulative Return (%)", color="white")
    ax1.tick_params(colors="white")
    ax1.spines["bottom"].set_color("#444")
    ax1.spines["left"].set_color("#444")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", framealpha=0.8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Drawdown chart
    port_drawdown = (port_cum + 1) / (port_cum + 1).expanding().max() - 1
    ax2.fill_between(port_drawdown.index, port_drawdown * 100, 0,
                     color="#00d4ff", alpha=0.3, label="Strategy Drawdown")
    ax2.plot(port_drawdown.index, port_drawdown * 100, color="#00d4ff", linewidth=1)
    ax2.set_ylabel("Drawdown (%)", color="white")
    ax2.set_xlabel("Date", color="white")
    ax2.tick_params(colors="white")
    ax2.spines["bottom"].set_color("#444")
    ax2.spines["left"].set_color("#444")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout(pad=2.0)

    import os
    os.makedirs("backtest", exist_ok=True)
    chart_path = "backtest/factor_strategy_vs_nifty50.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    return chart_path


if __name__ == "__main__":
    import os
    result = asyncio.run(run_factor_backtest(
        start_date=os.getenv("BACKTEST_START_DATE", "2023-01-01"),
        end_date=os.getenv("BACKTEST_END_DATE", "2024-12-31"),
        top_n=15,
        rebalance_months=3,
    ))

    print("\nBacktest Results:")
    print(f"  Factor Strategy Sharpe  : {result.get('portfolio_strategy', {}).get('sharpe_ratio', 'N/A')}")
    print(f"  Nifty 50 Sharpe         : {result.get('benchmark_nifty_50', {}).get('sharpe_ratio', 'N/A')}")
    print(f"  Factor Strategy Return  : {result.get('portfolio_strategy', {}).get('cumulative_return_pct', 'N/A')}%")
    print(f"  Nifty 50 Return         : {result.get('benchmark_nifty_50', {}).get('cumulative_return_pct', 'N/A')}%")
    print(f"  Alpha (annualized)      : {result.get('alpha', 'N/A')}%")
    print(f"  Max Drawdown (strategy) : {result.get('portfolio_strategy', {}).get('max_drawdown_pct', 'N/A')}%")
    print(f"\n  Chart saved to: {result.get('chart_path', 'N/A')}")
