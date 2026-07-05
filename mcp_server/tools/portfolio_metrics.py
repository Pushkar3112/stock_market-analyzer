"""
Portfolio metrics — pure math, fully deterministic.
Sharpe ratio, annualized volatility, max drawdown, VaR(95%), sector concentration.
All values reproducible to floating-point precision.
"""
import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from .market_data import NSE_SECTOR_MAP

logger = logging.getLogger(__name__)

RISK_FREE_RATE_ANNUAL = 0.065  # ~RBI repo rate / 91-day T-bill


async def compute_portfolio_metrics(
    holdings: list[dict],
    period_days: int = 365,
) -> dict[str, Any]:
    """
    Compute portfolio-level risk metrics.
    All metrics are pure math — independent of LLM output.
    
    Returns:
        sharpe_ratio, annualized_volatility, max_drawdown, var_95,
        sector_exposure, portfolio_value, individual_weights
    """
    if not holdings:
        return {"error": "No holdings provided"}

    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days + 10)

    # --- Fetch price data for all holdings ---
    symbols = [h["symbol"].upper() for h in holdings]
    yf_symbols = [f"{s}.NS" for s in symbols]

    try:
        raw = yf.download(
            yf_symbols,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            return {"error": "Failed to fetch price data for portfolio"}

        # Handle single vs multiple tickers
        if len(symbols) == 1:
            prices = raw[["Close"]].rename(columns={"Close": symbols[0]})
        else:
            prices = raw["Close"].copy()
            prices.columns = [c.replace(".NS", "") for c in prices.columns]

        prices = prices.dropna(how="all")

    except Exception as e:
        logger.error(f"Price fetch failed: {e}")
        return {"error": f"Price fetch failed: {str(e)}"}

    # --- Current portfolio values ---
    holding_map = {h["symbol"].upper(): h for h in holdings}
    portfolio_breakdown = []
    total_value = 0.0

    for sym in symbols:
        holding = holding_map[sym]
        qty = holding["quantity"]
        avg_price = holding["avg_price"]

        if sym in prices.columns and not prices[sym].empty:
            current_price = float(prices[sym].dropna().iloc[-1])
        else:
            current_price = avg_price

        current_value = qty * current_price
        cost_basis = qty * avg_price
        pnl = current_value - cost_basis
        pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0

        total_value += current_value
        portfolio_breakdown.append({
            "symbol": sym,
            "quantity": qty,
            "avg_price": round(avg_price, 2),
            "current_price": round(current_price, 2),
            "current_value": round(current_value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "sector": NSE_SECTOR_MAP.get(sym, "Unknown"),
        })

    # Weights
    for item in portfolio_breakdown:
        item["weight"] = round(item["current_value"] / total_value, 4) if total_value > 0 else 0

    # --- Daily returns ---
    available = [s for s in symbols if s in prices.columns]
    if not available:
        return {"error": "No price data available for any holding"}

    weights = np.array([
        holding_map[s]["quantity"] * float(prices[s].dropna().iloc[-1])
        for s in available
    ])
    weights = weights / weights.sum()

    daily_returns = prices[available].pct_change().dropna()
    portfolio_returns = (daily_returns * weights).sum(axis=1)

    # --- Sharpe Ratio ---
    annual_return = float(portfolio_returns.mean() * 252)
    annual_vol = float(portfolio_returns.std() * np.sqrt(252))
    sharpe = (annual_return - RISK_FREE_RATE_ANNUAL) / annual_vol if annual_vol > 0 else 0.0

    # --- Max Drawdown ---
    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    # --- VaR 95% (Historical) ---
    var_95 = float(np.percentile(portfolio_returns, 5))

    # --- Sector Exposure ---
    sector_exposure: dict[str, float] = {}
    for item in portfolio_breakdown:
        sector = item["sector"]
        weight = item["weight"]
        sector_exposure[sector] = round(sector_exposure.get(sector, 0) + weight, 4)

    max_sector = max(sector_exposure.items(), key=lambda x: x[1])
    max_stock_weight = max(portfolio_breakdown, key=lambda x: x["weight"])

    # --- Severity Flags ---
    severity_flags = []
    if abs(var_95) * 100 > 5.0:
        severity_flags.append(f"VaR(95%) of {var_95*100:.2f}% exceeds 5% threshold")
    if max_sector[1] > 0.40:
        severity_flags.append(f"Sector '{max_sector[0]}' at {max_sector[1]*100:.1f}% exceeds 40% limit")
    if max_stock_weight["weight"] > 0.25:
        severity_flags.append(f"Stock '{max_stock_weight['symbol']}' at {max_stock_weight['weight']*100:.1f}% exceeds 25% limit")

    return {
        "portfolio_value_inr": round(total_value, 2),
        "period_days": period_days,
        "metrics": {
            "sharpe_ratio": round(sharpe, 4),
            "annualized_return_pct": round(annual_return * 100, 2),
            "annualized_volatility_pct": round(annual_vol * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "var_95_pct": round(var_95 * 100, 2),
            "risk_free_rate_used": RISK_FREE_RATE_ANNUAL,
        },
        "sector_exposure": {k: round(v * 100, 2) for k, v in sector_exposure.items()},
        "top_sector": {"name": max_sector[0], "weight_pct": round(max_sector[1] * 100, 2)},
        "holdings": portfolio_breakdown,
        "severity_flags": severity_flags,
        "requires_high_severity": len(severity_flags) > 0,
        "data_source": "yfinance + numpy",
        "timestamp": datetime.now().isoformat(),
    }
