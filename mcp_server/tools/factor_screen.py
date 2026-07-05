"""
Factor screening — quantitative filtering of NSE universe.
Value, momentum, quality, low-volatility factors.
LLM only explains WHY a stock passed — never decides WHICH stocks to pick.
"""
import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from .market_data import NSE_SECTOR_MAP, get_all_stock_symbols

logger = logging.getLogger(__name__)

# Default screen universe (Nifty 100 for speed)
DEFAULT_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BHARTIARTL", "ITC", "BAJFINANCE", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT",
    "MARUTI", "TITAN", "SUNPHARMA", "WIPRO", "ULTRACEMCO", "NESTLEIND",
    "POWERGRID", "NTPC", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "HCLTECH",
    "TECHM", "ADANIPORTS", "COALINDIA", "ONGC", "DRREDDY", "CIPLA",
    "DIVISLAB", "BAJAJFINSV", "EICHERMOT", "HEROMOTOCO", "BRITANNIA", "SHREECEM",
    "GRASIM", "M&M", "HINDALCO", "BPCL", "IOC", "TATACONSUM",
    "APOLLOHOSP", "PIDILITIND", "TRENT", "IRCTC", "DMART", "BAJAJ-AUTO",
]


async def factor_screen(
    universe: list[str],
    factors: list[str],
    top_n: int = 10,
) -> dict[str, Any]:
    """
    Screen NSE stocks using quantitative factors.
    Returns candidates sorted by composite factor score.
    
    Factors supported: value, momentum, quality, low_volatility
    """
    if not universe:
        universe = DEFAULT_UNIVERSE

    symbols = [s.upper() for s in universe]
    yf_symbols = [f"{s}.NS" for s in symbols]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)  # ~1 year + buffer

    logger.info(f"Screening {len(symbols)} symbols for factors: {factors}")

    try:
        # Batch download
        raw = yf.download(
            yf_symbols,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            return {"error": "Failed to fetch data for factor screening"}

        # Handle single vs multi
        if len(symbols) == 1:
            close_prices = pd.DataFrame({symbols[0]: raw["Close"]})
        else:
            close_prices = raw["Close"].copy()
            close_prices.columns = [c.replace(".NS", "") for c in close_prices.columns]

        close_prices = close_prices.dropna(how="all")
        daily_returns = close_prices.pct_change().dropna()

        scores: dict[str, dict] = {s: {} for s in symbols if s in close_prices.columns}

        for sym in list(scores.keys()):
            prices = close_prices[sym].dropna()
            returns = daily_returns[sym].dropna()

            if len(prices) < 60 or len(returns) < 60:
                del scores[sym]
                continue

            factor_scores = {}

            # --- Momentum Factor ---
            if "momentum" in factors:
                ret_1m = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) >= 21 else 0
                ret_3m = (prices.iloc[-1] / prices.iloc[-63] - 1) if len(prices) >= 63 else 0
                ret_6m = (prices.iloc[-1] / prices.iloc[-126] - 1) if len(prices) >= 126 else 0
                ret_12m = (prices.iloc[-1] / prices.iloc[0] - 1)
                momentum_score = 0.1 * ret_1m + 0.2 * ret_3m + 0.3 * ret_6m + 0.4 * ret_12m
                factor_scores["momentum"] = {
                    "score": round(float(momentum_score), 4),
                    "return_1m_pct": round(float(ret_1m) * 100, 2),
                    "return_3m_pct": round(float(ret_3m) * 100, 2),
                    "return_6m_pct": round(float(ret_6m) * 100, 2),
                    "return_12m_pct": round(float(ret_12m) * 100, 2),
                }

            # --- Low Volatility Factor ---
            if "low_volatility" in factors:
                vol_60d = float(returns.tail(60).std() * np.sqrt(252))
                # Low vol score: inverse of volatility (normalized)
                factor_scores["low_volatility"] = {
                    "score": round(1 / (1 + vol_60d), 4),
                    "annualized_volatility_pct": round(vol_60d * 100, 2),
                }

            # --- Quality Factor (proxy via price stability + trend) ---
            if "quality" in factors:
                sma_50 = float(prices.rolling(50).mean().iloc[-1]) if len(prices) >= 50 else None
                sma_200 = float(prices.rolling(200).mean().iloc[-1]) if len(prices) >= 200 else None
                current = float(prices.iloc[-1])

                above_sma50 = current > sma_50 if sma_50 else False
                above_sma200 = current > sma_200 if sma_200 else False
                golden_cross = (sma_50 > sma_200) if (sma_50 and sma_200) else False

                quality_score = (
                    (0.4 if above_sma50 else 0) +
                    (0.4 if above_sma200 else 0) +
                    (0.2 if golden_cross else 0)
                )
                factor_scores["quality"] = {
                    "score": round(quality_score, 4),
                    "above_sma_50": above_sma50,
                    "above_sma_200": above_sma200,
                    "golden_cross": golden_cross,
                    "sma_50": round(sma_50, 2) if sma_50 else None,
                    "sma_200": round(sma_200, 2) if sma_200 else None,
                }

            # --- Value Factor (proxy via 52-week price position) ---
            if "value" in factors:
                high_52w = float(prices.tail(252).max()) if len(prices) >= 252 else float(prices.max())
                low_52w = float(prices.tail(252).min()) if len(prices) >= 252 else float(prices.min())
                current = float(prices.iloc[-1])
                range_52w = high_52w - low_52w

                # Value score: lower in 52-week range = potentially more value
                pct_from_low = (current - low_52w) / range_52w if range_52w > 0 else 0.5
                value_score = 1 - pct_from_low  # closer to 52w low = higher value score

                factor_scores["value"] = {
                    "score": round(value_score, 4),
                    "current_price": round(current, 2),
                    "52w_high": round(high_52w, 2),
                    "52w_low": round(low_52w, 2),
                    "pct_from_52w_high": round((1 - current / high_52w) * 100, 2),
                    "pct_from_52w_low": round((current / low_52w - 1) * 100, 2),
                }

            # Composite score (equal weight across selected factors)
            if factor_scores:
                composite = np.mean([v["score"] for v in factor_scores.values()])
                scores[sym] = {
                    "composite_score": round(float(composite), 4),
                    "factor_scores": factor_scores,
                    "sector": NSE_SECTOR_MAP.get(sym, "Unknown"),
                    "current_price": round(float(prices.iloc[-1]), 2),
                    "passed_factors": [f for f in factors if f in factor_scores],
                }

        # Sort by composite score descending
        ranked = sorted(
            [{"symbol": k, **v} for k, v in scores.items() if "composite_score" in v],
            key=lambda x: x["composite_score"],
            reverse=True,
        )

        top_candidates = ranked[:top_n]

        return {
            "screened_universe_size": len(symbols),
            "candidates_with_data": len(scores),
            "factors_applied": factors,
            "top_n": top_n,
            "candidates": top_candidates,
            "screening_note": (
                "Candidates ranked by composite quantitative factor score. "
                "This is NOT a buy recommendation — these stocks passed quantitative screens only."
            ),
            "timestamp": datetime.now().isoformat(),
            "data_source": "yfinance + numpy",
        }

    except Exception as e:
        logger.error(f"factor_screen failed: {e}")
        raise
