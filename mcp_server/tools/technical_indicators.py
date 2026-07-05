"""
Technical indicators — fully deterministic, computed from OHLCV data.
Never LLM-estimated. Uses pandas-ta for calculation.
"""
import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logging.warning("pandas-ta not available, using manual indicator calculations")

logger = logging.getLogger(__name__)


def _manual_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else None


def _manual_sma(close: pd.Series, period: int) -> float:
    if len(close) < period:
        return None
    return float(close.rolling(period).mean().iloc[-1])


def _manual_ema(close: pd.Series, period: int) -> float:
    if len(close) < period:
        return None
    return float(close.ewm(span=period, adjust=False).mean().iloc[-1])


def _manual_macd(close: pd.Series) -> dict:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd": float(macd_line.iloc[-1]),
        "signal": float(signal_line.iloc[-1]),
        "histogram": float(histogram.iloc[-1]),
    }


def _manual_bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> dict:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return {
        "upper": float(upper.iloc[-1]),
        "middle": float(sma.iloc[-1]),
        "lower": float(lower.iloc[-1]),
        "bandwidth": float(((upper - lower) / sma).iloc[-1]),
        "percent_b": float(((close - lower) / (upper - lower)).iloc[-1]),
    }


async def compute_technical_indicators(
    symbol: str,
    indicators: list[str] = None,
    period_days: int = 365,
) -> dict[str, Any]:
    """
    Compute technical indicators for a given NSE symbol.
    All calculations are deterministic — traceable to OHLCV data.
    """
    if indicators is None:
        indicators = ["ALL"]

    compute_all = "ALL" in indicators
    yf_symbol = f"{symbol.upper()}.NS"

    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days + 60)  # extra buffer for indicators

    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

        if df.empty or len(df) < 30:
            return {
                "symbol": symbol,
                "error": "Insufficient data to compute indicators (need >= 30 trading days)",
                "indicators": {},
            }

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        result: dict[str, Any] = {
            "symbol": symbol.upper(),
            "period_days": period_days,
            "data_points": len(df),
            "current_price": round(float(close.iloc[-1]), 2),
            "indicators": {},
            "data_source": "yfinance + pandas_ta",
        }

        ind = result["indicators"]

        # RSI
        if compute_all or "RSI" in indicators:
            if PANDAS_TA_AVAILABLE:
                rsi_series = ta.rsi(close, length=14)
                ind["RSI"] = {
                    "value": round(float(rsi_series.iloc[-1]), 2) if rsi_series is not None else None,
                    "period": 14,
                    "interpretation": _interpret_rsi(float(rsi_series.iloc[-1]) if rsi_series is not None else 50),
                }
            else:
                rsi_val = _manual_rsi(close)
                ind["RSI"] = {
                    "value": round(rsi_val, 2) if rsi_val else None,
                    "period": 14,
                    "interpretation": _interpret_rsi(rsi_val or 50),
                }

        # MACD
        if compute_all or "MACD" in indicators:
            macd_data = _manual_macd(close)
            ind["MACD"] = {
                "macd_line": round(macd_data["macd"], 4),
                "signal_line": round(macd_data["signal"], 4),
                "histogram": round(macd_data["histogram"], 4),
                "interpretation": "bullish" if macd_data["histogram"] > 0 else "bearish",
            }

        # SMA 50
        if compute_all or "SMA_50" in indicators:
            sma50 = _manual_sma(close, 50)
            ind["SMA_50"] = {
                "value": round(sma50, 2) if sma50 else None,
                "price_vs_sma": round(float(close.iloc[-1]) / sma50 - 1, 4) if sma50 else None,
                "interpretation": "above" if sma50 and float(close.iloc[-1]) > sma50 else "below",
            }

        # SMA 200
        if compute_all or "SMA_200" in indicators:
            sma200 = _manual_sma(close, 200)
            ind["SMA_200"] = {
                "value": round(sma200, 2) if sma200 else None,
                "price_vs_sma": round(float(close.iloc[-1]) / sma200 - 1, 4) if sma200 else None,
                "interpretation": "above (bullish long-term)" if sma200 and float(close.iloc[-1]) > sma200 else "below (bearish long-term)",
            }

        # Bollinger Bands
        if compute_all or "BB" in indicators:
            bb = _manual_bollinger(close)
            ind["Bollinger_Bands"] = {
                "upper": round(bb["upper"], 2),
                "middle": round(bb["middle"], 2),
                "lower": round(bb["lower"], 2),
                "bandwidth": round(bb["bandwidth"], 4),
                "percent_b": round(bb["percent_b"], 4),
                "interpretation": _interpret_bb(bb["percent_b"]),
            }

        # EMA 20
        if compute_all or "EMA_20" in indicators:
            ema20 = _manual_ema(close, 20)
            ind["EMA_20"] = {
                "value": round(ema20, 2) if ema20 else None,
                "price_vs_ema": round(float(close.iloc[-1]) / ema20 - 1, 4) if ema20 else None,
            }

        # Volume analysis
        if compute_all:
            avg_volume_20 = float(volume.tail(20).mean())
            current_volume = float(volume.iloc[-1])
            ind["Volume"] = {
                "current": int(current_volume),
                "avg_20d": int(avg_volume_20),
                "ratio": round(current_volume / avg_volume_20, 2) if avg_volume_20 > 0 else None,
                "interpretation": "above average" if current_volume > avg_volume_20 else "below average",
            }

        return result

    except Exception as e:
        logger.error(f"compute_technical_indicators failed for {symbol}: {e}")
        raise


def _interpret_rsi(rsi: float) -> str:
    if rsi >= 70:
        return "overbought — potential reversal or pullback"
    elif rsi >= 60:
        return "strong momentum, watch for overextension"
    elif rsi >= 40:
        return "neutral zone"
    elif rsi >= 30:
        return "weak momentum, near oversold"
    else:
        return "oversold — potential bounce"


def _interpret_bb(percent_b: float) -> str:
    if percent_b > 1.0:
        return "above upper band — strongly overbought"
    elif percent_b > 0.8:
        return "approaching upper band — overbought"
    elif percent_b > 0.2:
        return "within bands — normal range"
    elif percent_b > 0:
        return "approaching lower band — oversold"
    else:
        return "below lower band — strongly oversold"
