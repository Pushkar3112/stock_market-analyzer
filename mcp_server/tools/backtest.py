"""
Backtest alert rules against historical data.
Produces hit-rate (% correct direction), trade log, and Sharpe of signal-based strategy.
This is the source of the alert hit-rate metric — binary ground truth, no LLM judge.
"""
import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


async def backtest_alert_rule(
    symbol: str,
    rule: dict,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """
    Replay a signal rule against historical data.
    
    Supported rule types:
      - rsi_oversold: {"type": "rsi_oversold", "threshold": 30, "hold_days": 5}
      - rsi_overbought: {"type": "rsi_overbought", "threshold": 70, "hold_days": 5}
      - golden_cross: {"type": "golden_cross", "hold_days": 20}
      - death_cross: {"type": "death_cross", "hold_days": 20}
      - macd_crossover: {"type": "macd_crossover", "hold_days": 10}
      - bb_breakout: {"type": "bb_breakout", "direction": "upper|lower", "hold_days": 5}
    
    Returns:
      hit_rate, trades, sharpe_ratio, max_drawdown, total_signals
    """
    yf_symbol = f"{symbol.upper()}.NS"

    try:
        # Fetch historical data (need more history for indicator warmup)
        fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=250)).strftime("%Y-%m-%d")
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=fetch_start, end=end_date)

        if df.empty or len(df) < 100:
            return {
                "symbol": symbol,
                "error": "Insufficient historical data for backtesting",
                "rule": rule,
            }

        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        close = df["Close"]

        # --- Compute indicators needed for the rule ---
        rule_type = rule.get("type", "")
        hold_days = rule.get("hold_days", 5)
        signals_df = _generate_signals(df, close, rule_type, rule)

        # --- Filter to backtest window ---
        bt_start = pd.Timestamp(start_date)
        bt_end = pd.Timestamp(end_date)
        signals_df = signals_df[(signals_df["Date"] >= bt_start) & (signals_df["Date"] <= bt_end)]
        signal_rows = signals_df[signals_df["signal"] == 1]

        if len(signal_rows) == 0:
            return {
                "symbol": symbol,
                "rule": rule,
                "start_date": start_date,
                "end_date": end_date,
                "total_signals": 0,
                "message": "No signals triggered in this period with the given rule.",
                "hit_rate_pct": None,
                "sharpe_ratio": None,
            }

        # --- Trade simulation ---
        trades = []
        returns_list = []

        for _, row in signal_rows.iterrows():
            entry_date = row["Date"]
            entry_price = row["Close"]

            # Find exit price N days later
            future = signals_df[signals_df["Date"] > entry_date].head(hold_days)
            if future.empty:
                continue

            exit_price = float(future["Close"].iloc[-1])
            exit_date = future["Date"].iloc[-1]

            trade_return = (exit_price - entry_price) / entry_price
            direction_correct = trade_return > 0  # for long signals

            trades.append({
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "exit_date": exit_date.strftime("%Y-%m-%d"),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "return_pct": round(trade_return * 100, 2),
                "direction_correct": direction_correct,
            })
            returns_list.append(trade_return)

        if not trades:
            return {"symbol": symbol, "rule": rule, "error": "No complete trades in period"}

        returns_arr = np.array(returns_list)
        hit_rate = float(np.mean([t["direction_correct"] for t in trades]))
        mean_return = float(returns_arr.mean())
        vol = float(returns_arr.std()) if len(returns_arr) > 1 else 0.001
        sharpe = (mean_return / vol) * np.sqrt(252 / hold_days) if vol > 0 else 0

        # Max drawdown of strategy equity curve
        equity = (1 + returns_arr).cumprod()
        roll_max = np.maximum.accumulate(equity)
        drawdown = (equity - roll_max) / roll_max
        max_dd = float(drawdown.min())

        return {
            "symbol": symbol,
            "rule": rule,
            "backtest_period": {"start": start_date, "end": end_date},
            "total_signals": len(signal_rows),
            "total_trades": len(trades),
            "hit_rate_pct": round(hit_rate * 100, 2),
            "avg_return_per_trade_pct": round(mean_return * 100, 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "trades": trades[:20],  # cap at 20 for readability
            "note": (
                "Hit rate = % of signals where price moved in predicted direction over hold_days. "
                "Sharpe is annualized. This is historical — past performance does not predict future results."
            ),
        }

    except Exception as e:
        logger.error(f"backtest_alert_rule failed: {e}")
        raise


def _generate_signals(df: pd.DataFrame, close: pd.Series, rule_type: str, rule: dict) -> pd.DataFrame:
    """Generate entry signal column based on rule type."""
    out = df[["Date", "Close", "High", "Low", "Volume"]].copy()
    out["signal"] = 0

    if rule_type == "rsi_oversold":
        threshold = rule.get("threshold", 30)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        out["signal"] = ((rsi < threshold) & (rsi.shift(1) >= threshold)).astype(int)

    elif rule_type == "rsi_overbought":
        threshold = rule.get("threshold", 70)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        out["signal"] = ((rsi > threshold) & (rsi.shift(1) <= threshold)).astype(int)

    elif rule_type == "golden_cross":
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        out["signal"] = ((sma50 > sma200) & (sma50.shift(1) <= sma200.shift(1))).astype(int)

    elif rule_type == "death_cross":
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        out["signal"] = ((sma50 < sma200) & (sma50.shift(1) >= sma200.shift(1))).astype(int)

    elif rule_type == "macd_crossover":
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        out["signal"] = ((macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))).astype(int)

    elif rule_type == "bb_breakout":
        direction = rule.get("direction", "upper")
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        if direction == "upper":
            out["signal"] = ((close > upper) & (close.shift(1) <= upper.shift(1))).astype(int)
        else:
            out["signal"] = ((close < lower) & (close.shift(1) >= lower.shift(1))).astype(int)

    return out
