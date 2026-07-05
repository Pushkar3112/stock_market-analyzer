"""NSE market data tools using yfinance + nsepython."""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# NSE sector mapping for major stocks
NSE_SECTOR_MAP: dict[str, str] = {
    "RELIANCE": "Energy", "TCS": "Technology", "HDFCBANK": "Banking",
    "INFY": "Technology", "ICICIBANK": "Banking", "HINDUNILVR": "FMCG",
    "SBIN": "Banking", "BHARTIARTL": "Telecom", "ITC": "FMCG",
    "BAJFINANCE": "Financial Services", "KOTAKBANK": "Banking",
    "LT": "Construction", "AXISBANK": "Banking", "ASIANPAINT": "Paints",
    "MARUTI": "Automobile", "TITAN": "Consumer Goods", "SUNPHARMA": "Pharma",
    "WIPRO": "Technology", "ULTRACEMCO": "Cement", "NESTLEIND": "FMCG",
    "POWERGRID": "Utilities", "NTPC": "Utilities", "TATAMOTORS": "Automobile",
    "TATASTEEL": "Metal", "JSWSTEEL": "Metal", "HCLTECH": "Technology",
    "TECHM": "Technology", "ADANIPORTS": "Infrastructure", "COALINDIA": "Mining",
    "ONGC": "Energy", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "BAJAJFINSV": "Financial Services", "EICHERMOT": "Automobile",
    "HEROMOTOCO": "Automobile", "BRITANNIA": "FMCG", "SHREECEM": "Cement",
    "GRASIM": "Diversified", "M&M": "Automobile", "HINDALCO": "Metal",
    "VEDL": "Metal", "BPCL": "Energy", "IOC": "Energy",
    "TATACONSUM": "FMCG", "APOLLOHOSP": "Healthcare", "PIDILITIND": "Specialty Chemicals",
}

# Nifty 50 symbols for quick reference
NIFTY_50_SYMBOLS = list(NSE_SECTOR_MAP.keys())[:50]


def _to_yf_symbol(symbol: str) -> str:
    """Convert NSE symbol to Yahoo Finance format (append .NS)."""
    symbol = symbol.upper().strip()
    if not symbol.endswith(".NS"):
        return f"{symbol}.NS"
    return symbol


async def get_equity_details(symbol: str) -> dict[str, Any]:
    """Fetch current equity details for an NSE symbol."""
    yf_symbol = _to_yf_symbol(symbol)
    try:
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info
        hist = ticker.history(period="5d")

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if current_price is None and not hist.empty:
            current_price = float(hist["Close"].iloc[-1])

        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        change_pct = None
        if current_price and prev_close and prev_close > 0:
            change_pct = round(((current_price - prev_close) / prev_close) * 100, 2)

        return {
            "symbol": symbol.upper(),
            "company_name": info.get("longName", symbol),
            "current_price": current_price,
            "previous_close": prev_close,
            "change_pct": change_pct,
            "volume": info.get("volume") or info.get("regularMarketVolume"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "week_52_high": info.get("fiftyTwoWeekHigh"),
            "week_52_low": info.get("fiftyTwoWeekLow"),
            "dividend_yield": info.get("dividendYield"),
            "roe": info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
            "sector": info.get("sector") or NSE_SECTOR_MAP.get(symbol.upper(), "Unknown"),
            "industry": info.get("industry"),
            "data_source": "yfinance",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"get_equity_details failed for {symbol}: {e}")
        raise


async def get_equity_historical_data(
    symbol: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """Fetch OHLCV historical data for an NSE symbol."""
    yf_symbol = _to_yf_symbol(symbol)
    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            return {"symbol": symbol, "error": "No data available for the specified period", "data": []}

        df = df.reset_index()
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        records = df[["Date", "Open", "High", "Low", "Close", "Volume"]].round(2).to_dict("records")

        return {
            "symbol": symbol.upper(),
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "data": records,
            "data_source": "yfinance",
        }
    except Exception as e:
        logger.error(f"get_equity_historical_data failed for {symbol}: {e}")
        raise


async def get_all_stock_symbols() -> dict[str, Any]:
    """Return known NSE symbols (Nifty 500 subset for fast startup)."""
    # Extended Nifty 500 list — top 100 by market cap
    symbols = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
        "BHARTIARTL", "ITC", "BAJFINANCE", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT",
        "MARUTI", "TITAN", "SUNPHARMA", "WIPRO", "ULTRACEMCO", "NESTLEIND",
        "POWERGRID", "NTPC", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "HCLTECH",
        "TECHM", "ADANIPORTS", "COALINDIA", "ONGC", "DRREDDY", "CIPLA",
        "DIVISLAB", "BAJAJFINSV", "EICHERMOT", "HEROMOTOCO", "BRITANNIA", "SHREECEM",
        "GRASIM", "M&M", "HINDALCO", "VEDL", "BPCL", "IOC", "TATACONSUM",
        "APOLLOHOSP", "PIDILITIND", "DMART", "BAJAJ-AUTO", "ADANIENT",
        "TRENT", "ZOMATO", "PAYTM", "NYKAA", "IRCTC", "HAL", "BEL",
        "BHEL", "NHPC", "RECLTD", "PFC", "SAIL", "NMDC", "MOIL",
        "MCDOWELL-N", "UNITDSPR", "UBL", "INDIGO", "SPICEJET", "GMRINFRA",
        "IRFC", "RVNL", "TIINDIA", "MOTHERSON", "BALKRISIND", "CEAT",
        "MRF", "AMBUJACEM", "ACC", "DALBHARAT", "RAMCOCEM",
        "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "FEDERALBNK", "IDFCFIRSTB",
        "RBLBANK", "DCBBANK", "KARURVYSYA", "KTKBANK", "LAKSHVILAS",
        "HDFCLIFE", "SBILIFE", "ICICIPRULI", "LICI", "NIACL",
        "MANAPPURAM", "MUTHOOTFIN", "CHOLAFIN", "SHRIRAMFIN", "LICHSGFIN",
    ]
    return {
        "total_symbols": len(symbols),
        "symbols": symbols,
        "note": "Top 100 NSE equities by market cap. For full Nifty 500, set universe=[] in factor_screen.",
    }


async def get_indices() -> dict[str, Any]:
    """Fetch current NSE index values."""
    indices = {
        "^NSEI": "NIFTY 50",
        "^NSEBANK": "NIFTY BANK",
        "^CNXIT": "NIFTY IT",
        "^NSMIDCP": "NIFTY MIDCAP 100",
        "^NSEMDCP50": "NIFTY MIDCAP 50",
    }

    results = {}
    for yf_sym, name in indices.items():
        try:
            ticker = yf.Ticker(yf_sym)
            hist = ticker.history(period="2d")
            if not hist.empty:
                current = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current
                change_pct = round(((current - prev) / prev) * 100, 2)
                results[name] = {
                    "value": round(current, 2),
                    "change_pct": change_pct,
                }
        except Exception as e:
            results[name] = {"error": str(e)}

    return {
        "indices": results,
        "timestamp": datetime.now().isoformat(),
        "data_source": "yfinance",
    }
