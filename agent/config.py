"""
Configuration and environment loading for the stock portfolio analyst.
All thresholds here are the single source of truth — agents read from config, never hardcode.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # LLM
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # LangSmith
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "stock-portfolio-analyst")

    # WhatsApp
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_WHATSAPP_FROM: str = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    WHATSAPP_TO: str = os.getenv("WHATSAPP_TO", "")
    META_ACCESS_TOKEN: str = os.getenv("META_ACCESS_TOKEN", "")
    META_PHONE_NUMBER_ID: str = os.getenv("META_PHONE_NUMBER_ID", "")

    # Database
    POSTGRES_CONNECTION_STRING: str = os.getenv("POSTGRES_CONNECTION_STRING", "")

    # Alert thresholds
    ALERT_COOLDOWN_HOURS: int = int(os.getenv("ALERT_COOLDOWN_HOURS", "6"))
    VAR_THRESHOLD_PCT: float = float(os.getenv("VAR_THRESHOLD_PCT", "5.0"))
    SECTOR_CONCENTRATION_LIMIT: float = float(os.getenv("SECTOR_CONCENTRATION_LIMIT", "40.0"))
    SINGLE_STOCK_LIMIT: float = float(os.getenv("SINGLE_STOCK_LIMIT", "25.0"))
    HIGH_SEVERITY_PRICE_MOVE_PCT: float = float(os.getenv("HIGH_SEVERITY_PRICE_MOVE_PCT", "8.0"))

    # Backtest
    BACKTEST_START_DATE: str = os.getenv("BACKTEST_START_DATE", "2023-01-01")
    BACKTEST_END_DATE: str = os.getenv("BACKTEST_END_DATE", "2024-12-31")
    BENCHMARK_SYMBOL: str = os.getenv("BENCHMARK_SYMBOL", "^NSEI")

    # Regulatory disclaimer — immutable, always appended
    REGULATORY_DISCLAIMER: str = (
        "\n\n---\n"
        "DISCLAIMER: This is an automated analytical output, not personalized investment advice "
        "under SEBI Research Analyst Regulations. Past performance and backtested signals do not "
        "guarantee future results."
    )


config = Config()
