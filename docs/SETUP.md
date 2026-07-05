# Setup Guide

## Prerequisites

- Python 3.11+
- Git

## Installation

```bash
# 1. Clone
git clone https://github.com/Pushkar3112/stock_market-analyzer.git
cd stock_market-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Your Groq API key is already pre-filled in .env

# 4. Verify Groq API key
python scripts/verify_groq_api.py
```

## Running the Agent

```bash
# Full portfolio analysis (uses sample portfolio)
python -m agent.graph

# Factor backtest vs Nifty 50
python -m backtest.factor_backtest

# MCP Server standalone (for testing tools)
python -m mcp_server.server
```

## Running the MCP Server

The MCP server runs over stdio:
```bash
python -m mcp_server.server
```

To test individual tools:
```bash
python scripts/test_mcp_server.py
```

## Optional: LangSmith Tracing

1. Get a free LangSmith API key at https://smith.langchain.com
2. Set in .env:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=ls_your_key
   ```
3. Run agent — traces appear automatically in your LangSmith dashboard

## Optional: WhatsApp Delivery

Using Twilio WhatsApp sandbox (free for testing):
1. Sign up at https://www.twilio.com
2. Enable WhatsApp sandbox
3. Set in .env:
   ```
   TWILIO_ACCOUNT_SID=your_sid
   TWILIO_AUTH_TOKEN=your_token
   WHATSAPP_TO=whatsapp:+91XXXXXXXXXX
   ```

Without Twilio configured, the agent runs in mock mode (report printed to stdout).

## NSE Data Source

Data is fetched via `yfinance` using the `.NS` suffix for NSE stocks:
- `RELIANCE` → `RELIANCE.NS` on Yahoo Finance
- `^NSEI` → Nifty 50 index

No API key required for market data. Rate limiting applies for large universe scans.
