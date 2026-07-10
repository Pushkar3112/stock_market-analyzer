# Stock Market Analyzer — Agentic AI Portfolio System

A production-grade agentic AI system for NSE India stock portfolio analysis and alerting.

## Tech Stack
- **Data Layer**: `stock-nse-india` + `yfinance` via Python MCP server (9 tools)
- **Orchestration**: LangGraph — 10-node state machine with conditional edges + human-in-the-loop
- **LLM**: Groq (Llama 3.3 70B) via LangChain — grounded, not hallucinated
- **Observability**: LangSmith — custom evaluators, per-node cost/latency tracing
- **UI**: Streamlit — 9-page dashboard covering all system features
- **Backtesting**: Factor strategy vs Nifty 50 with deterministic Sharpe/hit-rate metrics

## Features
- ✅ Real-time NSE equity data + 7 technical indicators (RSI, MACD, Bollinger, SMA/EMA)
- ✅ Portfolio metrics: Sharpe ratio, VaR(95%), max drawdown, sector concentration
- ✅ Multi-factor screening: momentum, quality, value, low-volatility
- ✅ 7 guardrails: symbol validation, concentration limits, human-in-the-loop interrupt
- ✅ Grounding verifier: every LLM numeric claim traced to a tool_call_id
- ✅ WhatsApp delivery (Twilio) with retry logic
- ✅ Alert hit-rate backtesting + factor vs benchmark backtesting

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY
streamlit run app.py
```

## Architecture
```
fetch_market_data
    → compute_technical_indicators
    → compute_portfolio_metrics
    → [factor_screen_opportunities ‖ generate_candidate_alerts]
    → guardrail_validate → [human_approval] → generate_report
    → grounding_verifier → send_whatsapp → log_metrics_to_langsmith
```

## Backtested Metrics (deterministic)
- Alert hit-rate: signals verified against subsequent price moves
- Factor strategy: Sharpe ratio, CAGR vs Nifty 50 benchmark

---
*Built with LangGraph + LangSmith + NSE India data. Not investment advice.*
