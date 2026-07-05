# Architecture — Agentic Stock Portfolio Analyst

## System Overview

This is a **fully Python** agentic AI system for NSE India portfolio analysis.

```
                     ┌─────────────────────────────────────┐
                     │         Python MCP Server            │
                     │   (mcp_server/server.py via stdio)  │
                     │                                      │
                     │  Tools:                              │
                     │  • validate_symbol                   │
                     │  • get_equity_details                │
                     │  • get_equity_historical_data        │
                     │  • get_all_stock_symbols             │
                     │  • get_indices                       │
                     │  • compute_technical_indicators      │
                     │  • compute_portfolio_metrics         │
                     │  • factor_screen                     │
                     │  • backtest_alert_rule               │
                     └───────────────┬─────────────────────┘
                                     │ Tool Calls
                     ┌───────────────▼─────────────────────┐
                     │        LangGraph Agent               │
                     │       (agent/graph.py)               │
                     └─────────────────────────────────────┘
```

## LangGraph Node Graph

```
fetch_market_data                     ← validates all symbols first (guardrail 5.1)
    │
    ├──→ compute_technical_indicators  ← RSI, MACD, SMA50/200, BB (deterministic)
    │         │
    │         ├──→ factor_screen_opportunities  ← quant factors, LLM only explains
    │         └──→ generate_candidate_alerts    ← price moves + tech signals
    │
    └──→ compute_portfolio_metrics    ← Sharpe, VaR, max drawdown (pure math)
              │
              └──→ generate_candidate_alerts (portfolio risk alerts)
                        │
                   guardrail_validate  ← 5 safety checks
                        │
                   [severity == high?]
                    /          \
           human_approval   generate_report
                    \          /
                   grounding_verifier  ← hard gate: every number verified
                        │
                   [grounding passed?]
                    /          \
              send_whatsapp  log_metrics_to_langsmith (blocked)
                    │
              log_metrics_to_langsmith
```

## Data Flow

1. **Symbols validated** → `validate_symbol` called before any analysis
2. **Tool outputs cached** → `tool_call_registry[call_id]` stores every tool output
3. **Numbers traced** → every metric in alerts/report carries `source_tool_calls`
4. **Grounding verified** → `grounding_verifier` cross-checks report against registry
5. **Delivery gated** → `send_whatsapp` checks grounding + session window + human approval

## Python Stack

| Component | Library |
|---|---|
| LLM | `langchain-groq` (llama-3.3-70b-versatile) |
| Orchestration | `langgraph` |
| NSE Data | `yfinance` (.NS suffix) |
| Indicators | `pandas-ta` + manual fallback |
| Portfolio Math | `numpy` + `pandas` |
| State Persistence | `MemorySaver` (dev) / Postgres (prod) |
| WhatsApp | `twilio` |
| Observability | `langsmith` |
| Backtesting | `matplotlib` + `seaborn` |
