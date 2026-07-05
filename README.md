# Agentic Stock Portfolio Analyst & Alert System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade agentic AI system for NSE India portfolio analysis. Uses deterministic, backtestable metrics — not LLM opinions — as the source of truth.

> **"Backtested factor-screening strategy vs Nifty 50 benchmark with verifiable grounding accuracy across all numeric financial claims."**

---

## What Makes This Different

Most agentic AI demos rely on "the agent sounded smart." This system doesn't:
- Every price, ratio, and signal comes from a **deterministic tool call** with a traceable `tool_call_id`
- A **grounding verifier** hard-blocks delivery if any numeric claim can't be verified against raw tool output
- **Backtested Sharpe ratio** and **alert hit-rate** are real numbers from real historical data — not projected

---

## Architecture

```
fetch_market_data
      ↓
compute_technical_indicators ──→ compute_portfolio_metrics
      ↓                                    ↓
factor_screen_opportunities        generate_candidate_alerts
      ↓                                    ↓
      └──────────────→ guardrail_validate ←┘
                            ↓
              [conditional: severity check]
                 ↓                    ↓
        human_approval_node   generate_report
                 ↓                    ↓
                 └───→ grounding_verifier
                              ↓
                       send_whatsapp
                              ↓
                       log_metrics_to_langsmith
```

## Tech Stack

| Layer | Technology |
|---|---|
| Data | `stock-nse-india` (NSE API wrapper, MCP server) |
| Orchestration | LangGraph (Python) — state machine with interrupts |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Indicators | `pandas` + `pandas-ta` — deterministic, never LLM-estimated |
| Observability | LangSmith — custom evaluators, per-node traces |
| Delivery | WhatsApp (Twilio/Meta Business Cloud API) |
| State Persistence | Postgres checkpointing / MemorySaver (dev) |

---

## Key Metrics (Backtestable)

| Metric | Description |
|---|---|
| **Sharpe Ratio** | Factor-screen basket vs Nifty 50 over backtest period |
| **Grounding Accuracy** | % numeric claims verifiable against tool outputs |
| **Alert Hit-Rate** | % alerts where price moved in predicted direction within N days |
| **Guardrail Catch Rate** | `blocked_alerts / (validated + blocked)` |
| **Per-node Latency** | LangSmith trace breakdown |

---

## Guardrails

1. **Symbol Grounding** — `validate_symbol` called before any analysis
2. **Numeric Citation** — every number must map to a `tool_call_id`
3. **Human-in-the-Loop** — `severity: high` alerts require explicit approval
4. **Rate Limiting** — per-symbol cooldown (6h) in Postgres state
5. **Disclaimer Injection** — SEBI regulatory disclaimer on every report
6. **WhatsApp Template Compliance** — 24-hour session window enforcement
7. **Risk Ceiling** — sector/single-stock concentration limits

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Pushkar3112/stock_market-analyzer.git
cd stock_market-analyzer

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install MCP server dependencies
cd mcp_server && npm install && npm run build && cd ..

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Verify Groq API key
python scripts/verify_groq_api.py

# 6. Test MCP server
python scripts/test_mcp_server.py

# 7. Run the agent
python -m agent.graph
```

---

## Setup

See [docs/SETUP.md](docs/SETUP.md) for detailed installation and configuration instructions.

---

## Regulatory Disclaimer

> This is an automated analytical output, not personalized investment advice under SEBI Research Analyst Regulations. Past performance and backtested signals do not guarantee future results.
