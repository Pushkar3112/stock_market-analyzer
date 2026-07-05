# Guardrails — Safety Architecture

## Overview

The guardrail layer is what separates this system from a "toy" agentic demo.
Every alert must pass **5 sequential checks** before delivery. Failures are logged with specific reasons.

## Guardrail 5.1 — Symbol Grounding

**Node**: `fetch_market_data`

`validate_symbol()` is called for every ticker **before** any analysis. If it fails:
- The symbol is rejected and logged in `blocked_alerts`
- No equity data, technical indicators, or alerts are generated for it

This prevents hallucinated tickers (e.g., "FAKETATA") from propagating into the system.

## Guardrail 5.2 — Numeric Citation Enforcement

**Node**: `grounding_verifier`

Every numeric claim in the final report must map to a `tool_call_id` in `tool_call_registry`.
The verifier:
1. Uses LLM to extract all numeric claims from the report
2. Cross-checks each value against tool outputs stored in state
3. Returns `grounding_check_passed=False` if **any** claim is unverified
4. Blocks delivery — this is a hard gate, not a warning

**Grounding accuracy** = `verified_claims / total_claims` — your headline hallucination metric.

## Guardrail 5.3 — Human-in-the-Loop (High Severity)

**Node**: `human_approval_node` (LangGraph interrupt)

Any alert classified `severity: high` routes through an interrupt node requiring explicit approval.
High severity triggers:
- VaR(95%) > 5% daily loss threshold
- Sector concentration > 40%
- Single stock > 25% portfolio weight
- Price move > 8% single day

In production: uses `langgraph.types.interrupt()` to suspend execution and wait for external signal.
In dev: auto-approves with a warning log.

## Guardrail 5.4 — Alert Rate Limiting

**Node**: `guardrail_validate`

Per-symbol cooldown of 6 hours (configurable via `ALERT_COOLDOWN_HOURS`).
Tracked in `delivery_status.cooldown_registry` in Postgres state.
Prevents alert fatigue from repeated signals on the same ticker.

## Guardrail 5.5 — Regulatory Disclaimer

**Node**: `generate_report`

The SEBI disclaimer is appended by code **after** LLM generation — never left to the LLM's discretion:

> "This is an automated analytical output, not personalized investment advice under SEBI Research Analyst Regulations. Past performance and backtested signals do not guarantee future results."

The `_truncate_safely()` function in `send_whatsapp.py` ensures truncation for WhatsApp character limits **never removes the disclaimer**.

## Guardrail 5.6 — WhatsApp Template Compliance

**Node**: `send_whatsapp`

Meta's WhatsApp Business API prohibits freeform messages outside a 24-hour customer-initiated session window. The node:
1. Checks `last_user_message_at` timestamp in state
2. If outside the 24h window → forces pre-approved template message
3. If inside → allows freeform report

Most student projects miss this entirely — it's a real operational constraint that gets accounts banned.

## Guardrail 5.7 — Risk Ceiling for Opportunities

**Node**: `guardrail_validate`

Any `new_opportunity` alert that would push sector concentration past `SECTOR_CONCENTRATION_LIMIT` (default 40%) is blocked with an explicit reason, regardless of how good the factor score is.

## Metrics Produced by Guardrails

| Metric | Formula | Source |
|---|---|---|
| Guardrail catch rate | `blocked / (validated + blocked)` | `guardrail_validate` node |
| Grounding accuracy | `verified_claims / total_claims` | `grounding_verifier` node |
| Hallucinated tickers blocked | count from `blocked_alerts` | `fetch_market_data` + guardrail |
