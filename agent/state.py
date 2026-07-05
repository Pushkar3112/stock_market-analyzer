"""
PortfolioState — the single source of truth for all agent nodes.
Every number that flows through the graph must be traceable to a tool_call_id.
"""
from typing import TypedDict, Literal, Optional, Annotated
import operator


class Holding(TypedDict):
    symbol: str
    quantity: float
    avg_price: float


class Alert(TypedDict):
    symbol: str
    alert_type: Literal["price_move", "technical_signal", "portfolio_risk", "new_opportunity"]
    severity: Literal["low", "medium", "high"]
    grounded_facts: dict           # every number here MUST map to a tool_call_id
    rationale: str
    source_tool_calls: list[str]   # list of tool_call_ids backing each claim


class PortfolioState(TypedDict):
    # Input
    portfolio: list[Holding]
    trigger_type: Literal["scheduled_scan", "portfolio_update", "manual_query"]

    # Market data layer
    raw_market_data: dict          # symbol -> tool output (cached, keyed by tool_call_id)
    technical_indicators: dict     # symbol -> {rsi, macd, sma_50, sma_200, ...}
    portfolio_metrics: dict        # sharpe, volatility, max_drawdown, var_95, sector_exposure

    # Screening
    screened_opportunities: list[dict]   # factor-screened candidates, not LLM opinion

    # Alert pipeline
    candidate_alerts: list[Alert]
    validated_alerts: list[Alert]
    blocked_alerts: list[dict]          # alert + reason for guardrail rejection

    # Control flow
    requires_human_approval: bool
    human_approved: Optional[bool]

    # Report generation
    report_draft: str
    grounding_check_passed: bool
    grounding_failures: list[str]       # list of unverified numeric claims

    # Delivery
    delivery_status: dict               # {sent: int, failed: int, retried: int, messages: list}

    # Observability
    tool_call_registry: dict            # tool_call_id -> {tool, args, output, timestamp}
    errors: Annotated[list[str], operator.add]   # accumulated error log
