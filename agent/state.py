"""
PortfolioState — the single source of truth for all agent nodes.
Every number that flows through the graph must be traceable to a tool_call_id.
"""
from typing import TypedDict, Literal, Optional, Annotated, Any
import operator


def _merge_dicts(a: dict, b: dict) -> dict:
    """Reducer for dict keys written by concurrent nodes — merges without losing entries."""
    return {**a, **b}


def _concat_lists(a: list, b: list) -> list:
    """Reducer for list keys written by concurrent nodes — concatenates."""
    return a + b


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

    # Screening — written by factor_screen_opportunities (concurrent with generate_candidate_alerts)
    screened_opportunities: Annotated[list[dict], _concat_lists]

    # Alert pipeline — candidate_alerts written by generate_candidate_alerts (fan-in node)
    candidate_alerts: Annotated[list[Alert], _concat_lists]
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
    # tool_call_registry is written by EVERY node — needs dict-merge reducer for concurrent steps
    tool_call_registry: Annotated[dict[str, Any], _merge_dicts]
    errors: Annotated[list[str], operator.add]   # accumulated error log
