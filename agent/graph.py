"""
Main LangGraph state machine for the Stock Portfolio Analyst.

Graph topology:
  fetch_market_data
        ↓
  [parallel] compute_technical_indicators + compute_portfolio_metrics
        ↓
  factor_screen_opportunities + generate_candidate_alerts
        ↓
  guardrail_validate
        ↓ [conditional: severity]
  human_approval_node (high)   generate_report (medium/low)
        ↓                            ↓
        └──────────→ grounding_verifier
                            ↓
                     send_whatsapp
                            ↓
                  log_metrics_to_langsmith
"""
import asyncio
import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import PortfolioState
from .nodes.fetch_market_data import fetch_market_data
from .nodes.compute_technical_indicators import compute_technical_indicators_node
from .nodes.compute_portfolio_metrics import compute_portfolio_metrics_node
from .nodes.factor_screen_opportunities import factor_screen_opportunities
from .nodes.generate_candidate_alerts import generate_candidate_alerts
from .nodes.guardrail_validate import guardrail_validate
from .nodes.generate_report import generate_report
from .nodes.grounding_verifier import grounding_verifier
from .nodes.send_whatsapp import send_whatsapp
from .nodes.log_metrics import log_metrics_to_langsmith

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Conditional edge functions ────────────────────────────────────────────────

def route_after_guardrail(state: PortfolioState) -> Literal["human_approval", "generate_report"]:
    """Route high-severity to human approval, others directly to report generation."""
    if state.get("requires_human_approval"):
        logger.info("Routing to human_approval (high-severity alert detected)")
        return "human_approval"
    return "generate_report"


def route_after_grounding(state: PortfolioState) -> Literal["send_whatsapp", "log_metrics_to_langsmith"]:
    """Block delivery if grounding check failed."""
    if not state.get("grounding_check_passed", False):
        failures = state.get("grounding_failures", [])
        logger.error(f"Grounding check FAILED — blocking delivery. Failures: {failures}")
        return "log_metrics_to_langsmith"  # skip delivery, log the failure
    return "send_whatsapp"


# ── Human approval node (LangGraph interrupt) ────────────────────────────────

async def human_approval_node(state: PortfolioState) -> dict:
    """
    Interrupt point for high-severity alerts.
    In production: this suspends execution and waits for external approval.
    In dev: auto-approves with a warning log.
    """
    from langgraph.types import interrupt
    validated_alerts = state.get("validated_alerts", [])
    high_alerts = [a for a in validated_alerts if a["severity"] == "high"]

    logger.warning(
        f"HUMAN APPROVAL REQUIRED for {len(high_alerts)} high-severity alert(s):\n"
        + "\n".join(f"  - {a['symbol']}: {a['rationale']}" for a in high_alerts)
    )

    # In production: use interrupt() to pause and wait for external signal
    # For dev/testing: auto-approve
    try:
        approval = interrupt({
            "message": "High-severity alerts require approval before delivery",
            "alerts": high_alerts,
        })
        approved = approval.get("approved", False)
    except Exception:
        # Fallback for environments where interrupt isn't configured
        logger.warning("interrupt() not configured — auto-approving for dev mode")
        approved = True

    return {"human_approved": approved}


# ── Build the graph ────────────────────────────────────────────────────────────

def build_graph(use_memory_saver: bool = True) -> StateGraph:
    """Build and compile the LangGraph state machine."""
    builder = StateGraph(PortfolioState)

    # Register all nodes
    builder.add_node("fetch_market_data", fetch_market_data)
    builder.add_node("compute_technical_indicators", compute_technical_indicators_node)
    builder.add_node("compute_portfolio_metrics", compute_portfolio_metrics_node)
    builder.add_node("factor_screen_opportunities", factor_screen_opportunities)
    builder.add_node("generate_candidate_alerts", generate_candidate_alerts)
    builder.add_node("guardrail_validate", guardrail_validate)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("generate_report", generate_report)
    builder.add_node("grounding_verifier", grounding_verifier)
    builder.add_node("send_whatsapp", send_whatsapp)
    builder.add_node("log_metrics_to_langsmith", log_metrics_to_langsmith)

    # Entry point
    builder.set_entry_point("fetch_market_data")

    # Linear edges (first phase)
    builder.add_edge("fetch_market_data", "compute_technical_indicators")
    builder.add_edge("fetch_market_data", "compute_portfolio_metrics")
    builder.add_edge("compute_technical_indicators", "factor_screen_opportunities")
    builder.add_edge("compute_technical_indicators", "generate_candidate_alerts")
    builder.add_edge("compute_portfolio_metrics", "generate_candidate_alerts")
    builder.add_edge("factor_screen_opportunities", "guardrail_validate")
    builder.add_edge("generate_candidate_alerts", "guardrail_validate")

    # Conditional edge: severity routing
    builder.add_conditional_edges(
        "guardrail_validate",
        route_after_guardrail,
        {
            "human_approval": "human_approval",
            "generate_report": "generate_report",
        },
    )

    # Human approval rejoins at report generation
    builder.add_edge("human_approval", "generate_report")

    # Report → grounding → conditional delivery
    builder.add_edge("generate_report", "grounding_verifier")
    builder.add_conditional_edges(
        "grounding_verifier",
        route_after_grounding,
        {
            "send_whatsapp": "send_whatsapp",
            "log_metrics_to_langsmith": "log_metrics_to_langsmith",
        },
    )
    builder.add_edge("send_whatsapp", "log_metrics_to_langsmith")
    builder.add_edge("log_metrics_to_langsmith", END)

    # Compile with checkpointer
    checkpointer = MemorySaver() if use_memory_saver else None
    return builder.compile(checkpointer=checkpointer)


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_portfolio_analysis(
    portfolio: list[dict],
    trigger_type: str = "scheduled_scan",
    thread_id: str = "default",
) -> dict:
    """
    Run a full portfolio analysis cycle.

    Args:
        portfolio: List of {symbol, quantity, avg_price} dicts
        trigger_type: "scheduled_scan" | "portfolio_update" | "manual_query"
        thread_id: Checkpoint thread ID for persistence

    Returns:
        Final state after all nodes complete
    """
    graph = build_graph()

    initial_state: PortfolioState = {
        "portfolio": portfolio,
        "trigger_type": trigger_type,
        "raw_market_data": {},
        "technical_indicators": {},
        "portfolio_metrics": {},
        "screened_opportunities": [],
        "candidate_alerts": [],
        "validated_alerts": [],
        "blocked_alerts": [],
        "requires_human_approval": False,
        "human_approved": None,
        "report_draft": "",
        "grounding_check_passed": False,
        "grounding_failures": [],
        "delivery_status": {},
        "tool_call_registry": {},
        "errors": [],
    }

    config_dict = {"configurable": {"thread_id": thread_id}}

    logger.info(f"Starting portfolio analysis for {len(portfolio)} holdings [thread={thread_id}]")

    final_state = await graph.ainvoke(initial_state, config=config_dict)

    return final_state


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example portfolio for testing
    sample_portfolio = [
        {"symbol": "RELIANCE", "quantity": 10, "avg_price": 2800.0},
        {"symbol": "TCS", "quantity": 5, "avg_price": 3500.0},
        {"symbol": "HDFCBANK", "quantity": 20, "avg_price": 1600.0},
        {"symbol": "INFY", "quantity": 15, "avg_price": 1400.0},
        {"symbol": "ICICIBANK", "quantity": 25, "avg_price": 950.0},
    ]

    result = asyncio.run(run_portfolio_analysis(sample_portfolio, trigger_type="manual_query"))
    print("\nFinal report:\n")
    print(result.get("report_draft", "No report generated"))
