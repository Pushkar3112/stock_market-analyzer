"""
log_metrics node — logs all run metrics to LangSmith for observability.
Records grounding accuracy, guardrail catch rate, delivery status, per-run KPIs.
"""
import logging
from datetime import datetime

try:
    from langsmith import Client as LangSmithClient
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

from ..state import PortfolioState
from ..config import config

logger = logging.getLogger(__name__)


async def log_metrics_to_langsmith(state: PortfolioState) -> dict:
    """
    Compute and log all run metrics to LangSmith.
    Also prints a summary to stdout for local dev.
    """
    validated_alerts = state.get("validated_alerts", [])
    blocked_alerts = state.get("blocked_alerts", [])
    portfolio_metrics = state.get("portfolio_metrics", {})
    delivery_status = state.get("delivery_status", {})
    grounding_passed = state.get("grounding_check_passed", False)
    grounding_failures = state.get("grounding_failures", [])
    errors = state.get("errors", [])

    # --- Compute metrics ---
    total_alerts = len(validated_alerts) + len(blocked_alerts)
    guardrail_catch_rate = len(blocked_alerts) / total_alerts if total_alerts > 0 else 0.0
    grounding_failures_count = len(grounding_failures)

    pm = portfolio_metrics.get("metrics", {})
    run_summary = {
        "run_timestamp": datetime.now().isoformat(),
        "portfolio_metrics": {
            "sharpe_ratio": pm.get("sharpe_ratio"),
            "annualized_return_pct": pm.get("annualized_return_pct"),
            "annualized_volatility_pct": pm.get("annualized_volatility_pct"),
            "max_drawdown_pct": pm.get("max_drawdown_pct"),
            "var_95_pct": pm.get("var_95_pct"),
        },
        "alert_pipeline": {
            "candidate_alerts": len(state.get("candidate_alerts", [])),
            "validated_alerts": len(validated_alerts),
            "blocked_alerts": len(blocked_alerts),
            "guardrail_catch_rate": round(guardrail_catch_rate, 4),
        },
        "grounding": {
            "check_passed": grounding_passed,
            "unverified_claims": grounding_failures_count,
        },
        "delivery": delivery_status,
        "errors": errors[:10],  # cap for readability
    }

    # --- Print summary ---
    print("\n" + "="*60)
    print("STOCK PORTFOLIO ANALYST — RUN SUMMARY")
    print("="*60)
    print(f"Timestamp      : {run_summary['run_timestamp']}")
    print(f"Sharpe Ratio   : {pm.get('sharpe_ratio', 'N/A')}")
    print(f"Volatility     : {pm.get('annualized_volatility_pct', 'N/A')}%")
    print(f"Max Drawdown   : {pm.get('max_drawdown_pct', 'N/A')}%")
    print(f"VaR (95%)      : {pm.get('var_95_pct', 'N/A')}%")
    print(f"Validated Alerts: {len(validated_alerts)}")
    print(f"Blocked Alerts : {len(blocked_alerts)} (catch rate: {guardrail_catch_rate:.1%})")
    print(f"Grounding Pass : {'YES' if grounding_passed else 'NO'} ({grounding_failures_count} failures)")
    print(f"Delivery       : {delivery_status.get('status', 'N/A')}")
    if errors:
        print(f"Errors         : {len(errors)}")
    print("="*60 + "\n")

    # --- Log to LangSmith ---
    if LANGSMITH_AVAILABLE and config.LANGCHAIN_API_KEY and config.LANGCHAIN_TRACING_V2:
        try:
            ls_client = LangSmithClient(api_key=config.LANGCHAIN_API_KEY)
            ls_client.create_run(
                name="portfolio_analyst_run",
                run_type="chain",
                inputs={"portfolio_size": len(state.get("portfolio", []))},
                outputs=run_summary,
                project_name=config.LANGCHAIN_PROJECT,
            )
            logger.info("Metrics logged to LangSmith")
        except Exception as e:
            logger.warning(f"LangSmith logging failed (non-critical): {e}")
    else:
        logger.info("LangSmith tracing disabled or not configured — metrics printed to stdout only")

    return {}  # no state update needed
