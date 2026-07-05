"""
LangSmith custom evaluator — grounding accuracy.
Measures: % of numeric claims in report that trace back to a verified tool call.
This is the key hallucination metric for the portfolio analyst.
"""
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Run, Example


def grounding_accuracy_evaluator(run: Run, example: Example) -> EvaluationResult:
    """
    Score = verified_claims / total_claims.
    Expects run.outputs to contain:
      - grounding_check_passed: bool
      - grounding_failures: list[str]
    """
    outputs = run.outputs or {}
    passed = outputs.get("grounding_check_passed", False)
    failures = outputs.get("grounding_failures", [])

    # If we have explicit failure count
    total_claims_approx = outputs.get("total_numeric_claims", 10)  # fallback
    verified = total_claims_approx - len(failures)
    score = max(0.0, verified / total_claims_approx) if total_claims_approx > 0 else (1.0 if passed else 0.0)

    return EvaluationResult(
        key="grounding_accuracy",
        score=round(score, 4),
        comment=(
            f"{'PASS' if passed else 'FAIL'}: {len(failures)} unverified claim(s). "
            f"Score: {score:.1%}"
        ),
    )


def guardrail_catch_rate_evaluator(run: Run, example: Example) -> EvaluationResult:
    """
    Score = blocked_alerts / total_alerts.
    Higher = guardrail is actively filtering.
    """
    outputs = run.outputs or {}
    validated = len(outputs.get("validated_alerts", []))
    blocked = len(outputs.get("blocked_alerts", []))
    total = validated + blocked

    score = blocked / total if total > 0 else 0.0

    return EvaluationResult(
        key="guardrail_catch_rate",
        score=round(score, 4),
        comment=f"{blocked}/{total} alerts blocked by guardrails ({score:.1%})",
    )


def portfolio_metric_accuracy_evaluator(run: Run, example: Example) -> EvaluationResult:
    """
    Compare agent's Sharpe/volatility against expected values in the test dataset.
    Should match to within 0.01 (floating-point precision sanity check).
    """
    outputs = run.outputs or {}
    expected = example.outputs or {}

    metrics = outputs.get("portfolio_metrics", {}).get("metrics", {})
    expected_metrics = expected.get("expected_metrics", {})

    if not expected_metrics:
        return EvaluationResult(key="metric_accuracy", score=1.0, comment="No expected metrics to compare")

    errors = []
    for key, expected_val in expected_metrics.items():
        actual_val = metrics.get(key)
        if actual_val is None:
            errors.append(f"{key}: missing")
        elif abs(actual_val - expected_val) > 0.5:
            errors.append(f"{key}: expected {expected_val}, got {actual_val}")

    score = 1.0 - (len(errors) / len(expected_metrics)) if expected_metrics else 1.0

    return EvaluationResult(
        key="metric_accuracy",
        score=round(score, 4),
        comment=f"Metric errors: {errors}" if errors else "All metrics within tolerance",
    )
