# Metrics — What Gets Measured and Why

## The Philosophy

Every metric in this system is **deterministic and backtestable** — not vibes-based.
The grounding verifier ensures no LLM-invented numbers reach the report.

---

## 1. Backtested Sharpe Ratio (Headline Metric)

**Formula**: `(Annualized Return - Risk-Free Rate) / Annualized Volatility`

**Risk-free rate used**: 6.5% (approximate RBI repo rate / 91-day T-bill)

**How computed**:
1. Run `factor_screen` (momentum + quality + low_volatility) at each rebalance date
2. Build equal-weight portfolio from top 15 candidates
3. Rebalance quarterly
4. Compute portfolio returns vs Nifty 50 (^NSEI) benchmark

**Script**: `python -m backtest.factor_backtest`

**Output**: `backtest/factor_strategy_vs_nifty50.png` + `backtest/results.json`

---

## 2. Alert Hit-Rate

**Formula**: `correct_direction_trades / total_trades × 100`

**How computed**:
For each signal rule (RSI oversold, MACD crossover, golden cross, etc.):
1. Replay signal rule against historical OHLCV
2. Record entry price at signal date
3. Record exit price N days later (configurable `hold_days`)
4. Count as "correct" if price moved in predicted direction

**Script**: `python -m backtest.alert_backtest`

**Output**: `backtest/alert_backtest_results.json`

---

## 3. Grounding Accuracy

**Formula**: `verified_numeric_claims / total_numeric_claims × 100`

**How computed**:
1. `grounding_verifier` node uses LLM to extract all numeric claims from report
2. Each claim is cross-checked against `tool_call_registry` in state
3. A claim is "verified" if its value matches (within 0.05 tolerance) a value in any tool output

**LangSmith evaluator**: `evals/grounding_evaluator.py::grounding_accuracy_evaluator`

**Target**: > 95% — anything below is a hallucination problem

---

## 4. Guardrail Catch Rate

**Formula**: `blocked_alerts / (validated_alerts + blocked_alerts) × 100`

**How computed**: Logged in `log_metrics_to_langsmith` node every run.

**What it proves**: The guardrail layer is actively doing something — not decorative.

---

## 5. Portfolio Metric Accuracy

**Validation**: Agent's Sharpe/volatility output compared against independent numpy/pandas computation.
Should match to floating-point precision (< 0.01 difference).

**LangSmith evaluator**: `evals/grounding_evaluator.py::portfolio_metric_accuracy_evaluator`

---

## 6. Per-Node Latency & Cost

Automatically traced by LangSmith when `LANGCHAIN_TRACING_V2=true`.
View at: https://smith.langchain.com → your project → Traces

Key nodes to monitor:
- `fetch_market_data` — network bound (yfinance)
- `compute_technical_indicators` — CPU bound (pandas-ta)
- `factor_screen_opportunities` — network + LLM
- `grounding_verifier` — LLM call

---

## Resume-Ready Numbers (fill in after running backtests)

| Metric | Value | Notes |
|---|---|---|
| Factor Strategy Sharpe | _run backtest_ | vs Nifty 50 benchmark |
| Nifty 50 Sharpe | _run backtest_ | baseline |
| Alert Hit-Rate (avg) | _run backtest_ | across 5 rule types |
| Grounding Accuracy | _run agent_ | % numerics verified |
| Guardrail Catch Rate | _run agent_ | % alerts blocked |
