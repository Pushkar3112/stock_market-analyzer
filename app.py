"""
Agentic Stock Portfolio Analyst — Streamlit UI
Covers all project features: live data, technical indicators, portfolio metrics,
factor screening, backtesting, guardrails demo, and full agent run.

Run: streamlit run app.py
"""

import streamlit as st
import sys
import os

# ── Page config MUST be first ──────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Stock Portfolio Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, os.path.dirname(__file__))

# ── Imports ────────────────────────────────────────────────────────────────────
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Custom CSS — dark premium theme ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a0a1a 100%);
    color: #e0e0e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1a1a2e 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #c8d6e5 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 20px rgba(0,212,255,0.08);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1b2a;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #7a8fa6;
    font-weight: 500;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00d4ff22, #0066ff22) !important;
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0066ff);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,212,255,0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(0,212,255,0.5);
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background: #1a1a2e !important;
    border: 1px solid #1e3a5f !important;
    color: #e0e0e0 !important;
    border-radius: 8px !important;
}

/* Success/Warning/Error boxes */
.stSuccess { background: rgba(0,212,100,0.1) !important; border: 1px solid #00d464 !important; border-radius: 8px !important; }
.stWarning { background: rgba(255,165,0,0.1) !important; border: 1px solid #ffa500 !important; border-radius: 8px !important; }
.stError   { background: rgba(255,50,50,0.1)  !important; border: 1px solid #ff3232 !important; border-radius: 8px !important; }

/* DataFrames */
.stDataFrame { border: 1px solid #1e3a5f !important; border-radius: 10px !important; }

/* Expander */
.streamlit-expanderHeader {
    background: #1a1a2e !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #00d4ff !important;
}

/* Hero section */
.hero-card {
    background: linear-gradient(135deg, #0d1b2a, #16213e);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(0,212,255,0.1);
}

/* Badge */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
.badge-high   { background: rgba(255,50,50,0.2);  color: #ff6464; border: 1px solid #ff3232; }
.badge-medium { background: rgba(255,165,0,0.2);  color: #ffb732; border: 1px solid #ffa500; }
.badge-low    { background: rgba(0,212,100,0.2);  color: #00e47a; border: 1px solid #00d464; }
.badge-pass   { background: rgba(0,212,100,0.2);  color: #00e47a; border: 1px solid #00d464; }
.badge-fail   { background: rgba(255,50,50,0.2);  color: #ff6464; border: 1px solid #ff3232; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def run_async(coro):
    """Run an async coroutine from sync Streamlit context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=120)
        return loop.run_until_complete(coro)
    except Exception:
        return asyncio.run(coro)


def plotly_dark_layout(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#00d4ff", size=16)),
        paper_bgcolor="#0d1b2a",
        plot_bgcolor="#0d1b2a",
        font=dict(color="#c8d6e5", family="Inter"),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
        yaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
        legend=dict(bgcolor="#1a1a2e", bordercolor="#1e3a5f"),
    )
    return fig


@st.cache_data(ttl=300)
def cached_equity_details(symbol: str):
    from mcp_server.tools.market_data import get_equity_details
    return run_async(get_equity_details(symbol))


@st.cache_data(ttl=300)
def cached_historical_data(symbol: str, start: str, end: str):
    from mcp_server.tools.market_data import get_equity_historical_data
    return run_async(get_equity_historical_data(symbol, start, end))


@st.cache_data(ttl=300)
def cached_indicators(symbol: str):
    from mcp_server.tools.technical_indicators import compute_technical_indicators
    return run_async(compute_technical_indicators(symbol, ["ALL"], 365))


@st.cache_data(ttl=60)
def cached_validate(symbol: str):
    from mcp_server.tools.validate_symbol import validate_symbol
    return run_async(validate_symbol(symbol))


# ── Sidebar Navigation ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <h2 style='color:#00d4ff; margin:0; font-size:1.4rem;'>📈 NSE Analyst</h2>
        <p style='color:#7a8fa6; font-size:0.8rem; margin:0;'>Agentic Portfolio Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Dashboard",
            "🔍 Symbol Lookup",
            "📊 Technical Analysis",
            "💼 Portfolio Metrics",
            "🔎 Factor Screening",
            "🤖 Full Agent Run",
            "📉 Backtesting",
            "🛡️ Guardrails Demo",
            "⚙️ System Check",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("""
    <div style='font-size:0.75rem; color:#7a8fa6; padding: 0.5rem;'>
        <b style='color:#00d4ff;'>Stack</b><br>
        LangGraph · Groq LLM<br>
        yfinance · pandas-ta<br>
        LangSmith · Twilio WA
        <br><br>
        <b style='color:#00d4ff;'>Data</b><br>
        NSE India via yfinance
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Dashboard":
    st.markdown("""
    <div class='hero-card'>
        <h1 style='color:#00d4ff; margin:0 0 0.5rem;'>Agentic Stock Portfolio Analyst</h1>
        <p style='color:#7a8fa6; margin:0;'>
            Production-grade AI system for NSE India portfolio analysis.<br>
            Every metric is <b style='color:#00d4ff'>deterministic & backtestable</b> — not LLM opinion.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Nifty 50 live
    st.subheader("📡 Live Market Pulse")
    with st.spinner("Fetching live index data..."):
        try:
            from mcp_server.tools.market_data import get_indices
            indices = run_async(get_indices())
            idx_data = indices.get("indices", {})

            cols = st.columns(len(idx_data))
            for i, (name, val) in enumerate(idx_data.items()):
                v = val.get("value", "N/A")
                c = val.get("change_pct", 0)
                color = "#00d464" if c >= 0 else "#ff4444"
                arrow = "▲" if c >= 0 else "▼"
                cols[i].metric(
                    label=name,
                    value=f"{v:,.2f}" if isinstance(v, float) else v,
                    delta=f"{arrow} {abs(c):.2f}%" if isinstance(c, float) else "N/A",
                )
        except Exception as e:
            st.warning(f"Index fetch error: {e}")

    st.divider()

    # Architecture overview
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🗺️ System Architecture")
        st.code("""
fetch_market_data          ← validates all symbols (guardrail 5.1)
    │
    ├─→ compute_technical_indicators   ← RSI, MACD, BB (pandas-ta)
    │       ├─→ factor_screen_opportunities
    │       └─→ generate_candidate_alerts
    │
    └─→ compute_portfolio_metrics      ← Sharpe, VaR (pure math)
                └─→ generate_candidate_alerts (portfolio risk)
                        │
                   guardrail_validate  ← 5 safety checks
                        │
              ┌─────────┴─────────┐
         human_approval     generate_report
              └─────────┬─────────┘
                   grounding_verifier  ← hard gate
                        │
                   send_whatsapp → log_metrics_to_langsmith
        """, language="text")

    with col2:
        st.subheader("🛡️ Guardrails Active")
        guardrails = [
            ("5.1", "Symbol Validation", "✅"),
            ("5.2", "Numeric Grounding", "✅"),
            ("5.3", "Human-in-Loop", "✅"),
            ("5.4", "Rate Limiting", "✅"),
            ("5.5", "SEBI Disclaimer", "✅"),
            ("5.6", "WA Template Compliance", "✅"),
            ("5.7", "Risk Ceiling", "✅"),
        ]
        for num, name, status in guardrails:
            st.markdown(f"`G{num}` {status} **{name}**")

        st.divider()
        st.subheader("📦 Key Metrics")
        m1, m2 = st.columns(2)
        m1.metric("Agent Nodes", "10")
        m2.metric("MCP Tools", "9")
        m1.metric("Guardrails", "7")
        m2.metric("Python Only", "✅")

    st.divider()

    # Quick stock snapshot
    st.subheader("⚡ Quick Snapshot — Nifty Blue Chips")
    quick_symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
    quick_cols = st.columns(5)
    for i, sym in enumerate(quick_symbols):
        with quick_cols[i]:
            try:
                data = cached_equity_details(sym)
                price = data.get("current_price", "N/A")
                chg = data.get("change_pct", 0)
                color = "normal" if (chg or 0) >= 0 else "inverse"
                st.metric(
                    label=sym,
                    value=f"₹{price:,.2f}" if isinstance(price, float) else price,
                    delta=f"{chg:+.2f}%" if isinstance(chg, (int, float)) else "N/A",
                    delta_color=color,
                )
            except Exception:
                st.metric(label=sym, value="N/A")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SYMBOL LOOKUP
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Symbol Lookup":
    st.title("🔍 Symbol Lookup & Validation")
    st.caption("Validate any NSE symbol and fetch live equity details. Guardrail 5.1 in action.")

    col1, col2 = st.columns([1, 2])
    with col1:
        symbol_input = st.text_input("Enter NSE Symbol", value="RELIANCE", placeholder="e.g. TCS, INFY").upper().strip()
        validate_btn = st.button("Validate & Fetch", use_container_width=True)

    if validate_btn and symbol_input:
        with st.spinner(f"Validating {symbol_input}..."):
            validation = cached_validate(symbol_input)

        if validation["valid"]:
            st.success(f"✅ **{symbol_input}** is a valid NSE symbol")

            with st.spinner("Fetching equity details..."):
                data = cached_equity_details(symbol_input)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"₹{data.get('current_price', 'N/A'):,.2f}" if isinstance(data.get('current_price'), float) else "N/A")
            col2.metric("Change", f"{data.get('change_pct', 0):+.2f}%" if isinstance(data.get('change_pct'), float) else "N/A",
                        delta=f"{data.get('change_pct', 0):+.2f}%" if isinstance(data.get('change_pct'), float) else None,
                        delta_color="normal" if (data.get('change_pct') or 0) >= 0 else "inverse")
            col3.metric("52W High", f"₹{data.get('week_52_high', 'N/A'):,.2f}" if isinstance(data.get('week_52_high'), float) else "N/A")
            col4.metric("52W Low", f"₹{data.get('week_52_low', 'N/A'):,.2f}" if isinstance(data.get('week_52_low'), float) else "N/A")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("P/E Ratio", f"{data.get('pe_ratio', 'N/A'):.2f}" if isinstance(data.get('pe_ratio'), float) else "N/A")
            col2.metric("P/B Ratio", f"{data.get('pb_ratio', 'N/A'):.2f}" if isinstance(data.get('pb_ratio'), float) else "N/A")
            col3.metric("Market Cap", f"₹{data.get('market_cap', 0)/1e10:.2f}K Cr" if isinstance(data.get('market_cap'), (int, float)) else "N/A")
            col4.metric("Sector", data.get('sector', 'N/A'))

            with st.expander("📋 Full Details (Raw tool output)"):
                st.json(data)

            # Price chart
            st.subheader("📈 Price History (1 Year)")
            with st.spinner("Loading chart..."):
                end = datetime.now().strftime("%Y-%m-%d")
                start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                hist = cached_historical_data(symbol_input, start, end)
                records = hist.get("data", [])

                if records:
                    df = pd.DataFrame(records)
                    df["Date"] = pd.to_datetime(df["Date"])

                    fig = make_subplots(
                        rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03
                    )
                    fig.add_trace(go.Candlestick(
                        x=df["Date"], open=df["Open"], high=df["High"],
                        low=df["Low"], close=df["Close"],
                        increasing_line_color="#00d464", decreasing_line_color="#ff4444",
                        name="Price"
                    ), row=1, col=1)
                    fig.add_trace(go.Bar(
                        x=df["Date"], y=df["Volume"],
                        marker_color=["#00d46488" if c >= o else "#ff444488"
                                      for c, o in zip(df["Close"], df["Open"])],
                        name="Volume"
                    ), row=2, col=1)

                    plotly_dark_layout(fig, f"{symbol_input} — 1 Year OHLCV", height=500)
                    fig.update_layout(xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"❌ **{symbol_input}** is NOT a valid NSE symbol")
            st.info(f"Details: {validation['message']}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Technical Analysis":
    st.title("📊 Technical Analysis")
    st.caption("All indicators computed deterministically via pandas-ta — never LLM-estimated.")

    col1, col2 = st.columns([1, 3])
    with col1:
        ta_symbol = st.text_input("Symbol", value="TCS").upper().strip()
        run_ta = st.button("Compute Indicators", use_container_width=True)

    if run_ta and ta_symbol:
        # Validate first
        v = cached_validate(ta_symbol)
        if not v["valid"]:
            st.error(f"❌ Invalid symbol: {ta_symbol}. {v['message']}")
            st.stop()

        with st.spinner(f"Computing all indicators for {ta_symbol}..."):
            result = cached_indicators(ta_symbol)

        if "error" in result:
            st.error(f"Error: {result['error']}")
            st.stop()

        ind = result.get("indicators", {})
        current_price = result.get("current_price", 0)

        st.markdown(f"### {ta_symbol} — ₹{current_price:,.2f}")

        # Key metrics row
        c1, c2, c3, c4, c5 = st.columns(5)
        rsi_val = ind.get("RSI", {}).get("value")
        macd_hist = ind.get("MACD", {}).get("histogram")
        sma50 = ind.get("SMA_50", {}).get("value")
        sma200 = ind.get("SMA_200", {}).get("value")
        ema20 = ind.get("EMA_20", {}).get("value")

        c1.metric("RSI (14)", f"{rsi_val:.1f}" if rsi_val else "N/A",
                  delta="Overbought" if rsi_val and rsi_val > 70 else ("Oversold" if rsi_val and rsi_val < 30 else "Neutral"),
                  delta_color="inverse" if rsi_val and rsi_val > 70 else ("inverse" if rsi_val and rsi_val < 30 else "off"))
        c2.metric("MACD Hist", f"{macd_hist:.4f}" if macd_hist else "N/A",
                  delta_color="normal" if macd_hist and macd_hist > 0 else "inverse")
        c3.metric("SMA 50", f"₹{sma50:,.2f}" if sma50 else "N/A",
                  delta=f"{'Above' if current_price > (sma50 or 0) else 'Below'} SMA50",
                  delta_color="normal" if current_price > (sma50 or 0) else "inverse")
        c4.metric("SMA 200", f"₹{sma200:,.2f}" if sma200 else "N/A",
                  delta=f"{'Above' if current_price > (sma200 or 0) else 'Below'} SMA200",
                  delta_color="normal" if current_price > (sma200 or 0) else "inverse")
        c5.metric("EMA 20", f"₹{ema20:,.2f}" if ema20 else "N/A")

        st.divider()

        # RSI gauge chart
        col_l, col_r = st.columns(2)
        with col_l:
            if rsi_val:
                fig_rsi = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=rsi_val,
                    title={"text": "RSI (14)", "font": {"color": "#00d4ff"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#c8d6e5"},
                        "bar": {"color": "#00d4ff"},
                        "steps": [
                            {"range": [0, 30], "color": "#ff444433"},
                            {"range": [30, 70], "color": "#1e3a5f33"},
                            {"range": [70, 100], "color": "#ff444433"},
                        ],
                        "threshold": {"line": {"color": "#ffa500", "width": 3}, "value": rsi_val},
                    },
                    number={"font": {"color": "#e0e0e0"}}
                ))
                plotly_dark_layout(fig_rsi, height=280)
                st.plotly_chart(fig_rsi, use_container_width=True)
                st.caption(ind.get("RSI", {}).get("interpretation", ""))

        with col_r:
            # MACD bar chart
            if ind.get("MACD"):
                macd_d = ind["MACD"]
                fig_macd = go.Figure()
                categories = ["MACD Line", "Signal Line", "Histogram"]
                vals = [macd_d.get("macd_line", 0), macd_d.get("signal_line", 0), macd_d.get("histogram", 0)]
                colors = ["#00d4ff", "#ff9500", "#00d464" if (macd_d.get("histogram", 0) or 0) > 0 else "#ff4444"]
                fig_macd.add_trace(go.Bar(x=categories, y=vals, marker_color=colors))
                plotly_dark_layout(fig_macd, "MACD Components", height=280)
                st.plotly_chart(fig_macd, use_container_width=True)

        # Bollinger Bands chart
        st.subheader("📐 Bollinger Bands")
        if ind.get("Bollinger_Bands") and sma50:
            bb = ind["Bollinger_Bands"]
            labels = ["Lower Band", "Middle (SMA20)", "Current Price", "Upper Band"]
            vals = [bb.get("lower", 0), bb.get("middle", 0), current_price, bb.get("upper", 0)]
            colors_bb = ["#ff4444", "#ffa500", "#00d4ff", "#00d464"]

            fig_bb = go.Figure()
            fig_bb.add_trace(go.Bar(x=labels, y=vals, marker_color=colors_bb, text=[f"₹{v:,.2f}" for v in vals], textposition="auto"))
            plotly_dark_layout(fig_bb, "Bollinger Bands — Current Snapshot", height=300)
            st.plotly_chart(fig_bb, use_container_width=True)

            pb = bb.get("percent_b", 0.5)
            st.progress(min(max(pb, 0), 1), text=f"Percent B: {pb:.2%} — {bb.get('interpretation', '')}")

        # Full indicator JSON
        with st.expander("🔬 Raw Indicator Output (tool_call_id traceable)"):
            st.json(result)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PORTFOLIO METRICS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💼 Portfolio Metrics":
    st.title("💼 Portfolio Metrics")
    st.caption("Sharpe ratio, volatility, max drawdown, VaR(95%), sector concentration — pure math, fully reproducible.")

    st.subheader("📋 Enter Your Portfolio")

    # Pre-filled sample
    if "holdings" not in st.session_state:
        st.session_state.holdings = [
            {"symbol": "RELIANCE", "quantity": 10, "avg_price": 2800.0},
            {"symbol": "TCS", "quantity": 5, "avg_price": 3500.0},
            {"symbol": "HDFCBANK", "quantity": 20, "avg_price": 1600.0},
            {"symbol": "INFY", "quantity": 15, "avg_price": 1400.0},
            {"symbol": "ICICIBANK", "quantity": 25, "avg_price": 950.0},
        ]

    # Editable holdings table
    holdings_df = pd.DataFrame(st.session_state.holdings)
    edited = st.data_editor(
        holdings_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "symbol": st.column_config.TextColumn("Symbol", help="NSE symbol e.g. RELIANCE"),
            "quantity": st.column_config.NumberColumn("Quantity", min_value=1),
            "avg_price": st.column_config.NumberColumn("Avg Buy Price (₹)", min_value=0.01, format="₹%.2f"),
        },
        key="portfolio_editor",
    )

    if st.button("Compute Portfolio Metrics", use_container_width=False):
        holdings = edited.to_dict("records")
        holdings = [h for h in holdings if h.get("symbol") and h.get("quantity") and h.get("avg_price")]

        if not holdings:
            st.error("Please add at least one holding.")
            st.stop()

        with st.spinner("Computing portfolio metrics (fetching 1 year of price data)..."):
            from mcp_server.tools.portfolio_metrics import compute_portfolio_metrics
            result = run_async(compute_portfolio_metrics(holdings))

        if "error" in result:
            st.error(f"Error: {result['error']}")
            st.stop()

        metrics = result.get("metrics", {})
        holdings_data = result.get("holdings", [])
        sector_exp = result.get("sector_exposure", {})
        severity_flags = result.get("severity_flags", [])

        # Headline metrics
        st.subheader("📊 Risk Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        sharpe = metrics.get("sharpe_ratio", 0)
        c1.metric("Sharpe Ratio", f"{sharpe:.4f}",
                  delta="Good (>1)" if sharpe > 1 else "Low (<1)",
                  delta_color="normal" if sharpe > 1 else "inverse")
        c2.metric("Annual Return", f"{metrics.get('annualized_return_pct', 0):.2f}%")
        c3.metric("Volatility", f"{metrics.get('annualized_volatility_pct', 0):.2f}%")
        c4.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%",
                  delta_color="inverse")
        c5.metric("VaR (95%)", f"{metrics.get('var_95_pct', 0):.2f}%",
                  delta_color="inverse")

        c1.metric("Portfolio Value", f"₹{result.get('portfolio_value_inr', 0):,.2f}")

        # Severity flags
        if severity_flags:
            for flag in severity_flags:
                st.warning(f"⚠️ **HIGH SEVERITY**: {flag}")
        else:
            st.success("✅ No risk threshold breaches detected")

        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            # Sector pie
            if sector_exp:
                fig_sector = go.Figure(go.Pie(
                    labels=list(sector_exp.keys()),
                    values=list(sector_exp.values()),
                    hole=0.5,
                    marker_colors=px.colors.qualitative.Set2,
                    textinfo="label+percent",
                ))
                plotly_dark_layout(fig_sector, "Sector Exposure", height=360)
                st.plotly_chart(fig_sector, use_container_width=True)

        with col_r:
            # Holdings waterfall (P&L)
            if holdings_data:
                symbols = [h["symbol"] for h in holdings_data]
                pnl_pcts = [h.get("pnl_pct", 0) for h in holdings_data]
                colors_pnl = ["#00d464" if p >= 0 else "#ff4444" for p in pnl_pcts]

                fig_pnl = go.Figure(go.Bar(
                    x=symbols, y=pnl_pcts,
                    marker_color=colors_pnl,
                    text=[f"{p:+.1f}%" for p in pnl_pcts],
                    textposition="outside",
                ))
                plotly_dark_layout(fig_pnl, "P&L per Holding (%)", height=360)
                fig_pnl.add_hline(y=0, line_dash="dash", line_color="#7a8fa6")
                st.plotly_chart(fig_pnl, use_container_width=True)

        # Holdings table
        st.subheader("📋 Holdings Detail")
        df_display = pd.DataFrame(holdings_data)[[
            "symbol", "current_price", "avg_price", "quantity", "current_value", "pnl", "pnl_pct", "sector", "weight"
        ]] if holdings_data else pd.DataFrame()
        if not df_display.empty:
            df_display["weight"] = df_display["weight"].apply(lambda x: f"{x*100:.1f}%")
            df_display["pnl_pct"] = df_display["pnl_pct"].apply(lambda x: f"{x:+.2f}%")
            df_display["current_value"] = df_display["current_value"].apply(lambda x: f"₹{x:,.0f}")
            df_display["pnl"] = df_display["pnl"].apply(lambda x: f"₹{x:+,.0f}")
            st.dataframe(df_display, use_container_width=True)

        with st.expander("🔬 Raw tool output (for grounding verification)"):
            st.json(result)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — FACTOR SCREENING
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔎 Factor Screening":
    st.title("🔎 Quantitative Factor Screening")
    st.caption("Screen NSE stocks using momentum, quality, value & low-volatility factors. LLM explains WHY — never picks stocks.")

    col1, col2, col3 = st.columns(3)
    with col1:
        factors = st.multiselect(
            "Factors to Apply",
            ["momentum", "quality", "low_volatility", "value"],
            default=["momentum", "quality", "low_volatility"],
        )
    with col2:
        top_n = st.slider("Top N Candidates", 3, 20, 10)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_screen = st.button("Run Factor Screen", use_container_width=True)

    if run_screen:
        if not factors:
            st.error("Select at least one factor.")
            st.stop()

        with st.spinner(f"Screening NSE universe with {factors} factors (this takes ~30s)..."):
            from mcp_server.tools.factor_screen import factor_screen
            result = run_async(factor_screen(universe=[], factors=factors, top_n=top_n))

        candidates = result.get("candidates", [])

        if not candidates:
            st.warning("No candidates found with sufficient data.")
            st.stop()

        st.success(f"✅ Screened **{result.get('screened_universe_size', 0)}** symbols → **{len(candidates)}** candidates")
        st.caption(result.get("screening_note", ""))

        # Composite score bar chart
        symbols = [c["symbol"] for c in candidates]
        scores = [c["composite_score"] for c in candidates]
        sectors = [c.get("sector", "Unknown") for c in candidates]

        fig_scores = go.Figure(go.Bar(
            x=scores, y=symbols,
            orientation="h",
            marker=dict(
                color=scores,
                colorscale=[[0, "#1e3a5f"], [0.5, "#0066ff"], [1, "#00d4ff"]],
                showscale=True,
                colorbar=dict(title="Score", tickfont=dict(color="#c8d6e5")),
            ),
            text=[f"{s:.4f}" for s in scores],
            textposition="outside",
        ))
        plotly_dark_layout(fig_scores, "Factor Screen — Composite Scores (higher = stronger)", height=max(350, len(candidates) * 35))
        fig_scores.update_layout(yaxis={"autorange": "reversed"})
        st.plotly_chart(fig_scores, use_container_width=True)

        # Candidate cards
        st.subheader("📋 Top Candidates — Detailed Breakdown")
        for i, cand in enumerate(candidates[:10]):
            with st.expander(f"#{i+1} **{cand['symbol']}** — Score: {cand['composite_score']:.4f} | Sector: {cand.get('sector', 'N/A')} | ₹{cand.get('current_price', 0):,.2f}", expanded=i < 3):
                factor_scores = cand.get("factor_scores", {})
                fc_cols = st.columns(len(factor_scores))
                for j, (fname, fdata) in enumerate(factor_scores.items()):
                    fc_cols[j].metric(
                        label=fname.replace("_", " ").title(),
                        value=f"{fdata.get('score', 0):.4f}",
                    )
                    with fc_cols[j]:
                        if fname == "momentum":
                            st.caption(f"1M: {fdata.get('return_1m_pct', 0):+.1f}% | 3M: {fdata.get('return_3m_pct', 0):+.1f}% | 12M: {fdata.get('return_12m_pct', 0):+.1f}%")
                        elif fname == "low_volatility":
                            st.caption(f"Ann. Vol: {fdata.get('annualized_volatility_pct', 0):.1f}%")
                        elif fname == "quality":
                            gc = "Yes ✅" if fdata.get("golden_cross") else "No ❌"
                            st.caption(f"Above SMA50: {'Yes' if fdata.get('above_sma_50') else 'No'} | Golden Cross: {gc}")
                        elif fname == "value":
                            st.caption(f"From 52W High: -{fdata.get('pct_from_52w_high', 0):.1f}%")

                if cand.get("explanation"):
                    st.info(f"💡 **LLM Explanation**: {cand['explanation']}")
                else:
                    st.caption("_Explanation not available (LLM not invoked for non-top-5 candidates)_")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — FULL AGENT RUN
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Full Agent Run":
    st.title("🤖 Full LangGraph Agent Run")
    st.caption("Runs the complete 10-node state machine: fetch → indicators → metrics → screen → guardrails → report → grounding.")

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            groq_key = os.environ.get("GROQ_API_KEY", "")
        except Exception:
            pass

    if not groq_key:
        st.warning("⚠️ GROQ_API_KEY not found in environment. Add it to your `.env` file.")

    st.subheader("📋 Portfolio for Agent Run")

    if "agent_holdings" not in st.session_state:
        st.session_state.agent_holdings = [
            {"symbol": "RELIANCE", "quantity": 10, "avg_price": 2800.0},
            {"symbol": "TCS", "quantity": 5, "avg_price": 3500.0},
            {"symbol": "HDFCBANK", "quantity": 20, "avg_price": 1600.0},
        ]

    agent_df = pd.DataFrame(st.session_state.agent_holdings)
    agent_edited = st.data_editor(agent_df, num_rows="dynamic", use_container_width=True, key="agent_editor")

    trigger_type = st.selectbox("Trigger Type", ["manual_query", "scheduled_scan", "portfolio_update"])
    run_agent = st.button("🚀 Run Full Agent Pipeline", use_container_width=False)

    if run_agent:
        holdings = agent_edited.dropna().to_dict("records")
        holdings = [h for h in holdings if h.get("symbol")]

        if not holdings:
            st.error("Add at least one holding.")
            st.stop()

        with st.status("Running LangGraph agent pipeline...", expanded=True) as status:
            st.write("🔄 Fetching market data & validating symbols...")
            st.write("📊 Computing technical indicators...")
            st.write("💼 Computing portfolio metrics...")
            st.write("🔎 Running factor screening...")
            st.write("🛡️ Validating alerts through guardrails...")
            st.write("📝 Generating report with grounding verification...")

            try:
                from agent.graph import run_portfolio_analysis
                result = run_async(run_portfolio_analysis(
                    portfolio=holdings,
                    trigger_type=trigger_type,
                    thread_id=f"ui_{int(time.time())}",
                ))
                status.update(label="✅ Agent pipeline complete!", state="complete")
            except Exception as e:
                status.update(label=f"❌ Error: {e}", state="error")
                st.error(str(e))
                st.stop()

        # Results
        col1, col2, col3, col4 = st.columns(4)
        validated = len(result.get("validated_alerts", []))
        blocked = len(result.get("blocked_alerts", []))
        grounding = "PASS ✅" if result.get("grounding_check_passed") else "FAIL ❌"
        delivery = result.get("delivery_status", {}).get("status", "N/A")
        col1.metric("Validated Alerts", validated)
        col2.metric("Blocked by Guardrails", blocked)
        col3.metric("Grounding Check", grounding)
        col4.metric("Delivery", delivery)

        # Catch rate
        total = validated + blocked
        if total > 0:
            catch_rate = blocked / total
            st.metric("Guardrail Catch Rate", f"{catch_rate:.1%}")

        st.divider()

        # Final report
        st.subheader("📄 Generated Report")
        report = result.get("report_draft", "No report generated.")
        st.markdown(f"""
        <div style='background:#1a1a2e; border:1px solid #1e3a5f; border-radius:12px; padding:1.5rem; font-size:0.9rem; line-height:1.7; white-space:pre-wrap;'>{report}</div>
        """, unsafe_allow_html=True)

        # Alerts
        if result.get("validated_alerts"):
            st.subheader("🔔 Validated Alerts")
            for alert in result["validated_alerts"]:
                sev = alert.get("severity", "low")
                badge_class = f"badge-{sev}"
                st.markdown(f"""
                <div style='background:#1a1a2e; border:1px solid #1e3a5f; border-radius:10px; padding:1rem; margin:0.5rem 0;'>
                    <span class='badge {badge_class}'>{sev.upper()}</span>
                    <b style='color:#00d4ff;'> {alert.get('symbol')}</b> — {alert.get('alert_type', '').replace('_', ' ').title()}
                    <p style='color:#c8d6e5; margin:0.5rem 0 0;'>{alert.get('rationale', '')}</p>
                </div>
                """, unsafe_allow_html=True)

        if result.get("blocked_alerts"):
            st.subheader("🚫 Blocked Alerts (Guardrails)")
            for ba in result["blocked_alerts"]:
                st.warning(f"**{ba['alert']['symbol']}** — {', '.join(ba['reasons'])}")

        with st.expander("🔬 Full State Dump"):
            safe_result = {k: v for k, v in result.items() if k != "tool_call_registry"}
            st.json(safe_result)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📉 Backtesting":
    st.title("📉 Backtesting")
    st.caption("Factor strategy vs Nifty 50 | Alert hit-rate for signal rules. Binary ground truth — no LLM judge.")

    tab1, tab2 = st.tabs(["📈 Factor Strategy vs Nifty 50", "🎯 Alert Hit-Rate"])

    with tab1:
        st.subheader("Factor-Screened Portfolio vs Nifty 50 Benchmark")
        col1, col2, col3 = st.columns(3)
        with col1:
            bt_start = st.date_input("Start Date", value=datetime(2023, 1, 1))
        with col2:
            bt_end = st.date_input("End Date", value=datetime(2024, 12, 31))
        with col3:
            rebal_months = st.selectbox("Rebalance Every", [1, 3, 6], index=1, format_func=lambda x: f"{x} months")

        run_bt = st.button("Run Factor Backtest", use_container_width=False)

        if run_bt:
            with st.spinner("Running backtest (fetching 2 years of data, may take ~60s)..."):
                try:
                    from backtest.factor_backtest import run_factor_backtest
                    bt_result = run_async(run_factor_backtest(
                        start_date=str(bt_start),
                        end_date=str(bt_end),
                        top_n=15,
                        rebalance_months=rebal_months,
                    ))
                except Exception as e:
                    st.error(f"Backtest error: {e}")
                    st.stop()

            if "error" in bt_result:
                st.error(bt_result["error"])
                st.stop()

            ps = bt_result.get("portfolio_strategy", {})
            bs = bt_result.get("benchmark_nifty_50", {})
            alpha = bt_result.get("alpha", 0)

            # Metrics comparison
            st.subheader("📊 Strategy vs Benchmark")
            col1, col2, col3 = st.columns(3)
            col1.metric("Factor Strategy Sharpe", f"{ps.get('sharpe_ratio', 'N/A'):.4f}" if isinstance(ps.get('sharpe_ratio'), float) else "N/A",
                        delta=f"vs {bs.get('sharpe_ratio', 0):.2f} Nifty",
                        delta_color="normal" if (ps.get('sharpe_ratio') or 0) > (bs.get('sharpe_ratio') or 0) else "inverse")
            col2.metric("Cumulative Return", f"{ps.get('cumulative_return_pct', 0):+.2f}%",
                        delta=f"vs {bs.get('cumulative_return_pct', 0):+.2f}% Nifty",
                        delta_color="normal" if (ps.get('cumulative_return_pct') or 0) > (bs.get('cumulative_return_pct') or 0) else "inverse")
            col3.metric("Alpha (annualized)", f"{alpha:+.2f}%",
                        delta_color="normal" if alpha > 0 else "inverse")

            col1.metric("Volatility", f"{ps.get('annualized_volatility_pct', 0):.2f}%")
            col2.metric("Max Drawdown", f"{ps.get('max_drawdown_pct', 0):.2f}%")
            col3.metric("Nifty Max Drawdown", f"{bs.get('max_drawdown_pct', 0):.2f}%")

            # Check for chart file
            chart_path = bt_result.get("chart_path")
            if chart_path and os.path.exists(chart_path):
                st.subheader("📈 Cumulative Returns Chart")
                st.image(chart_path, use_container_width=True)

            # Comparison bar chart
            fig_compare = go.Figure()
            categories = ["Sharpe Ratio", "Ann. Return %", "Volatility %", "Max Drawdown %"]
            strategy_vals = [
                ps.get("sharpe_ratio", 0),
                ps.get("annualized_return_pct", 0),
                ps.get("annualized_volatility_pct", 0),
                abs(ps.get("max_drawdown_pct", 0)),
            ]
            bench_vals = [
                bs.get("sharpe_ratio", 0),
                bs.get("annualized_return_pct", 0),
                bs.get("annualized_volatility_pct", 0),
                abs(bs.get("max_drawdown_pct", 0)),
            ]
            fig_compare.add_trace(go.Bar(name="Factor Strategy", x=categories, y=strategy_vals, marker_color="#00d4ff"))
            fig_compare.add_trace(go.Bar(name="Nifty 50", x=categories, y=bench_vals, marker_color="#ff9500"))
            plotly_dark_layout(fig_compare, "Strategy vs Nifty 50 — Key Metrics", height=380)
            fig_compare.update_layout(barmode="group")
            st.plotly_chart(fig_compare, use_container_width=True)

    with tab2:
        st.subheader("Alert Rule Hit-Rate Backtest")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            bt_symbol = st.text_input("Symbol", "RELIANCE").upper()
        with col2:
            rule_type = st.selectbox("Signal Rule", ["rsi_oversold", "rsi_overbought", "macd_crossover", "golden_cross", "death_cross", "bb_breakout"])
        with col3:
            hold_days = st.slider("Hold Days", 1, 30, 5)
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            run_alert_bt = st.button("Run Hit-Rate Backtest", use_container_width=True)

        if run_alert_bt:
            rule = {"type": rule_type, "hold_days": hold_days}
            if rule_type in ["rsi_oversold", "rsi_overbought"]:
                threshold = st.session_state.get("rsi_threshold", 30 if "oversold" in rule_type else 70)
                rule["threshold"] = threshold

            with st.spinner(f"Backtesting {rule_type} on {bt_symbol}..."):
                from mcp_server.tools.backtest import backtest_alert_rule
                bt_res = run_async(backtest_alert_rule(
                    symbol=bt_symbol,
                    rule=rule,
                    start_date="2023-01-01",
                    end_date="2024-12-31",
                ))

            if "error" in bt_res:
                st.error(bt_res["error"])
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Signals", bt_res.get("total_signals", 0))
                c2.metric("Hit Rate", f"{bt_res.get('hit_rate_pct', 0):.1f}%",
                          delta="Good (>55%)" if (bt_res.get("hit_rate_pct") or 0) > 55 else "Low (<50%)",
                          delta_color="normal" if (bt_res.get("hit_rate_pct") or 0) > 55 else "inverse")
                c3.metric("Sharpe", f"{bt_res.get('sharpe_ratio', 0):.4f}")
                c4.metric("Avg Return/Trade", f"{bt_res.get('avg_return_per_trade_pct', 0):+.2f}%",
                          delta_color="normal" if (bt_res.get("avg_return_per_trade_pct") or 0) > 0 else "inverse")

                trades = bt_res.get("trades", [])
                if trades:
                    df_trades = pd.DataFrame(trades)
                    df_trades["color"] = df_trades["direction_correct"].map({True: "#00d464", False: "#ff4444"})

                    fig_trades = go.Figure(go.Bar(
                        x=df_trades["entry_date"],
                        y=df_trades["return_pct"],
                        marker_color=df_trades["color"],
                        text=df_trades["return_pct"].apply(lambda x: f"{x:+.1f}%"),
                        textposition="outside",
                    ))
                    plotly_dark_layout(fig_trades, f"{bt_symbol} — Trade Returns ({rule_type})", height=380)
                    fig_trades.add_hline(y=0, line_dash="dash", line_color="#7a8fa6")
                    st.plotly_chart(fig_trades, use_container_width=True)

                    st.subheader("📋 Trade Log")
                    st.dataframe(pd.DataFrame(trades), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — GUARDRAILS DEMO
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🛡️ Guardrails Demo":
    st.title("🛡️ Guardrails in Action")
    st.caption("Live demonstration of each guardrail — what gets caught, why, and how it's logged.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "5.1 Symbol Validation",
        "5.3 Human-in-Loop",
        "5.4 Rate Limiting",
        "5.7 Concentration Risk",
    ])

    with tab1:
        st.subheader("Guardrail 5.1 — Hallucinated Ticker Detection")
        st.info("Every ticker must pass `validate_symbol` before any analysis. This prevents hallucinated stocks from reaching the agent.")

        col1, col2 = st.columns(2)
        with col1:
            test_symbols = st.text_area(
                "Enter symbols to test (one per line)",
                value="RELIANCE\nFAKESTOCKXYZ\nTCS\nINVALIDCORP\nHDFCBANK",
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_validation = st.button("Run Validation on All", use_container_width=True)

        if run_validation:
            symbols = [s.strip().upper() for s in test_symbols.split("\n") if s.strip()]
            results = []
            prog = st.progress(0)
            for i, sym in enumerate(symbols):
                v = cached_validate(sym)
                results.append({"Symbol": sym, "Valid": v["valid"], "Message": v["message"]})
                prog.progress((i + 1) / len(symbols))

            df_val = pd.DataFrame(results)
            passed = df_val[df_val["Valid"] == True]
            blocked = df_val[df_val["Valid"] == False]

            c1, c2 = st.columns(2)
            c1.metric("Valid Symbols", len(passed), delta_color="normal")
            c2.metric("Blocked (hallucinated)", len(blocked), delta_color="inverse")

            for _, row in df_val.iterrows():
                icon = "✅" if row["Valid"] else "❌"
                color = "#00d464" if row["Valid"] else "#ff4444"
                st.markdown(f"""
                <div style='background:#1a1a2e; border-left:4px solid {color}; padding:0.8rem 1rem; margin:0.4rem 0; border-radius:0 8px 8px 0;'>
                    {icon} <b style='color:{color};'>{row['Symbol']}</b> — {row['Message']}
                </div>
                """, unsafe_allow_html=True)

            # Catch rate
            catch_rate = len(blocked) / len(df_val)
            st.metric("Symbol Validation Catch Rate", f"{catch_rate:.1%}")
            st.caption("This is the first line of defence — blocks hallucinated tickers before any API call or analysis.")

    with tab2:
        st.subheader("Guardrail 5.3 — Human Approval for High-Severity Alerts")
        st.info("Any alert classified `severity: high` is held in the interrupt node before delivery. Demo shows the severity classification logic.")

        col1, col2 = st.columns(2)
        with col1:
            var_95 = st.number_input("VaR (95%) %", value=6.5, step=0.1, format="%.1f")
            sector_max = st.number_input("Max Sector Weight %", value=45.0, step=1.0)
        with col2:
            stock_max = st.number_input("Max Single Stock Weight %", value=30.0, step=1.0)
            price_move = st.number_input("Price Move % (1 day)", value=9.0, step=0.5)

        if st.button("Check Severity Classification"):
            flags = []
            if var_95 > 5.0:
                flags.append(f"VaR(95%) {var_95:.1f}% > 5.0% threshold")
            if sector_max > 40.0:
                flags.append(f"Sector concentration {sector_max:.1f}% > 40% limit")
            if stock_max > 25.0:
                flags.append(f"Single stock weight {stock_max:.1f}% > 25% limit")
            if price_move >= 8.0:
                flags.append(f"Price move {price_move:.1f}% >= 8% high-severity threshold")

            if flags:
                st.error("🔴 **HIGH SEVERITY** — Human approval required before delivery")
                for f in flags:
                    st.markdown(f"- ⚠️ {f}")
                st.markdown("""
                <div style='background:#2a1a1a; border:1px solid #ff3232; border-radius:10px; padding:1rem; margin:1rem 0;'>
                    <b style='color:#ff6464;'>LangGraph Interrupt Triggered</b><br>
                    <code style='color:#e0e0e0;'>interrupt({"message": "High-severity alert requires approval", "alerts": [...]})</code><br>
                    <small style='color:#7a8fa6;'>Execution suspended — waiting for external approval signal via Postgres checkpoint</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("🟢 **LOW/MEDIUM SEVERITY** — Auto-route to report generation")

    with tab3:
        st.subheader("Guardrail 5.4 — Rate Limiting (Alert Cooldown)")
        st.info("Each symbol has a configurable cooldown window (default 6h) to prevent alert fatigue.")

        cooldown_data = {
            "Symbol": ["RELIANCE", "TCS", "HDFCBANK", "INFY"],
            "Alert Type": ["price_move", "technical_signal", "portfolio_risk", "price_move"],
            "Last Sent": [
                (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"),
                (datetime.now() - timedelta(hours=8)).strftime("%Y-%m-%d %H:%M"),
                (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"),
                (datetime.now() - timedelta(hours=10)).strftime("%Y-%m-%d %H:%M"),
            ],
            "Cooldown (hrs)": [6, 6, 6, 6],
        }
        df_cd = pd.DataFrame(cooldown_data)
        df_cd["Hours Since"] = [(datetime.now() - datetime.strptime(t, "%Y-%m-%d %H:%M")).total_seconds() / 3600 for t in df_cd["Last Sent"]]
        df_cd["Status"] = df_cd["Hours Since"].apply(lambda h: "🔒 BLOCKED" if h < 6 else "✅ ALLOWED")

        st.dataframe(df_cd[["Symbol", "Alert Type", "Last Sent", "Hours Since", "Status"]].round(1), use_container_width=True)

        blocked_count = (df_cd["Hours Since"] < 6).sum()
        st.metric("Would-be alerts blocked by rate limit", f"{blocked_count}/{len(df_cd)}")

    with tab4:
        st.subheader("Guardrail 5.7 — Sector Concentration Risk Ceiling")
        current_banking = st.slider("Current Banking Sector Weight %", 0, 100, 35)
        proposed_add = st.selectbox("Proposed new holding", ["KOTAKBANK (Banking)", "INFY (Technology)", "SUNPHARMA (Pharma)"])
        proposed_weight = st.slider("Weight it would add %", 1, 30, 12)

        sector = "Banking" if "Banking" in proposed_add else ("Technology" if "Technology" in proposed_add else "Pharma")
        new_weight = current_banking + proposed_weight if sector == "Banking" else proposed_weight

        col1, col2 = st.columns(2)
        col1.metric("Current Banking Weight", f"{current_banking}%")
        col2.metric("Post-addition Weight", f"{new_weight}%", delta=f"+{proposed_weight}%",
                    delta_color="inverse" if new_weight > 40 else "normal")

        if sector == "Banking" and new_weight > 40:
            st.error(f"🚫 **BLOCKED by Guardrail 5.7** — Banking would reach {new_weight}% (> 40% limit). Recommendation rejected regardless of factor score.")
        else:
            st.success(f"✅ **ALLOWED** — {sector} exposure would be {new_weight}% (within 40% limit).")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — SYSTEM CHECK
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚙️ System Check":
    st.title("⚙️ System Check & Installation Verification")
    st.caption("Verifies all dependencies, API keys, and MCP tools are working correctly.")

    if st.button("Run Full System Check", use_container_width=False):
        checks = []

        # Package checks
        packages = {
            "yfinance": "NSE market data",
            "pandas": "Data manipulation",
            "numpy": "Numerical computation",
            "pandas_ta": "Technical indicators",
            "langchain_groq": "Groq LLM integration",
            "langgraph": "Agent orchestration",
            "mcp": "MCP server protocol",
            "matplotlib": "Chart generation",
            "seaborn": "Statistical plots",
            "plotly": "Interactive charts",
            "streamlit": "UI framework",
            "dotenv": "Environment config",
        }

        st.subheader("📦 Package Verification")
        pkg_cols = st.columns(3)
        for i, (pkg, desc) in enumerate(packages.items()):
            try:
                mod = __import__(pkg.replace("-", "_"))
                ver = getattr(mod, "__version__", "installed")
                pkg_cols[i % 3].success(f"✅ **{pkg}** v{ver}")
                checks.append(("package", pkg, True, ver))
            except ImportError as e:
                pkg_cols[i % 3].error(f"❌ **{pkg}** — NOT INSTALLED")
                checks.append(("package", pkg, False, str(e)))

        # Environment checks
        st.subheader("🔑 Environment Variables")
        from dotenv import load_dotenv
        load_dotenv()

        env_vars = {
            "GROQ_API_KEY": "LLM (required)",
            "LANGCHAIN_API_KEY": "LangSmith (optional)",
            "TWILIO_ACCOUNT_SID": "WhatsApp (optional)",
            "WHATSAPP_TO": "Delivery target (optional)",
        }
        env_cols = st.columns(2)
        for i, (var, desc) in enumerate(env_vars.items()):
            val = os.environ.get(var, "")
            masked = val[:8] + "..." if val else "NOT SET"
            if val:
                env_cols[i % 2].success(f"✅ **{var}**: `{masked}` — {desc}")
            else:
                env_cols[i % 2].warning(f"⚠️ **{var}**: NOT SET — {desc}")

        # MCP tool checks
        st.subheader("🛠️ MCP Tool Verification")
        tool_checks = [
            ("validate_symbol", lambda: run_async(__import__("mcp_server.tools.validate_symbol", fromlist=["validate_symbol"]).validate_symbol("RELIANCE"))),
            ("get_equity_details", lambda: run_async(__import__("mcp_server.tools.market_data", fromlist=["get_equity_details"]).get_equity_details("TCS"))),
            ("compute_technical_indicators", lambda: run_async(__import__("mcp_server.tools.technical_indicators", fromlist=["compute_technical_indicators"]).compute_technical_indicators("HDFCBANK", ["RSI"], 60))),
        ]

        tool_cols = st.columns(len(tool_checks))
        for i, (tool_name, tool_fn) in enumerate(tool_checks):
            with tool_cols[i]:
                with st.spinner(f"Testing {tool_name}..."):
                    try:
                        result = tool_fn()
                        st.success(f"✅ **{tool_name}**")
                        with st.expander("Output"):
                            st.json(result if isinstance(result, dict) else {"result": str(result)})
                    except Exception as e:
                        st.error(f"❌ **{tool_name}**")
                        st.caption(str(e)[:100])

        # Summary
        st.divider()
        passed = sum(1 for _, _, ok, _ in checks if ok)
        total = len(checks)
        pct = passed / total * 100

        if pct >= 90:
            st.success(f"🎉 System ready: **{passed}/{total}** packages installed ({pct:.0f}%)")
        elif pct >= 70:
            st.warning(f"⚠️ Partial setup: **{passed}/{total}** packages installed ({pct:.0f}%). Some features may not work.")
        else:
            st.error(f"❌ Setup incomplete: **{passed}/{total}** packages installed. Run: `pip install -r requirements.txt`")

        st.code("pip install -r requirements.txt", language="bash")
