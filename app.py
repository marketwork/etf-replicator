"""ETF Replicator – Streamlit app."""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta

from data import fetch_prices, to_returns, get_universe, get_stock_meta
from optimizer import replicate


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ETF Replicator",
    page_icon="🔬",
    layout="wide",
)

# ── Colour palette ────────────────────────────────────────────────────────────
ETF_CLR = "#4A90D9"
REP_CLR = "#E86B3A"
GRID_CLR = "rgba(200,200,200,0.15)"

POPULAR_ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "VTI",
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLP", "XLU", "XLB", "XLRE",
    "GLD", "TLT", "HYG", "EEM", "EFA",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fmt_pct(v: float, decimals: int = 2) -> str:
    return f"{v * 100:.{decimals}f}%"


def _metric_card(col, label: str, value: str, delta: str = ""):
    col.metric(label, value, delta or None)


def _plotly_base() -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CCCCCC", size=12),
        xaxis=dict(showgrid=True, gridcolor=GRID_CLR, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=GRID_CLR, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_cached(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    return fetch_prices(list(tickers), start, end)


@st.cache_data(ttl=1800, show_spinner=False)
def _get_universe_cached(etf: str) -> tuple:
    return get_universe(etf)


@st.cache_data(ttl=3600, show_spinner=False)
def _meta_cached(tickers: tuple) -> pd.DataFrame:
    return get_stock_meta(list(tickers))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 ETF Replicator")
    st.caption("Find 20-30 stocks that best replicate any ETF")
    st.divider()

    etf_input = st.text_input(
        "Target ETF",
        value="SPY",
        help="Ticker of the ETF to replicate (e.g. SPY, QQQ, XLK)",
    ).upper().strip()

    st.caption("Popular ETFs: " + " · ".join(POPULAR_ETFS[:10]))

    today = date.today()
    col_s, col_e = st.columns(2)
    start_date = col_s.date_input("Start", value=today - timedelta(days=3 * 365))
    end_date = col_e.date_input("End", value=today)

    n_stocks = st.slider("Stocks in replica", min_value=20, max_value=30, value=25)

    method = st.selectbox(
        "Selection method",
        options=["lasso", "greedy", "hybrid"],
        format_func=lambda x: {
            "lasso": "LASSO (fast, ~5s)",
            "greedy": "Greedy (moderate, ~30s)",
            "hybrid": "Hybrid (best quality, ~45s)",
        }[x],
    )

    with st.expander("Advanced"):
        train_pct = st.slider("Training data %", 50, 90, 70,
                              help="Remaining % is used for out-of-sample evaluation")
        min_cov = st.slider("Min data coverage %", 50, 99, 80,
                            help="Drop stocks with fewer observations than this")

    st.divider()
    run_btn = st.button("Run optimisation", type="primary", use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("## ETF Replicator")
st.caption(
    "Selects an optimal subset of stocks whose blended return best tracks "
    "a target ETF, minimising annualised tracking error."
)

if not run_btn:
    # Landing info
    c1, c2, c3 = st.columns(3)
    c1.info("**LASSO**\nSparse regression — fast and consistent.", icon="⚡")
    c2.info("**Greedy**\nCorrelation-driven forward selection — intuitive.", icon="🔁")
    c3.info("**Hybrid**\nLASSO pre-filters, greedy refines — best of both.", icon="🏆")

    st.markdown("""
**How it works**

1. Pull the constituent universe for your ETF (S&P 500 for SPY, NASDAQ-100 for QQQ, etc.)
2. Download daily prices for all constituents and compute returns
3. Apply the chosen selection method to pick the *n* stocks with the highest
   collective power to replicate the ETF's return
4. Solve a constrained QP: **min_w ‖S·w – e‖²** subject to **Σwᵢ = 1, wᵢ ≥ 0**
5. Report in-sample and out-of-sample tracking error, R², and correlation
    """)
    st.stop()


# ── Run ───────────────────────────────────────────────────────────────────────
progress_bar = st.progress(0, text="Fetching universe…")

universe, universe_label = _get_universe_cached(etf_input)
if not universe:
    st.error(f"Could not find a constituent universe for {etf_input}.")
    st.stop()

progress_bar.progress(10, text=f"Universe: {universe_label}. Downloading prices…")

all_tickers = tuple(sorted(set([etf_input] + universe)))
try:
    prices = _fetch_cached(all_tickers, str(start_date), str(end_date))
except Exception as exc:
    st.error(f"Price download failed: {exc}")
    st.stop()

if etf_input not in prices.columns:
    st.error(f"{etf_input} price data not available for the selected date range.")
    st.stop()

progress_bar.progress(35, text="Computing returns…")

etf_prices = prices[etf_input]
stock_prices = prices.drop(columns=[etf_input], errors="ignore")

returns = to_returns(prices, min_coverage=min_cov / 100)
etf_r = returns[etf_input] if etf_input in returns.columns else to_returns(
    etf_prices.to_frame(), 0.5
).iloc[:, 0]
stock_r = returns.drop(columns=[etf_input], errors="ignore")

# Remove tickers with no data
stock_r = stock_r.loc[:, stock_r.abs().sum() > 0]

n_avail = len(stock_r.columns)
progress_bar.progress(40, text=f"{n_avail} stocks available. Running optimiser…")


def _progress(frac: float):
    pct = int(40 + frac * 50)
    progress_bar.progress(pct, text=f"Optimising… {int(frac * 100)}%")


try:
    result = replicate(
        etf_r=etf_r,
        stock_r=stock_r,
        n_stocks=n_stocks,
        method=method,
        train_frac=train_pct / 100,
        progress_cb=_progress,
    )
except Exception as exc:
    st.error(f"Optimisation failed: {exc}")
    st.stop()

progress_bar.progress(95, text="Fetching stock metadata…")
meta = _meta_cached(tuple(result["tickers"]))
progress_bar.progress(100, text="Done!")
progress_bar.empty()


# ── Metric row ────────────────────────────────────────────────────────────────
st.markdown("### Results")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Stocks selected", result["n_stocks"])
m2.metric("Tracking error (IS)", _fmt_pct(result["tracking_error"]))
m3.metric("Tracking error (OOS)", _fmt_pct(result["oos_tracking_error"]),
          delta=_fmt_pct(result["oos_tracking_error"] - result["tracking_error"]),
          delta_color="inverse")
m4.metric("R² (in-sample)", f"{result['r_squared']:.4f}")
m5.metric("Correlation (IS)", f"{result['correlation']:.4f}")
m6.metric("Correlation (OOS)", f"{result['oos_correlation']:.4f}")

st.caption(
    f"Universe: **{universe_label}** · "
    f"Train: **{start_date} → {result['train_cutoff'].date()}** · "
    f"OOS: **{result['train_cutoff'].date()} → {end_date}**"
)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_perf, tab_hold, tab_analysis = st.tabs(["📈 Performance", "📋 Holdings", "🔬 Analysis"])


# ──── Tab 1: Performance ──────────────────────────────────────────────────────
with tab_perf:
    st.subheader("Cumulative Returns")

    train_cut = result["train_cutoff"]

    fig = go.Figure()
    fig.add_vrect(
        x0=str(etf_r.index[0].date()), x1=str(train_cut.date()),
        fillcolor="rgba(100,150,255,0.06)", line_width=0,
        annotation_text="In-sample", annotation_position="top left",
        annotation_font_color="#8899AA",
    )
    fig.add_vline(x=str(train_cut.date()), line_dash="dot", line_color="#8899AA",
                  annotation_text="OOS start", annotation_position="top right",
                  annotation_font_color="#8899AA")
    fig.add_trace(go.Scatter(
        x=result["etf_cum"].index, y=result["etf_cum"].values,
        name=etf_input, line=dict(color=ETF_CLR, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=result["port_cum"].index, y=result["port_cum"].values,
        name=f"Replica ({result['n_stocks']} stocks)", line=dict(color=REP_CLR, width=2),
    ))
    fig.update_layout(
        **_plotly_base(),
        yaxis_title="Growth of $1",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rolling tracking error
    st.subheader("Rolling 63-Day Tracking Error (annualised)")
    rte = result["rolling_te"].dropna()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=rte.index, y=rte.values * 100,
        fill="tozeroy", fillcolor="rgba(232,107,58,0.15)",
        line=dict(color=REP_CLR, width=1.5),
        name="Rolling TE (%)",
    ))
    fig2.add_hline(y=result["tracking_error"] * 100, line_dash="dash",
                   line_color=ETF_CLR, annotation_text="Avg TE",
                   annotation_position="top left")
    fig2.update_layout(**_plotly_base(), yaxis_title="Tracking Error (%)", height=280)
    st.plotly_chart(fig2, use_container_width=True)


# ──── Tab 2: Holdings ─────────────────────────────────────────────────────────
with tab_hold:
    tickers = result["tickers"]
    weights = result["weights"]

    # Build holdings table
    df_hold = pd.DataFrame({
        "Weight": weights,
        "Weight %": [_fmt_pct(w) for w in weights],
    }, index=tickers)
    df_hold.index.name = "Ticker"

    if not meta.empty:
        df_hold = df_hold.join(meta[["name", "sector", "market_cap_B"]], how="left")
        df_hold.rename(columns={
            "name": "Name", "sector": "Sector", "market_cap_B": "Mkt Cap ($B)"
        }, inplace=True)

    df_hold = df_hold.sort_values("Weight", ascending=False)

    col_tbl, col_bar = st.columns([1, 1])

    with col_tbl:
        st.subheader(f"Selected {result['n_stocks']} stocks")
        display_cols = ["Weight %"] + [c for c in ["Name", "Sector", "Mkt Cap ($B)"]
                                        if c in df_hold.columns]
        st.dataframe(
            df_hold[display_cols],
            use_container_width=True,
            height=min(600, 40 + 35 * len(df_hold)),
        )

    with col_bar:
        st.subheader("Weight distribution")
        fig3 = go.Figure(go.Bar(
            x=df_hold["Weight"].values * 100,
            y=df_hold.index.tolist(),
            orientation="h",
            marker_color=REP_CLR,
            text=[f"{w:.1f}%" for w in df_hold["Weight"].values * 100],
            textposition="outside",
        ))
        fig3.update_layout(
            **_plotly_base(),
            xaxis_title="Weight (%)",
            height=max(400, 28 * len(df_hold)),
            yaxis=dict(autorange="reversed", showgrid=False),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Sector breakdown if available
    if "Sector" in df_hold.columns:
        st.subheader("Sector breakdown")
        sec = (df_hold.groupby("Sector")["Weight"]
                      .sum()
                      .sort_values(ascending=False))
        fig4 = go.Figure(go.Pie(
            labels=sec.index.tolist(),
            values=(sec.values * 100).round(1),
            hole=0.4,
            textinfo="label+percent",
            marker_colors=[
                "#4A90D9", "#E86B3A", "#50C878", "#FFD700", "#C678DD",
                "#56B6C2", "#E06C75", "#98C379", "#D19A66", "#ABB2BF",
            ],
        ))
        fig4.update_layout(
            **_plotly_base(),
            height=360,
            showlegend=False,
        )
        st.plotly_chart(fig4, use_container_width=True)


# ──── Tab 3: Analysis ────────────────────────────────────────────────────────
with tab_analysis:
    col_sc, col_diff = st.columns(2)

    # Return scatter
    with col_sc:
        st.subheader("Daily return scatter")
        etf_plot = result["etf_returns"] * 100
        port_plot = result["port_returns"] * 100
        fig5 = go.Figure()
        # 45° reference line
        rng = max(abs(etf_plot.min()), abs(etf_plot.max())) * 1.1
        fig5.add_trace(go.Scatter(
            x=[-rng, rng], y=[-rng, rng],
            mode="lines", line=dict(color="#666", dash="dot", width=1),
            showlegend=False,
        ))
        fig5.add_trace(go.Scatter(
            x=etf_plot.values, y=port_plot.values,
            mode="markers",
            marker=dict(size=3, color=REP_CLR, opacity=0.5),
            name="Daily returns",
        ))
        fig5.update_layout(
            **_plotly_base(),
            xaxis_title=f"{etf_input} return (%)",
            yaxis_title="Replica return (%)",
            height=360,
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Return difference
    with col_diff:
        st.subheader("Daily return difference (replica − ETF)")
        diff = (result["port_returns"] - result["etf_returns"]) * 100
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(
            x=diff.index, y=diff.values,
            marker_color=np.where(diff.values >= 0, "#50C878", "#E06C75"),
            name="Return diff",
        ))
        fig6.add_hline(y=0, line_color="#888", line_width=1)
        fig6.update_layout(
            **_plotly_base(),
            yaxis_title="Difference (% pts)",
            height=360,
            bargap=0,
        )
        st.plotly_chart(fig6, use_container_width=True)

    # Annualised return comparison table
    st.subheader("Annualised performance summary")

    def _ann_ret(r: pd.Series) -> float:
        n_years = len(r) / 252
        return float((1 + r).prod() ** (1 / n_years) - 1) if n_years > 0 else 0.0

    def _ann_vol(r: pd.Series) -> float:
        return float(r.std(ddof=1) * np.sqrt(252))

    def _max_dd(r: pd.Series) -> float:
        cum = (1 + r).cumprod()
        return float((cum / cum.cummax() - 1).min())

    rows = []
    for label, ret in [(etf_input, result["etf_returns"]),
                        (f"Replica ({result['n_stocks']})", result["port_returns"])]:
        rows.append({
            "": label,
            "Ann. Return": _fmt_pct(_ann_ret(ret)),
            "Ann. Volatility": _fmt_pct(_ann_vol(ret)),
            "Max Drawdown": _fmt_pct(_max_dd(ret)),
            "Sharpe (0% rf)": f"{_ann_ret(ret) / _ann_vol(ret):.2f}" if _ann_vol(ret) else "—",
        })

    df_perf = pd.DataFrame(rows).set_index("")
    st.dataframe(df_perf, use_container_width=True)

    # Download weights
    st.divider()
    st.subheader("Export")
    export_df = pd.DataFrame({
        "Ticker": result["tickers"],
        "Weight": result["weights"],
        "Weight_%": [round(w * 100, 2) for w in result["weights"]],
    })
    if not meta.empty:
        export_df = export_df.join(meta[["name", "sector"]], on="Ticker")
    csv_bytes = export_df.to_csv(index=False).encode()
    st.download_button(
        "Download weights as CSV",
        data=csv_bytes,
        file_name=f"{etf_input}_replica_{n_stocks}stocks.csv",
        mime="text/csv",
    )
