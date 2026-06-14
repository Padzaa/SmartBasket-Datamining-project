"""
Page 3 — Association Rules Explorer

Interactive filterable table of association rules with charts.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "src"))
from rule_filtering import filter_rules
import theme

st.set_page_config(page_title="Rules Explorer · SmartBasket AI", page_icon="🔍", layout="wide")
theme.apply_theme()


# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_rules():
    return pd.read_csv(BASE / "models" / "association_rules.csv")

with st.spinner("Loading rules…"):
    rules_raw = load_rules()


# ── Masthead ────────────────────────────────────────────────────────────────--
theme.masthead(
    eyebrow="Association Rules",
    title="Rules Explorer",
    subtitle=f"Browse, filter and visualise the {len(rules_raw):,} rules mined by FP-Growth. "
             "Use the sidebar sliders to narrow down to the strongest patterns.",
)


# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##### Filter rules")

    min_support = st.slider(
        "Min support", 0.001, 0.05, 0.001, 0.001,
        format="%.3f",
        help="Fraction of transactions containing the itemset",
    )
    min_confidence = st.slider(
        "Min confidence", 0.05, 1.0, 0.20, 0.05,
        format="%.2f",
        help="P(consequent | antecedent)",
    )
    min_lift = st.slider(
        "Min lift", 1.0, 20.0, 1.2, 0.1,
        format="%.1f",
        help="How much more likely vs independent purchase",
    )
    top_n = st.selectbox("Show top N rows", [25, 50, 100, 250, 500], index=1)

    st.markdown("---")
    search_term = st.text_input("🔎 Search product",
                                placeholder="e.g. JUMBO BAG",
                                help="Filter rules containing this text in antecedent or consequent")

    sort_by = st.radio("Sort by", ["lift", "confidence", "support"], horizontal=True)


# ── Apply filters ──────────────────────────────────────────────────────────────
filtered = filter_rules(
    rules_raw,
    min_support=min_support,
    min_confidence=min_confidence,
    min_lift=min_lift,
)

if search_term.strip():
    term = search_term.strip().upper()
    mask = (
        filtered["antecedents_str"].str.upper().str.contains(term, na=False)
        | filtered["consequents_str"].str.upper().str.contains(term, na=False)
    )
    filtered = filtered[mask]

filtered = filtered.sort_values(sort_by, ascending=False).head(top_n)


# ── Toolbar: count + download ──────────────────────────────────────────────────
col_badge, col_dl = st.columns([3, 1])
with col_badge:
    st.markdown(
        f'<span class="sb-count">Showing {len(filtered):,} rules</span>'
        f'<div class="sb-hint">Sorted by {sort_by} · {len(rules_raw):,} total rules in dataset</div>',
        unsafe_allow_html=True,
    )
with col_dl:
    dl_df = filtered[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].copy()
    dl_df.columns = ["Antecedent", "Consequent", "Support", "Confidence", "Lift"]
    st.download_button(
        "⬇ Export CSV",
        data=dl_df.to_csv(index=False),
        file_name="filtered_rules.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("")

display = filtered[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].copy()
display.columns = ["Antecedent (IF)", "Consequent (THEN)", "Support", "Confidence", "Lift"]
display["Support"]    = display["Support"].apply(lambda x: f"{x:.3f}")
display["Confidence"] = display["Confidence"].apply(lambda x: f"{x:.2%}")
display["Lift"]       = display["Lift"].apply(lambda x: f"{x:.2f}×")
display.index = range(1, len(display) + 1)

st.dataframe(display, use_container_width=True, height=420)


# ── Charts ─────────────────────────────────────────────────────────────────────
theme.section("Rule landscape", "Strongest patterns and how they're distributed")

c1, c2 = st.columns(2, gap="medium")

with c1:
    top_lift = filtered.sort_values("lift", ascending=False).head(15)
    labels = [
        f"{r['antecedents_str'][:20]}… → {r['consequents_str'][:20]}…"
        if len(r["antecedents_str"]) + len(r["consequents_str"]) > 40
        else f"{r['antecedents_str']} → {r['consequents_str']}"
        for _, r in top_lift.iterrows()
    ]
    fig_bar = go.Figure(go.Bar(
        x=top_lift["lift"].tolist(),
        y=labels,
        orientation="h",
        marker=dict(
            color=top_lift["lift"].tolist(),
            colorscale=[[0, "#cffafe"], [1, theme.ACCENT]],
        ),
    ))
    fig_bar.update_layout(
        title="Top 15 rules by lift",
        xaxis_title="Lift",
        yaxis={"categoryorder": "total ascending"},
    )
    theme.style_fig(fig_bar, height=420)
    with st.container(border=True):
        st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    fig_sc = px.scatter(
        filtered.head(2000),
        x="support", y="confidence",
        size="lift", color="lift",
        color_continuous_scale=theme.TEAL_SCALE,
        hover_data={"antecedents_str": True, "consequents_str": True,
                    "support": ":.4f", "confidence": ":.2%", "lift": ":.2f"},
        labels={"support": "Support", "confidence": "Confidence"},
        title="Support vs confidence (bubble = lift)",
        opacity=0.6,
    )
    theme.style_fig(fig_sc, height=420)
    with st.container(border=True):
        st.plotly_chart(fig_sc, use_container_width=True)

c3, c4 = st.columns(2, gap="medium")
with c3:
    fig_hist_conf = px.histogram(
        filtered, x="confidence", nbins=30,
        title="Confidence distribution",
        labels={"confidence": "Confidence", "count": "Rules"},
        color_discrete_sequence=["#0891b2"],
    )
    theme.style_fig(fig_hist_conf, height=300)
    with st.container(border=True):
        st.plotly_chart(fig_hist_conf, use_container_width=True)

with c4:
    fig_hist_lift = px.histogram(
        filtered, x="lift", nbins=30,
        title="Lift distribution",
        labels={"lift": "Lift", "count": "Rules"},
        color_discrete_sequence=[theme.ACCENT],
    )
    theme.style_fig(fig_hist_lift, height=300)
    with st.container(border=True):
        st.plotly_chart(fig_hist_lift, use_container_width=True)
