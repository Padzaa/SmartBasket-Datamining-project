"""
SmartBasket AI — Landing Page

Entry point for the Streamlit multipage app.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="SmartBasket AI",
    page_icon="🧺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE / "src"))
import theme  # noqa: E402

theme.apply_theme()


@st.cache_data(show_spinner=False)
def _load_stats():
    try:
        df = pd.read_csv(BASE / "data" / "processed" / "transactions_clean.csv")
        rules = pd.read_csv(BASE / "models" / "association_rules.csv")
        return {
            "transactions": df["BillNo"].nunique(),
            "products": df["Itemname"].nunique(),
            "customers": df["CustomerID"].nunique(),
            "rules": len(rules),
        }
    except Exception:
        return {}


stats = _load_stats()

# ── Hero ─────────────────────────────────────────────────────────────────────
theme.masthead(
    eyebrow="Market Basket Intelligence",
    title="SmartBasket AI",
    subtitle="A hybrid recommendation and market-basket analytics platform — "
             "FP-Growth rules, ALS collaborative filtering and margin-aware "
             "ranking, in one explainable workspace.",
)

# ── KPI strip (main area) ──────────────────────────────────────────────────--
if stats:
    theme.kpi_row([
        {"label": "Transactions", "value": f"{stats['transactions']:,}", "sub": "unique baskets"},
        {"label": "Products", "value": f"{stats['products']:,}", "sub": "in catalogue", "accent": theme.INDIGO},
        {"label": "Customers", "value": f"{stats['customers']:,}", "sub": "across 30 countries"},
        {"label": "Assoc. rules", "value": f"{stats['rules']:,}", "sub": "mined by FP-Growth", "accent": theme.INDIGO},
    ])
else:
    st.info("Run the data pipeline to populate dataset statistics.")

# ── Sidebar ────────────────────────────────────────────────────────────────--
with st.sidebar:
    st.markdown("##### Model status")
    als_path = BASE / "models" / "als_model.pkl"
    if als_path.exists():
        st.success("ALS model — trained ✓")
    else:
        st.warning("ALS model — not trained")
        st.caption("Run `python scripts/train_als.py` to train.")

    st.markdown("##### Pipeline")
    st.markdown(
        '<div class="sb-side-stat"><span class="s-lbl">FP-Growth rules</span>'
        f'<span class="s-val">{stats.get("rules", 0):,}</span></div>'
        '<div class="sb-side-stat"><span class="s-lbl">Catalogue</span>'
        f'<span class="s-val">{stats.get("products", 0):,}</span></div>'
        '<div class="sb-side-stat"><span class="s-lbl">Customers</span>'
        f'<span class="s-val">{stats.get("customers", 0):,}</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("SmartBasket AI · Data Mining Project")

# ── Navigation grid ──────────────────────────────────────────────────────────
theme.section("Explore the workspace", "Four lenses on the same dataset")

c1, c2 = st.columns(2, gap="medium")
with c1:
    st.markdown(theme.nav_card(
        "📈", "Executive Overview",
        "KPIs, transaction trends, top products, revenue by category and rule-quality summaries.",
    ), unsafe_allow_html=True)
    st.markdown(theme.nav_card(
        "🔗", "Network Graph",
        "Interactive association-rule graph with lift-based edge colouring and product highlighting.",
    ), unsafe_allow_html=True)
with c2:
    st.markdown(theme.nav_card(
        "🎯", "Recommendation Engine",
        "Hybrid FP-Growth + ALS + margin recommender with MMR diversity and plain-language explanations.",
    ), unsafe_allow_html=True)
    st.markdown(theme.nav_card(
        "🔍", "Rules Explorer",
        "Browse and filter every association rule interactively, with live charts and CSV export.",
    ), unsafe_allow_html=True)

st.caption("Use the sidebar to switch between pages.")

# ── About ──────────────────────────────────────────────────────────────────--
theme.section("Under the hood")

a1, a2 = st.columns([1.3, 1], gap="large")
with a1:
    st.markdown(
        "**SmartBasket AI** is an end-to-end market-basket intelligence platform built on the "
        "Online Retail dataset (UCI, 2011). It combines four techniques into a single "
        "explainable ranking pipeline:"
    )
    st.markdown(
        "- **FP-Growth** association-rule mining (mlxtend)\n"
        "- **Alternating Least Squares** collaborative filtering (implicit)\n"
        "- **Maximal Marginal Relevance** diversity filtering\n"
        "- **Business-margin–weighted** hybrid ranking"
    )
with a2:
    theme.chips([
        "≈512K transactions",
        "19,389 baskets",
        "3,993 products",
        "4,292 customers",
        "30 countries",
    ])
