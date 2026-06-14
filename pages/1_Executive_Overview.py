"""
Page 1 — Executive Overview

KPIs, top products, category breakdown, rule quality summary.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "src"))
from rule_filtering import load_and_filter
import theme

st.set_page_config(page_title="Executive Overview · SmartBasket AI", page_icon="📈", layout="wide")
theme.apply_theme()


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_transactions():
    return pd.read_csv(BASE / "data" / "processed" / "transactions_clean.csv", parse_dates=["Date"])

@st.cache_data(show_spinner=False)
def load_rules():
    return pd.read_csv(BASE / "models" / "association_rules.csv")

@st.cache_data(show_spinner=False)
def load_filtered_rules():
    return load_and_filter(str(BASE / "models" / "association_rules.csv"))


with st.spinner("Loading data…"):
    df = load_transactions()
    rules_raw = load_rules()
    rules_filtered = load_filtered_rules()


# ── Category helper ────────────────────────────────────────────────────────────
CATEGORY_MAP = {
    "BAG": "Bags & Storage",
    "LUNCH": "Bags & Storage",
    "HOT WATER BOTTLE": "Home & Comfort",
    "MUG": "Mugs & Drinkware",
    "CUP": "Mugs & Drinkware",
    "TEACUP": "Mugs & Drinkware",
    "CAKE": "Baking & Kitchen",
    "BAKING": "Baking & Kitchen",
    "KITCHEN": "Baking & Kitchen",
    "CHRISTMAS": "Seasonal & Gifts",
    "VINTAGE": "Seasonal & Gifts",
    "HEART": "Décor & Gifts",
    "DECORATION": "Décor & Gifts",
    "DOORMAT": "Décor & Gifts",
    "CANDLE": "Décor & Gifts",
    "T-LIGHT": "Décor & Gifts",
    "CARD": "Stationery",
    "NOTEBOOK": "Stationery",
    "PENCIL": "Stationery",
    "BOOK": "Stationery",
}

def categorise(name: str) -> str:
    n = name.upper()
    for kw, cat in CATEGORY_MAP.items():
        if kw in n:
            return cat
    return "Other"

df["Category"] = df["Itemname"].apply(categorise)


# ── Masthead ────────────────────────────────────────────────────────────────--
theme.masthead(
    eyebrow="Market Basket · Overview",
    title="Executive Overview",
    subtitle="High-level performance metrics and top-line insights drawn from the full transaction dataset.",
)

# ── KPI row ───────────────────────────────────────────────────────────────────
total_txn     = df["BillNo"].nunique()
total_prod    = df["Itemname"].nunique()
total_cust    = df["CustomerID"].nunique()
total_rules   = len(rules_raw)
total_recs    = rules_filtered["consequents_str"].nunique() if not rules_filtered.empty else 0

theme.kpi_row([
    {"label": "Transactions", "value": f"{total_txn:,}", "sub": "unique baskets"},
    {"label": "Unique products", "value": f"{total_prod:,}", "sub": "in catalogue", "accent": theme.INDIGO},
    {"label": "Customers", "value": f"{total_cust:,}", "sub": "identified buyers"},
    {"label": "Assoc. rules", "value": f"{total_rules:,}", "sub": "before filtering", "accent": theme.INDIGO},
    {"label": "Recommendable", "value": f"{total_recs:,}", "sub": "quality consequents"},
])


# ── Transaction trend ─────────────────────────────────────────────────────────
theme.section("Transaction momentum", "Unique baskets per week")

daily = (
    df.groupby(df["Date"].dt.to_period("W").dt.start_time)["BillNo"]
    .nunique()
    .reset_index()
)
daily.columns = ["Week", "Transactions"]

fig_trend = px.area(
    daily, x="Week", y="Transactions",
    labels={"Transactions": "Unique transactions", "Week": ""},
    color_discrete_sequence=[theme.ACCENT],
)
fig_trend.update_traces(
    line_color=theme.ACCENT, fillcolor="rgba(13,148,136,0.12)", line_width=2.5,
    hovertemplate="Week of %{x|%b %d, %Y}<br>%{y:,} transactions<extra></extra>",
)
theme.style_fig(fig_trend, height=280)
with st.container(border=True):
    st.plotly_chart(fig_trend, use_container_width=True)


# ── Product & category mix ─────────────────────────────────────────────────────
theme.section("Product & category mix")

col_top, col_cat = st.columns([1.5, 1], gap="medium")

with col_top:
    top15 = df["Itemname"].value_counts().head(15).reset_index()
    top15.columns = ["Product", "Count"]
    fig_top = px.bar(
        top15, x="Count", y="Product", orientation="h",
        color="Count", color_continuous_scale=theme.TEAL_SCALE,
        labels={"Count": "Purchases", "Product": ""},
        title="Top 15 products by purchase frequency",
    )
    fig_top.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
    fig_top.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} purchases<extra></extra>")
    theme.style_fig(fig_top, height=470)
    with st.container(border=True):
        st.plotly_chart(fig_top, use_container_width=True)

with col_cat:
    cat_counts = df.groupby("Category")["BillNo"].nunique().reset_index()
    cat_counts.columns = ["Category", "Transactions"]
    cat_counts = cat_counts.sort_values("Transactions", ascending=False)
    fig_cat = px.pie(
        cat_counts, values="Transactions", names="Category", hole=0.58,
        color_discrete_sequence=px.colors.sequential.Teal,
        title="Transactions by category",
    )
    fig_cat.update_traces(textposition="inside", textinfo="percent")
    fig_cat.update_layout(legend=dict(orientation="h", y=-0.15, font=dict(size=10)))
    theme.style_fig(fig_cat, height=470)
    with st.container(border=True):
        st.plotly_chart(fig_cat, use_container_width=True)


# ── Revenue ─────────────────────────────────────────────────────────────────--
theme.section("Revenue by category")

cat_rev = df.groupby("Category")["TransactionValue"].sum().reset_index()
cat_rev.columns = ["Category", "Revenue"]
cat_rev = cat_rev.sort_values("Revenue", ascending=False)
fig_rev = px.bar(
    cat_rev, x="Revenue", y="Category", orientation="h",
    color="Revenue", color_continuous_scale=["#cffafe", theme.INDIGO],
    labels={"Revenue": "Revenue ($)", "Category": ""},
    title="Total revenue per category",
)
fig_rev.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
theme.style_fig(fig_rev, height=340)
with st.container(border=True):
    st.plotly_chart(fig_rev, use_container_width=True)


# ── Rule quality ────────────────────────────────────────────────────────────--
theme.section("Association-rule quality", "Distribution and trade-off of filtered rules")

col1, col2 = st.columns(2, gap="medium")
with col1:
    fig_lift = px.histogram(
        rules_filtered, x="lift", nbins=40,
        title="Lift distribution",
        labels={"lift": "Lift", "count": "Rules"},
        color_discrete_sequence=[theme.ACCENT],
    )
    theme.style_fig(fig_lift, height=330)
    with st.container(border=True):
        st.plotly_chart(fig_lift, use_container_width=True)

with col2:
    fig_sc = px.scatter(
        rules_filtered.head(3000), x="support", y="confidence",
        size="lift", color="lift",
        color_continuous_scale=theme.TEAL_SCALE,
        title="Support vs confidence (size = lift)",
        labels={"support": "Support", "confidence": "Confidence"},
        opacity=0.6,
    )
    theme.style_fig(fig_sc, height=330)
    with st.container(border=True):
        st.plotly_chart(fig_sc, use_container_width=True)


# ── Quick insights ─────────────────────────────────────────────────────────--
theme.section("Quick insights")

avg_basket = df.groupby("BillNo")["Itemname"].count().mean()
avg_value  = df.groupby("BillNo")["TransactionValue"].sum().mean()
high_lift  = len(rules_filtered[rules_filtered["lift"] > 3.0])
top_country = df["Country"].value_counts().index[0]

theme.chips([
    f"📦 Avg basket <b>{avg_basket:.1f}</b> items",
    f"💰 Avg order <b>${avg_value:.2f}</b>",
    f"⚡ <b>{high_lift:,}</b> rules with lift &gt; 3×",
    f"🌍 Top market <b>{top_country}</b>",
    f"🗓️ <b>{df['Date'].dt.date.min()}</b> → <b>{df['Date'].dt.date.max()}</b>",
])
