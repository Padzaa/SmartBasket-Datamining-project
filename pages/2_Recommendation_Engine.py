"""
Page 2 — Recommendation Engine (main demo page)

Hybrid FP-Growth + ALS + Margin recommender with MMR diversity
and explainable outputs.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "src"))

from recommendations import RecommendationEngine
from als_recommender import ALSRecommender
from business_margin import load_margins
from hybrid_recommender import HybridRecommender
from rule_filtering import load_and_filter
from explanations import ExplanationGenerator
from mmr import diversify
import theme

st.set_page_config(page_title="Recommendations · SmartBasket AI", page_icon="🎯", layout="wide")
theme.apply_theme()


# ── Load all models ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_transactions():
    return pd.read_csv(BASE / "data" / "processed" / "transactions_clean.csv", parse_dates=["Date"])

@st.cache_data(show_spinner=False)
def _load_filtered_rules():
    return load_and_filter(str(BASE / "models" / "association_rules.csv"))

@st.cache_resource(show_spinner=False)
def _load_fp_engine(rules_path: str):
    eng = RecommendationEngine()
    eng.load_rules(rules_path)
    return eng

@st.cache_resource(show_spinner=False)
def _load_als():
    rec = ALSRecommender()
    p = BASE / "models" / "als_model.pkl"
    if p.exists():
        rec.load(str(p))
    return rec

@st.cache_resource(show_spinner=False)
def _load_margins():
    return load_margins(str(BASE / "data" / "product_margins.csv"))

@st.cache_resource(show_spinner=False)
def _load_explainer():
    # Use all rules for explanation lookup — weaker rules still produce valid citation text.
    # Quality filtering is for recommendations, not for explanations.
    raw = pd.read_csv(BASE / "models" / "association_rules.csv")
    return ExplanationGenerator(raw)


with st.spinner("Loading models…"):
    df = _load_transactions()
    rules_df = _load_filtered_rules()
    fp_engine = _load_fp_engine(str(BASE / "models" / "association_rules.csv"))
    als_engine = _load_als()
    margin_engine = _load_margins()
    explainer = _load_explainer()

hybrid = HybridRecommender(fp_engine, als_engine, margin_engine)

all_products = sorted(df["Itemname"].unique().tolist())
all_customers = sorted(df["CustomerID"].dropna().astype(int).unique().tolist())


# ── Masthead ────────────────────────────────────────────────────────────────--
theme.masthead(
    eyebrow="Hybrid Recommender",
    title="Recommendation Engine",
    subtitle="Build a basket and get hybrid, margin-aware recommendations — each one "
             "explained with the rule, lift and confidence behind it.",
)

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##### Output")
    n_recs = st.slider("Number of recommendations", 3, 15, 8)
    use_mmr = st.toggle("Apply MMR diversity filter", value=True,
                        help="Removes highly similar products from the results")
    mmr_lambda = st.slider("MMR λ (relevance vs diversity)", 0.3, 1.0, 0.7, 0.05,
                            disabled=not use_mmr,
                            help="1.0 = pure relevance, 0.0 = pure diversity")

    st.markdown("##### Signal weights")
    fp_w = st.slider("FP-Growth weight", 0.0, 1.0, 0.4, 0.05)
    als_w = st.slider("ALS weight", 0.0, 1.0, 0.4, 0.05)
    margin_w = st.slider("Margin weight", 0.0, 1.0, 0.2, 0.05)
    total_w = fp_w + als_w + margin_w
    if abs(total_w - 1.0) > 0.01:
        st.warning(f"Weights sum to {total_w:.2f} (ideally 1.0)")

    st.markdown("---")
    if not als_engine.is_trained:
        st.warning("ALS model not trained. Run `python scripts/train_als.py`.")
    else:
        st.success(f"ALS: {len(als_engine.user_ids):,} users · {len(als_engine.item_ids):,} items")


# ── Input panel ────────────────────────────────────────────────────────────────
col_input, col_results = st.columns([1, 1.6], gap="large")

with col_input:
    with st.container(border=True):
        st.markdown("##### 🛒 Build a basket")

        selected_products = st.multiselect(
            "Choose products",
            options=all_products,
            placeholder="Start typing a product name…",
            help="Select one or more products to see what else customers tend to buy.",
            label_visibility="collapsed",
        )

        use_customer = st.toggle("Personalise by customer", value=False,
                                 help="If enabled, the ALS model uses that customer's history")
        customer_id = None
        if use_customer:
            customer_id = st.selectbox(
                "Customer ID", options=all_customers,
                format_func=lambda x: f"Customer #{x}",
            )

        generate = st.button("✨ Generate Recommendations", type="primary",
                             disabled=len(selected_products) == 0,
                             use_container_width=True)

    if selected_products:
        st.markdown("**Current basket**")
        st.markdown(
            "".join(f'<span class="meta-pill">{p.title()}</span>' for p in selected_products),
            unsafe_allow_html=True,
        )


# ── Results panel ─────────────────────────────────────────────────────────────
with col_results:
    if not selected_products:
        st.info("👈 Select products and click **Generate Recommendations**.")
        # Clear stale cache when basket is emptied
        for k in ["last_basket", "last_ranked", "last_explanations"]:
            st.session_state.pop(k, None)

    elif not generate and "last_ranked" not in st.session_state:
        st.info("👆 Click **Generate Recommendations** to see results.")

    elif generate or "last_basket" in st.session_state:
        # Only re-run when the button is clicked; otherwise show cached results
        run_now = generate

        if run_now:
            # Normalise weights so they always sum to 1
            _total_w = fp_w + als_w + margin_w
            if _total_w > 0:
                fp_w_n = fp_w / _total_w
                als_w_n = als_w / _total_w
                mar_w_n = margin_w / _total_w
            else:
                fp_w_n = als_w_n = mar_w_n = 1 / 3

            with st.spinner("Running hybrid recommender…"):
                ranked = hybrid.rank(
                    basket=selected_products,
                    customer_id=customer_id,
                    n=max(n_recs * 3, 30),
                    fp_weight=fp_w_n,
                    als_weight=als_w_n,
                    margin_weight=mar_w_n,
                )
            if use_mmr and not ranked.empty:
                ranked = diversify(ranked, n=n_recs, lambda_val=mmr_lambda)
            else:
                ranked = ranked.head(n_recs)

            explanations = []
            if not ranked.empty:
                explanations = explainer.generate_batch(selected_products, ranked)

            st.session_state["last_basket"] = selected_products
            st.session_state["last_ranked"] = ranked
            st.session_state["last_explanations"] = explanations
        else:
            ranked = st.session_state.get("last_ranked", pd.DataFrame())
            explanations = st.session_state.get("last_explanations", [])

        if ranked is None or ranked.empty:
            # Fallback: popular products
            st.markdown('<div class="no-rec-box">No rule-based recommendations found for this basket. Showing popular products instead.</div>', unsafe_allow_html=True)
            popular = (
                df[~df["Itemname"].isin(selected_products)]
                ["Itemname"].value_counts().head(n_recs).reset_index()
            )
            popular.columns = ["Product", "Purchases"]
            for _, row in popular.iterrows():
                st.markdown(f'<div class="fallback-item">📦 <b>{row["Product"].title()}</b> — {row["Purchases"]:,} purchases</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"**{len(ranked)} recommendations** — {'diversity-filtered (MMR)' if use_mmr else 'ranked by hybrid score'}")

            for rank, (_, row) in enumerate(ranked.iterrows(), 1):
                explanation = explanations[rank - 1] if rank - 1 < len(explanations) else ""
                score_pct = int(row["final_score"] * 100)
                conf_pct  = f"{row['confidence']:.0%}" if row["confidence"] > 0 else "—"
                lift_val  = f"{row['lift']:.2f}×" if row["lift"] > 0 else "—"

                st.markdown(f"""
                <div class="rec-card">
                  <span class="rank-badge">#{rank}</span>
                  <h4>{row['product'].title()}</h4>
                  <div class="explanation">{explanation}</div>
                  <div class="score-bar"><div class="score-fill" style="width:{score_pct}%"></div></div>
                  <span class="meta-pill">Score {row['final_score']:.3f}</span>
                  <span class="meta-pill">Confidence {conf_pct}</span>
                  <span class="meta-pill">Lift {lift_val}</span>
                </div>
                """, unsafe_allow_html=True)

            # Signal contribution chart
            if len(ranked) > 0:
                theme.section("Signal contribution", "How each engine voted, per recommendation")

                fig = go.Figure()
                products_display = [r["product"].title()[:28] for _, r in ranked.iterrows()]
                for signal, color, label in [
                    ("fp_score",     theme.ACCENT,  "FP-Growth"),
                    ("als_score",    "#0891b2",     "ALS"),
                    ("margin_score", theme.INDIGO,  "Margin"),
                ]:
                    fig.add_trace(go.Bar(
                        name=label,
                        x=products_display,
                        y=ranked[signal].tolist(),
                        marker_color=color,
                    ))
                fig.update_layout(
                    barmode="stack",
                    legend=dict(orientation="h", y=-0.45),
                    xaxis_tickangle=-40,
                    yaxis_title="Normalised score",
                )
                theme.style_fig(fig, height=320)
                fig.update_layout(margin=dict(t=10, b=90, l=8, r=8))
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)

                # Compact table + CSV export
                with st.expander("📋 View as table / export"):
                    export_df = ranked[["product", "final_score", "confidence", "lift", "support", "based_on"]].copy()
                    export_df.columns = ["Product", "Score", "Confidence", "Lift", "Support", "Based On"]
                    export_df["Score"]      = export_df["Score"].round(4)
                    export_df["Confidence"] = export_df["Confidence"].apply(lambda x: f"{x:.1%}" if x > 0 else "—")
                    export_df["Lift"]       = export_df["Lift"].apply(lambda x: f"{x:.2f}×" if x > 0 else "—")
                    export_df["Support"]    = export_df["Support"].apply(lambda x: f"{x:.4f}" if x > 0 else "—")
                    export_df.index = range(1, len(export_df) + 1)
                    st.dataframe(export_df, use_container_width=True)

                    raw = ranked.copy()
                    raw["basket"] = ", ".join(selected_products)
                    st.download_button(
                        "⬇ Export recommendations as CSV",
                        data=raw.to_csv(index=False),
                        file_name="recommendations.csv",
                        mime="text/csv",
                    )
