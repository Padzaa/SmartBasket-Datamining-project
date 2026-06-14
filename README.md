<div align="center">

# 🧺 SmartBasket AI

### Market-Basket Intelligence & Hybrid Recommendation Platform

*FP-Growth association rules · ALS collaborative filtering · margin-aware ranking — in one explainable workspace.*

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.47-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![mlxtend](https://img.shields.io/badge/mlxtend-FP--Growth-0d9488)](http://rasbt.github.io/mlxtend/)
[![implicit](https://img.shields.io/badge/implicit-ALS-6366f1)](https://github.com/benfred/implicit)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#-license)

</div>

---

## 📖 Overview

**SmartBasket AI** is an end-to-end market-basket analytics platform built on the
[UCI *Online Retail* dataset](https://archive.ics.uci.edu/dataset/352/online+retail) — one year of
transactions from a UK-based online gift retailer. It turns raw point-of-sale data into
**explainable, business-aware product recommendations**.

Unlike a black-box recommender, every suggestion comes with the **rule, lift and confidence** behind
it, and you can see *how each engine voted* for each product.

The platform combines **four techniques** into a single ranking pipeline:

| Technique | Library | Role |
|---|---|---|
| **FP-Growth** association-rule mining | `mlxtend` | Discovers "customers who buy X also buy Y" patterns |
| **Alternating Least Squares (ALS)** | `implicit` | Collaborative filtering on implicit purchase feedback |
| **Maximal Marginal Relevance (MMR)** | custom | Removes near-duplicate products for a diverse result set |
| **Business-margin weighting** | custom | Tilts ranking toward profitable products |

---

## ✨ Features

- 🎯 **Hybrid recommender** — blends FP-Growth, ALS and margin signals with tunable weights.
- 🗣️ **Explainable output** — plain-language reason for every recommendation, citing the supporting rule.
- 🎚️ **Live tuning** — adjust signal weights, result count and MMR diversity from the sidebar.
- 👤 **Personalisation** — optionally condition recommendations on a specific customer's history (ALS).
- 🔍 **Rules Explorer** — browse, filter, search and export every mined association rule.
- 🔗 **Interactive network graph** — products as nodes, rules as edges, coloured by lift strength.
- 📈 **Executive dashboard** — KPIs, transaction trends, category mix, revenue and rule-quality charts.
- 📤 **CSV export** everywhere — recommendations and filtered rules.

---

## 📊 Dataset at a glance

| Metric | Value |
|---|---|
| Total records | **512,133** |
| Unique transactions (baskets) | **19,389** |
| Unique products | **3,993** |
| Unique customers | **4,292** |
| Countries | **30** |
| Date range | **2010-12-01 → 2011-12-09** |

> Raw data is **not** committed (see `.gitignore`); the repo ships with the processed
> `transactions_clean.csv` and pre-computed model artifacts so the app runs out of the box.

---

## 🚀 Quick start

### Prerequisites
- Python **3.13** (a `.python-version` is provided for `pyenv` / Streamlit Cloud)

### Installation

```bash
# 1. Clone
git clone <repo-url>
cd market-basket-analytics-master

# 2. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run Home.py
```

The app opens at **http://localhost:8501**.

> 💡 **ALS model.** The repo includes a pre-trained `models/als_model.pkl`. If it is missing
> (the sidebar will say *"ALS model — not trained"*), regenerate it with:
> ```bash
> python scripts/train_als.py
> ```

---

## 🧭 The four pages

| Page | Icon | What it does |
|---|---|---|
| **Executive Overview** | 📈 | KPIs, weekly transaction trend, top products, category & revenue breakdowns, rule-quality charts. |
| **Recommendation Engine** | 🎯 | Build a basket → get hybrid, margin-aware, explained recommendations with per-signal contribution chart. |
| **Rules Explorer** | 🔍 | Filterable/searchable table of all association rules + lift/support/confidence charts + CSV export. |
| **Network Graph** | 🔗 | Interactive NetworkX + Plotly graph of rules; edge colour = lift tier, node size = connectivity. |

---

## 📁 Project structure

```
market-basket-analytics-master/
├── Home.py                       # Streamlit landing page (entry point)
├── pages/                        # Multipage app
│   ├── 1_Executive_Overview.py
│   ├── 2_Recommendation_Engine.py
│   ├── 3_Rules_Explorer.py
│   └── 4_Network_Graph.py
├── src/                          # Core library
│   ├── data_preprocessing.py     # Cleaning & feature engineering
│   ├── association_rules.py      # Apriori / FP-Growth mining
│   ├── rule_filtering.py         # Quality thresholds & dedup
│   ├── recommendations.py        # Rule-based recommendation engine
│   ├── als_recommender.py        # ALS collaborative filtering
│   ├── business_margin.py        # Margin scoring
│   ├── mmr.py                    # MMR diversity filter
│   ├── hybrid_recommender.py     # Combines FP + ALS + margin
│   ├── explanations.py           # Natural-language explanations
│   ├── segmentation.py           # RFM + clustering customer segmentation
│   └── theme.py                  # Shared UI theme & components
├── scripts/                      # One-off utilities
│   ├── train_als.py              # Train & save the ALS model
│   ├── test_recommendations.py   # Sanity-check the recommender
│   ├── diagnose_products.py      # Debug product-name matching
│   └── generate_social_media_plots.py
├── data/
│   ├── processed/                # transactions_clean.csv, data_summary.txt
│   ├── transactions/             # basket_matrix.csv
│   └── product_margins.csv
├── models/                       # Pre-computed artifacts
│   ├── association_rules.csv     # All mined rules
│   ├── frequent_itemsets.csv
│   ├── top_rules_by_{lift,confidence,support}.csv
│   ├── als_model.pkl             # Trained ALS model
│   ├── customer_rfm.csv          # RFM segmentation output
│   ├── segment_{summary,recommendations}.csv
│   └── clustering_metrics.txt
├── .streamlit/config.toml        # Theme & server config
├── requirements.txt
└── setup.py
```

---

## 🔬 How the hybrid recommender works

```
                ┌──────────────────┐
   basket  ───► │   FP-Growth      │──► association-rule score
                ├──────────────────┤
   customer ──► │   ALS            │──► collaborative-filtering score
                ├──────────────────┤
   catalogue ─► │   Business margin│──► profitability score
                └────────┬─────────┘
                         │  min-max normalise each signal
                         ▼
              weighted sum  (default 0.4 / 0.4 / 0.2)
                         │
                         ▼
              MMR diversity filter (λ relevance ↔ diversity)
                         │
                         ▼
              ranked + explained recommendations
```

- **Score normalisation** — each signal is min-max normalised to `[0, 1]` before combining.
- **Graceful degradation** — if a signal is unavailable (e.g. unknown customer), its weight is
  redistributed proportionally across the remaining signals.
- **Rule-quality defaults** — `min_support = 0.001`, `min_confidence = 0.20`, `min_lift = 1.20`
  (see `src/rule_filtering.py`).
- **MMR similarity** — token-level Jaccard over product names (Carbonell & Goldstein, 1998).
- **Fallback** — if no rule-based recommendations exist for a basket, the app shows popular products instead.

---

## 🛠️ Tech stack

**Data & ML:** pandas · numpy · scikit-learn · mlxtend (FP-Growth) · implicit (ALS)
**Visualisation:** Plotly · matplotlib · seaborn · NetworkX
**App:** Streamlit (multipage)

Pinned versions are in [`requirements.txt`](requirements.txt).

---

## 🔄 Reproducing the model artifacts

The `models/` and `data/processed/` artifacts are committed, so you don't need raw data to run the app.
To regenerate from scratch (raw data required in `data/raw/`):

```bash
python scripts/train_als.py          # retrain the ALS collaborative-filtering model
```

The preprocessing, rule-mining and segmentation classes in `src/` can be used directly from a script or
notebook to rebuild `transactions_clean.csv`, `association_rules.csv` and the RFM segments.

---

## 📜 License

Released under the **MIT License**. The *Online Retail* dataset is © its original authors (UCI ML Repository).

---

<div align="center">

*SmartBasket AI · Data Mining Project*

</div>
