"""
ALS Training Script

Run this once to train the ALS collaborative filtering model and save it to
models/als_model.pkl. Re-run whenever new transaction data is available.

Usage:
    python scripts/train_als.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from als_recommender import ALSRecommender

TRANSACTIONS_PATH = Path(__file__).parent.parent / "data" / "processed" / "transactions_clean.csv"
MODEL_OUTPUT_PATH = Path(__file__).parent.parent / "models" / "als_model.pkl"


def main():
    print("=" * 60)
    print("SmartBasket AI — ALS Collaborative Filter Training")
    print("=" * 60)

    print(f"\nLoading transactions from:\n  {TRANSACTIONS_PATH}")
    df = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["Date"])
    print(f"Loaded {len(df):,} rows")

    valid = df.dropna(subset=["CustomerID"])
    print(f"Rows with CustomerID: {len(valid):,}")
    print(f"Unique customers:     {valid['CustomerID'].nunique():,}")
    print(f"Unique products:      {valid['Itemname'].nunique():,}")

    print("\nTraining ALS model...")
    rec = ALSRecommender()
    rec.fit(
        valid,
        factors=64,
        iterations=20,
        regularization=0.01,
        alpha=40.0,
    )

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec.save(str(MODEL_OUTPUT_PATH))

    # Quick sanity check
    print("\nSanity check — item-based recommendations for 'JUMBO BAG RED RETROSPOT':")
    sample = rec.recommend_for_items(["JUMBO BAG RED RETROSPOT"], n=5)
    for product, score in list(sample.items())[:5]:
        print(f"  {product}  ({score:.4f})")

    print("\nDone. Model saved to:", MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()
