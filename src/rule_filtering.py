"""
Rule Filtering Module

Removes low-quality, duplicate, and weak association rules.
Default thresholds are tuned for the Online Retail dataset.
"""

import pandas as pd
from pathlib import Path


MIN_SUPPORT = 0.001
MIN_CONFIDENCE = 0.20
MIN_LIFT = 1.20


def filter_rules(
    rules_df: pd.DataFrame,
    min_support: float = MIN_SUPPORT,
    min_confidence: float = MIN_CONFIDENCE,
    min_lift: float = MIN_LIFT,
) -> pd.DataFrame:
    """
    Apply quality thresholds and remove duplicate rules.

    Args:
        rules_df: Raw association rules DataFrame (from models/association_rules.csv).
        min_support: Minimum support threshold.
        min_confidence: Minimum confidence threshold.
        min_lift: Minimum lift threshold.

    Returns:
        Filtered DataFrame sorted by lift descending.
    """
    df = rules_df.copy()
    before = len(df)

    df = df[
        (df["support"] >= min_support)
        & (df["confidence"] >= min_confidence)
        & (df["lift"] >= min_lift)
    ]

    df = _remove_duplicates(df)
    df = df.sort_values("lift", ascending=False).reset_index(drop=True)

    print(
        f"Rule filtering: {before:,} → {len(df):,} rules "
        f"(removed {before - len(df):,})"
    )
    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate antecedent/consequent pairs (same direction)."""
    if "antecedents_str" in df.columns and "consequents_str" in df.columns:
        return df.drop_duplicates(subset=["antecedents_str", "consequents_str"])
    return df.drop_duplicates()


def load_and_filter(
    rules_path: str,
    min_support: float = MIN_SUPPORT,
    min_confidence: float = MIN_CONFIDENCE,
    min_lift: float = MIN_LIFT,
) -> pd.DataFrame:
    """
    Convenience: load rules from CSV then filter.

    Returns the filtered DataFrame ready for the recommendation pipeline.
    """
    df = pd.read_csv(rules_path)
    return filter_rules(df, min_support, min_confidence, min_lift)


if __name__ == "__main__":
    rules_path = Path(__file__).parent.parent / "models" / "association_rules.csv"
    filtered = load_and_filter(str(rules_path))
    print(f"\nFiltered rules shape: {filtered.shape}")
    print(filtered[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].head(10))
