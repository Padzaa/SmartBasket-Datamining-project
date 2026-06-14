"""
Maximal Marginal Relevance (MMR) Diversity Filter

Selects a diverse subset from a ranked list of recommendations by
penalising candidates that are too similar to already-selected items.

Reference: Carbonell & Goldstein (1998)
"""

import pandas as pd
from typing import List, Dict


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two product name strings."""
    set_a = set(a.upper().split())
    set_b = set(b.upper().split())
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def diversify(
    candidates_df: pd.DataFrame,
    n: int = 10,
    lambda_val: float = 0.7,
    score_col: str = "final_score",
    product_col: str = "product",
) -> pd.DataFrame:
    """
    Apply MMR to diversify the candidates DataFrame.

    Args:
        candidates_df: DataFrame with at least `product_col` and `score_col`.
        n: Number of items to return.
        lambda_val: Trade-off between relevance (1.0) and diversity (0.0).
        score_col: Column name for the relevance score.
        product_col: Column name for the product name.

    Returns:
        DataFrame of size ≤ n, ordered by MMR selection sequence.
    """
    if candidates_df.empty:
        return candidates_df

    if len(candidates_df) <= n:
        return candidates_df.reset_index(drop=True)

    scores: Dict[str, float] = dict(
        zip(candidates_df[product_col], candidates_df[score_col])
    )
    remaining: List[str] = candidates_df[product_col].tolist()
    selected: List[str] = []

    while len(selected) < n and remaining:
        if not selected:
            best = max(remaining, key=lambda p: scores[p])
        else:
            best = None
            best_mmr = float("-inf")
            for p in remaining:
                relevance = scores[p]
                max_sim = max(_jaccard(p, s) for s in selected)
                mmr_score = lambda_val * relevance - (1.0 - lambda_val) * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best = p

        selected.append(best)
        remaining.remove(best)

    result = (
        candidates_df[candidates_df[product_col].isin(selected)]
        .set_index(product_col)
        .loc[selected]
        .reset_index()
    )
    return result
