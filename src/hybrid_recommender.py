"""
Hybrid Recommendation Engine

Combines three signals into a single ranked recommendation list:
  - FP-Growth association rules  (weight 0.4)
  - ALS collaborative filtering  (weight 0.4)
  - Business margin score        (weight 0.2)

Scores are min-max normalised before combining.
If a signal is unavailable the remaining weights are redistributed proportionally.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


FP_WEIGHT = 0.4
ALS_WEIGHT = 0.4
MARGIN_WEIGHT = 0.2


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize a score dict to [0, 1]. Returns {} for empty input."""
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: float((v - lo) / (hi - lo)) for k, v in scores.items()}


class HybridRecommender:
    """
    Fuses FP-Growth, ALS and margin signals into a ranked DataFrame.

    Parameters
    ----------
    fp_engine   : RecommendationEngine (src/recommendations.py)
    als_engine  : ALSRecommender (src/als_recommender.py)
    margin_engine : BusinessMargin (src/business_margin.py)
    """

    def __init__(self, fp_engine, als_engine, margin_engine):
        self.fp = fp_engine
        self.als = als_engine
        self.margin = margin_engine

    def rank(
        self,
        basket: List[str],
        customer_id: Optional[int] = None,
        n: int = 30,
        fp_weight: float = FP_WEIGHT,
        als_weight: float = ALS_WEIGHT,
        margin_weight: float = MARGIN_WEIGHT,
    ) -> pd.DataFrame:
        """
        Produce a ranked recommendation DataFrame.

        Columns returned:
            product, final_score, fp_score, als_score, margin_score,
            confidence, lift, support, based_on
        """
        # --- FP-Growth signal ---
        fp_df = self.fp.recommend_products(basket, n=60, min_confidence=0.05)
        fp_scores_raw: Dict[str, float] = {}
        fp_meta: Dict[str, dict] = {}
        if fp_df is not None and not fp_df.empty:
            for _, row in fp_df.iterrows():
                name = row["Product"]
                fp_scores_raw[name] = float(row.get("Score", row.get("Confidence", 0)))
                fp_meta[name] = {
                    "confidence": row.get("Confidence", 0),
                    "lift": row.get("Lift", 0),
                    "support": row.get("Support", 0),
                    "based_on": row.get("Based_On", ""),
                }

        # --- ALS signal ---
        als_scores_raw: Dict[str, float] = {}
        if self.als.is_trained:
            if customer_id is not None:
                als_scores_raw = self.als.recommend_for_user(
                    customer_id, n=60, exclude=basket
                )
            else:
                als_scores_raw = self.als.recommend_for_items(basket, n=60)

        # --- Candidate pool ---
        all_candidates = set(fp_scores_raw.keys()) | set(als_scores_raw.keys())
        # Remove basket items from candidates
        basket_upper = {b.upper().strip() for b in basket}
        all_candidates = {c for c in all_candidates if c.upper().strip() not in basket_upper}

        if not all_candidates:
            return pd.DataFrame()

        # --- Margin signal ---
        margin_scores_raw: Dict[str, float] = {
            c: self.margin.margin_score(c) for c in all_candidates
        }

        # --- Normalise ---
        fp_norm = _normalize(fp_scores_raw)
        als_norm = _normalize(als_scores_raw)
        margin_norm = _normalize(margin_scores_raw)

        # --- Redistribute weights if ALS has no signal ---
        if not als_scores_raw:
            fp_w = fp_weight + als_weight * (fp_weight / (fp_weight + margin_weight))
            margin_w = margin_weight + als_weight * (margin_weight / (fp_weight + margin_weight))
            als_w = 0.0
        else:
            fp_w, als_w, margin_w = fp_weight, als_weight, margin_weight

        # --- Combine ---
        rows = []
        for candidate in all_candidates:
            fp_s = fp_norm.get(candidate, 0.0)
            als_s = als_norm.get(candidate, 0.0)
            mar_s = margin_norm.get(candidate, 0.0)
            final = fp_w * fp_s + als_w * als_s + margin_w * mar_s

            meta = fp_meta.get(candidate, {})
            rows.append(
                {
                    "product": candidate,
                    "final_score": round(final, 6),
                    "fp_score": round(fp_s, 6),
                    "als_score": round(als_s, 6),
                    "margin_score": round(mar_s, 6),
                    "confidence": meta.get("confidence", 0.0),
                    "lift": meta.get("lift", 0.0),
                    "support": meta.get("support", 0.0),
                    "based_on": meta.get("based_on", ""),
                }
            )

        result = (
            pd.DataFrame(rows)
            .sort_values("final_score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
        return result
