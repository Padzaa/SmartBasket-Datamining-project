"""
ALS Collaborative Filtering Recommender

Uses the `implicit` library's AlternatingLeastSquares model trained on
purchase-count implicit feedback data.

Two recommendation modes:
  - recommend_for_user(user_id)  — personalised for a known customer
  - recommend_for_items(basket)  — item-based similarity (no customer ID needed)
"""

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from pathlib import Path
from typing import Dict, List, Optional


class ALSRecommender:
    def __init__(self):
        self.model = None
        self.user_item: Optional[sparse.csr_matrix] = None  # (n_users, n_items)
        self.item_ids: List[str] = []
        self.user_ids: List[int] = []
        self.item_to_idx: Dict[str, int] = {}
        self.user_to_idx: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        transactions_df: pd.DataFrame,
        factors: int = 64,
        iterations: int = 20,
        regularization: float = 0.01,
        alpha: float = 40.0,
    ) -> "ALSRecommender":
        """
        Build user-item matrix and train ALS.

        Args:
            transactions_df: Cleaned transactions with CustomerID, Itemname, Quantity.
            factors: Number of latent factors.
            iterations: ALS training iterations.
            regularization: L2 regularisation.
            alpha: Confidence scaling factor for implicit feedback.
        """
        from implicit.als import AlternatingLeastSquares

        df = transactions_df.dropna(subset=["CustomerID"]).copy()
        df["CustomerID"] = df["CustomerID"].astype(int)
        df["Itemname"] = df["Itemname"].str.upper().str.strip()

        # Build index mappings
        self.user_ids = sorted(df["CustomerID"].unique().tolist())
        self.item_ids = sorted(df["Itemname"].unique().tolist())
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item_to_idx = {it: i for i, it in enumerate(self.item_ids)}

        # Aggregate purchase counts per (customer, item)
        agg = df.groupby(["CustomerID", "Itemname"])["Quantity"].sum().reset_index()

        rows = [self.user_to_idx[r] for r in agg["CustomerID"]]
        cols = [self.item_to_idx[it] for it in agg["Itemname"]]
        data = agg["Quantity"].values.astype(np.float32)

        n_users, n_items = len(self.user_ids), len(self.item_ids)
        self.user_item = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

        self.model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            alpha=alpha,
            use_gpu=False,
            random_state=42,
        )
        # implicit >= 0.5 expects user_items (users × items)
        self.model.fit(self.user_item, show_progress=True)
        print(f"ALS trained: {n_users} users × {n_items} items, {factors} factors")
        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        payload = {
            "model": self.model,
            "user_item": self.user_item,
            "item_ids": self.item_ids,
            "user_ids": self.user_ids,
            "item_to_idx": self.item_to_idx,
            "user_to_idx": self.user_to_idx,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"ALS model saved → {path}")

    def load(self, path: str) -> "ALSRecommender":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.user_item = payload["user_item"]
        self.item_ids = payload["item_ids"]
        self.user_ids = payload["user_ids"]
        self.item_to_idx = payload["item_to_idx"]
        self.user_to_idx = payload["user_to_idx"]
        print(f"ALS model loaded from {path} ({len(self.user_ids)} users, {len(self.item_ids)} items)")
        return self

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def recommend_for_user(
        self,
        user_id: int,
        n: int = 20,
        exclude: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Return top-n product recommendations for a known customer.
        Returns empty dict if user_id not in training data.
        """
        if not self.is_trained:
            return {}
        uid_int = int(user_id)
        if uid_int not in self.user_to_idx:
            return {}

        user_idx = self.user_to_idx[uid_int]
        user_row = self.user_item[user_idx]
        ids, scores = self.model.recommend(user_idx, user_row, N=n + 20, filter_already_liked_items=True)

        exclude_upper = {e.upper().strip() for e in (exclude or [])}
        result: Dict[str, float] = {}
        for item_idx, score in zip(ids, scores):
            name = self.item_ids[int(item_idx)]
            if name not in exclude_upper and len(result) < n:
                result[name] = float(score)
        return result

    def recommend_for_items(
        self,
        basket_items: List[str],
        n: int = 30,
    ) -> Dict[str, float]:
        """
        Return product scores using item-based similarity (no customer ID needed).
        For each basket item, retrieves similar items and aggregates scores.
        """
        if not self.is_trained:
            return {}

        basket_upper = {it.upper().strip() for it in basket_items}
        aggregated: Dict[str, float] = {}

        for item in basket_items:
            item_upper = item.upper().strip()
            if item_upper not in self.item_to_idx:
                continue
            item_idx = self.item_to_idx[item_upper]
            ids, scores = self.model.similar_items(item_idx, N=n + 1)
            for sim_idx, sim_score in zip(ids, scores):
                idx = int(sim_idx)
                if idx < 0 or idx >= len(self.item_ids):
                    continue
                sim_name = self.item_ids[idx]
                if sim_name not in basket_upper:
                    aggregated[sim_name] = aggregated.get(sim_name, 0.0) + float(sim_score)

        return aggregated
