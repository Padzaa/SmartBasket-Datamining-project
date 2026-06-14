"""
Business Margin Module

Loads product margin data and provides normalized margin scores
used during recommendation ranking.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List


DEFAULT_MARGIN = 0.40


class BusinessMargin:
    """
    Provides product margin scores in [0, 1] for use in hybrid ranking.
    Products not found in the CSV receive the average margin.
    """

    def __init__(self, margins_csv: str = None):
        self._margin_dict: Dict[str, float] = {}
        self._avg_margin: float = DEFAULT_MARGIN

        if margins_csv:
            self.load(margins_csv)

    def load(self, margins_csv: str) -> "BusinessMargin":
        """Load margin data from CSV (columns: product, margin)."""
        df = pd.read_csv(margins_csv)
        df["product"] = df["product"].str.upper().str.strip()
        self._margin_dict = dict(zip(df["product"], df["margin"].astype(float)))
        self._avg_margin = float(df["margin"].mean())
        print(f"Loaded {len(self._margin_dict)} product margins (avg={self._avg_margin:.2f})")
        return self

    def margin_score(self, product: str) -> float:
        """Return the raw margin for a product (default = dataset average)."""
        return self._margin_dict.get(product.upper().strip(), self._avg_margin)

    def get_scores(self, products: List[str]) -> Dict[str, float]:
        """Return a raw margin score dict for a list of products."""
        return {p: self.margin_score(p) for p in products}

    def get_normalized_scores(self, products: List[str]) -> Dict[str, float]:
        """Return margin scores normalized to [0, 1] over the provided list."""
        scores = self.get_scores(products)
        if not scores:
            return {}
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    @property
    def avg_margin(self) -> float:
        return self._avg_margin

    @property
    def known_products(self) -> List[str]:
        return list(self._margin_dict.keys())


def load_margins(margins_csv: str = None) -> BusinessMargin:
    """Convenience factory - resolves default path if none given."""
    if margins_csv is None:
        margins_csv = str(
            Path(__file__).parent.parent / "data" / "product_margins.csv"
        )
    return BusinessMargin(margins_csv)


if __name__ == "__main__":
    bm = load_margins()
    sample = [
        "WHITE HANGING HEART T-LIGHT HOLDER",
        "JUMBO BAG RED RETROSPOT",
        "UNKNOWN PRODUCT XYZ",
    ]
    print("\nMargin scores:")
    for p in sample:
        print(f"  {p}: {bm.margin_score(p):.2f}")
