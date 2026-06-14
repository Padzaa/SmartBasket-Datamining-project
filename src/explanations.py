"""
Explainable Recommendations Module

Generates human-readable business explanations for each recommendation,
citing the strongest supporting association rule when available.
"""

import pandas as pd
from typing import List, Optional


class ExplanationGenerator:
    """
    Produces markdown-formatted explanations for recommended products.
    Prefers FP-Growth rule evidence; falls back to ALS / popularity signals.
    """

    def __init__(self, rules_df: pd.DataFrame):
        """
        Args:
            rules_df: Filtered association rules DataFrame.
                      Must have columns: antecedents_str, consequents_str,
                      support, confidence, lift.
        """
        self.rules_df = rules_df.copy()
        # Pre-build a lookup: consequent product → list of rules
        self._consequent_index: dict = {}
        for _, row in rules_df.iterrows():
            key = row["consequents_str"].upper().strip()
            self._consequent_index.setdefault(key, []).append(row)

    def _best_rule_for(
        self, basket_items: List[str], product: str
    ) -> Optional[pd.Series]:
        """Find the highest-confidence rule whose consequent is `product` and
        whose antecedent overlaps with the basket."""
        product_upper = product.upper().strip()
        basket_upper = {it.upper().strip() for it in basket_items}

        candidate_rules = self._consequent_index.get(product_upper, [])
        best = None
        best_conf = -1.0

        for rule in candidate_rules:
            antecedents = {
                a.strip() for a in rule["antecedents_str"].upper().split(",")
            }
            if antecedents & basket_upper and rule["confidence"] > best_conf:
                best_conf = rule["confidence"]
                best = rule

        return best

    def generate(
        self,
        basket_items: List[str],
        product: str,
        fp_score: float = 0.0,
        als_score: float = 0.0,
    ) -> str:
        """
        Generate a single explanation string for a recommended product.

        Returns a markdown-formatted string suitable for st.markdown().
        """
        rule = self._best_rule_for(basket_items, product)

        if rule is not None and fp_score >= als_score:
            antecedents_display = rule["antecedents_str"].title()
            confidence_pct = f"{rule['confidence']:.0%}"
            lift_val = f"{rule['lift']:.2f}"
            support_pct = f"{rule['support']:.1%}"
            return (
                f"Customers who buy **{antecedents_display}** also buy "
                f"**{product.title()}** in **{confidence_pct}** of cases "
                f"(lift {lift_val}×, seen in {support_pct} of transactions)."
            )

        if als_score > fp_score:
            return (
                f"Users with similar purchasing behaviour frequently add "
                f"**{product.title()}** to their basket."
            )

        return (
            f"**{product.title()}** is a popular complementary product "
            f"based on purchase patterns across similar customers."
        )

    def generate_batch(
        self,
        basket_items: List[str],
        recommendations_df: pd.DataFrame,
        product_col: str = "product",
        fp_col: str = "fp_score",
        als_col: str = "als_score",
    ) -> List[str]:
        """Generate explanations for every row in a recommendations DataFrame."""
        results = []
        for _, row in recommendations_df.iterrows():
            explanation = self.generate(
                basket_items=basket_items,
                product=row[product_col],
                fp_score=row.get(fp_col, 0.0),
                als_score=row.get(als_col, 0.0),
            )
            results.append(explanation)
        return results
