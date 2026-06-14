"""
Recommendation Engine Module

Implements product recommendation system based on association rules.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


class RecommendationEngine:
    """
    Generate product recommendations based on association rules.
    """
    
    def __init__(self, rules_df: pd.DataFrame = None):
        """
        Initialize recommendation engine.
        
        Args:
            rules_df: DataFrame with association rules
        """
        self.rules_df = rules_df
        
    def load_rules(self, filepath: str):
        """Load association rules from file."""
        print(f"Loading association rules from {filepath}...")
        self.rules_df = pd.read_csv(filepath)
        
        # Convert string representations back to sets
        self.rules_df['antecedents'] = self.rules_df['antecedents_str'].apply(
            lambda x: set(x.split(', '))
        )
        self.rules_df['consequents'] = self.rules_df['consequents_str'].apply(
            lambda x: set(x.split(', '))
        )
        
        print(f"Loaded {len(self.rules_df)} rules")
        return self
    
    def recommend_products(self, basket: List[str], n: int = 5, 
                          min_confidence: float = 0.2) -> pd.DataFrame:
        """
        Recommend products based on current basket.
        
        Args:
            basket: List of items in current basket
            n: Number of recommendations to return
            min_confidence: Minimum confidence threshold
            
        Returns:
            DataFrame with recommended products and scores
        """
        if self.rules_df is None:
            raise ValueError("Must load rules first")
        
        basket_set = set([item.upper().strip() for item in basket])
        recommendations = []
        
        # Strategy 1: Find rules where basket contains ANY antecedent items
        for idx, rule in self.rules_df.iterrows():
            antecedents = rule['antecedents']
            
            # Check if basket has ANY overlap with antecedents
            overlap = antecedents.intersection(basket_set)
            
            if overlap:  # If there's any overlap
                consequents = rule['consequents']
                
                # Only recommend items not already in basket
                new_items = consequents - basket_set
                
                if new_items and rule['confidence'] >= min_confidence:
                    # Calculate match score based on overlap
                    match_ratio = len(overlap) / len(antecedents)
                    
                    for item in new_items:
                        recommendations.append({
                            'Product': item,
                            'Confidence': rule['confidence'],
                            'Lift': rule['lift'],
                            'Support': rule['support'],
                            'Based_On': ', '.join(overlap),
                            'Match_Ratio': match_ratio
                        })
        
        if not recommendations:
            return pd.DataFrame()
        
        # Convert to DataFrame and aggregate scores
        rec_df = pd.DataFrame(recommendations)
        
        # Aggregate by product (take max confidence and best match)
        rec_agg = rec_df.groupby('Product').agg({
            'Confidence': 'max',
            'Lift': 'max',
            'Support': 'max',
            'Based_On': 'first',
            'Match_Ratio': 'max'
        }).reset_index()
        
        # Sort by match ratio, then confidence and lift
        rec_agg['Score'] = rec_agg['Match_Ratio'] * rec_agg['Confidence'] * rec_agg['Lift']
        rec_agg = rec_agg.sort_values('Score', ascending=False)
        
        return rec_agg.head(n)
    
    def find_complementary_products(self, product: str, n: int = 10) -> pd.DataFrame:
        """
        Find products frequently bought with a given product.
        
        Args:
            product: Product name
            n: Number of complementary products to return
            
        Returns:
            DataFrame with complementary products
        """
        if self.rules_df is None:
            raise ValueError("Must load rules first")
        
        product_upper = product.upper()
        complementary = []
        
        # Find rules where product is in antecedents
        for idx, rule in self.rules_df.iterrows():
            if product_upper in rule['antecedents']:
                for item in rule['consequents']:
                    complementary.append({
                        'Product': item,
                        'Confidence': rule['confidence'],
                        'Lift': rule['lift'],
                        'Support': rule['support']
                    })
        
        if not complementary:
            return pd.DataFrame()
        
        # Convert to DataFrame and aggregate
        comp_df = pd.DataFrame(complementary)
        comp_agg = comp_df.groupby('Product').agg({
            'Confidence': 'max',
            'Lift': 'max',
            'Support': 'max'
        }).reset_index()
        
        comp_agg = comp_agg.sort_values('Lift', ascending=False)
        
        return comp_agg.head(n)
    
    def create_product_bundles(self, min_items: int = 2, max_items: int = 3,
                              min_lift: float = 3.0, n: int = 20) -> pd.DataFrame:
        """
        Create product bundles based on strong associations.
        
        Args:
            min_items: Minimum items in bundle
            max_items: Maximum items in bundle
            min_lift: Minimum lift threshold
            n: Number of bundles to return
            
        Returns:
            DataFrame with product bundles
        """
        if self.rules_df is None:
            raise ValueError("Must load rules first")
        
        bundles = []
        
        for idx, rule in self.rules_df.iterrows():
            total_items = len(rule['antecedents']) + len(rule['consequents'])
            
            if (min_items <= total_items <= max_items and 
                rule['lift'] >= min_lift):
                
                bundle_items = list(rule['antecedents'].union(rule['consequents']))
                
                bundles.append({
                    'Bundle': ', '.join(bundle_items),
                    'Items': bundle_items,
                    'Item_Count': len(bundle_items),
                    'Lift': rule['lift'],
                    'Confidence': rule['confidence'],
                    'Support': rule['support']
                })
        
        if not bundles:
            return pd.DataFrame()
        
        bundle_df = pd.DataFrame(bundles)
        bundle_df = bundle_df.sort_values('Lift', ascending=False)
        
        # Remove duplicate bundles
        bundle_df['Bundle_Sorted'] = bundle_df['Items'].apply(lambda x: ', '.join(sorted(x)))
        bundle_df = bundle_df.drop_duplicates('Bundle_Sorted')
        bundle_df = bundle_df.drop('Bundle_Sorted', axis=1)
        
        return bundle_df.head(n)
    
    def get_recommendation_stats(self):
        """Get statistics about the recommendation engine."""
        if self.rules_df is None:
            raise ValueError("Must load rules first")
        
        stats = {
            'total_rules': len(self.rules_df),
            'unique_products': len(set().union(*self.rules_df['antecedents'], 
                                              *self.rules_df['consequents'])),
            'avg_confidence': self.rules_df['confidence'].mean(),
            'avg_lift': self.rules_df['lift'].mean(),
            'high_confidence_rules': len(self.rules_df[self.rules_df['confidence'] > 0.5]),
            'high_lift_rules': len(self.rules_df[self.rules_df['lift'] > 3.0])
        }
        
        return stats


def main():
    """Main execution function."""
    # Define paths
    models_dir = Path(__file__).parent.parent / 'models'
    rules_file = models_dir / 'association_rules.csv'
    
    # Initialize recommendation engine
    engine = RecommendationEngine()
    engine.load_rules(rules_file)
    
    # Print stats
    stats = engine.get_recommendation_stats()
    print("\n" + "="*80)
    print("RECOMMENDATION ENGINE STATISTICS")
    print("="*80)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Example: Recommend products for a sample basket
    print("\n" + "="*80)
    print("SAMPLE RECOMMENDATIONS")
    print("="*80)
    
    sample_basket = ['WHITE HANGING HEART T-LIGHT HOLDER', 'CREAM CUPID HEARTS COAT HANGER']
    print(f"\nBasket: {sample_basket}")
    recommendations = engine.recommend_products(sample_basket, n=5)
    if not recommendations.empty:
        print("\nRecommended Products:")
        print(recommendations[['Product', 'Confidence', 'Lift']].to_string(index=False))
    else:
        print("No recommendations found")
    
    # Create product bundles
    print("\n" + "="*80)
    print("TOP PRODUCT BUNDLES")
    print("="*80)
    bundles = engine.create_product_bundles(min_items=2, max_items=3, min_lift=5.0, n=10)
    if not bundles.empty:
        print(bundles[['Bundle', 'Item_Count', 'Lift', 'Confidence']].to_string(index=False))
    
    print("\n" + "="*80)
    print("RECOMMENDATION ENGINE READY")
    print("="*80)


if __name__ == "__main__":
    main()
