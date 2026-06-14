"""
Association Rules Mining Module

Implements Apriori and FP-Growth algorithms for market basket analysis.
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class AssociationRuleMiner:
    """
    Mine association rules from transaction data.
    """
    
    def __init__(self, basket_matrix: pd.DataFrame = None):
        """
        Initialize the miner.
        
        Args:
            basket_matrix: Binary transaction-item matrix
        """
        self.basket_matrix = basket_matrix
        self.frequent_itemsets = None
        self.rules = None
        
    def load_basket_matrix(self, filepath: str):
        """Load transaction-item matrix from file."""
        print(f"Loading basket matrix from {filepath}...")
        self.basket_matrix = pd.read_csv(filepath, index_col=0)
        print(f"Loaded matrix with shape: {self.basket_matrix.shape}")
        return self
    
    def mine_frequent_itemsets(self, min_support: float = 0.01, 
                               use_fpgrowth: bool = True):
        """
        Mine frequent itemsets using Apriori or FP-Growth.
        
        Args:
            min_support: Minimum support threshold
            use_fpgrowth: Use FP-Growth (faster) instead of Apriori
            
        Returns:
            DataFrame of frequent itemsets
        """
        print(f"\nMining frequent itemsets (min_support={min_support})...")
        print(f"Algorithm: {'FP-Growth' if use_fpgrowth else 'Apriori'}")
        
        if use_fpgrowth:
            self.frequent_itemsets = fpgrowth(
                self.basket_matrix, 
                min_support=min_support, 
                use_colnames=True
            )
        else:
            self.frequent_itemsets = apriori(
                self.basket_matrix, 
                min_support=min_support, 
                use_colnames=True
            )
        
        print(f"Found {len(self.frequent_itemsets)} frequent itemsets")
        
        # Add itemset size
        self.frequent_itemsets['length'] = self.frequent_itemsets['itemsets'].apply(len)
        
        return self.frequent_itemsets
    
    def generate_rules(self, metric: str = "lift", min_threshold: float = 1.0):
        """
        Generate association rules from frequent itemsets.
        
        Args:
            metric: Metric to use for filtering (lift, confidence, support)
            min_threshold: Minimum threshold for the metric
            
        Returns:
            DataFrame of association rules
        """
        if self.frequent_itemsets is None:
            raise ValueError("Must mine frequent itemsets first")
        
        print(f"\nGenerating association rules (metric={metric}, min_threshold={min_threshold})...")
        
        self.rules = association_rules(
            self.frequent_itemsets,
            metric=metric,
            min_threshold=min_threshold
        )
        
        print(f"Generated {len(self.rules)} rules")
        
        # Add additional metrics
        self.rules['antecedent_len'] = self.rules['antecedents'].apply(len)
        self.rules['consequent_len'] = self.rules['consequents'].apply(len)
        
        # Convert frozensets to strings for better readability
        self.rules['antecedents_str'] = self.rules['antecedents'].apply(
            lambda x: ', '.join(list(x))
        )
        self.rules['consequents_str'] = self.rules['consequents'].apply(
            lambda x: ', '.join(list(x))
        )
        
        return self.rules
    
    def get_top_rules(self, n: int = 20, sort_by: str = 'lift'):
        """
        Get top N rules sorted by a metric.
        
        Args:
            n: Number of top rules to return
            sort_by: Metric to sort by (lift, confidence, support, conviction)
            
        Returns:
            DataFrame of top rules
        """
        if self.rules is None:
            raise ValueError("Must generate rules first")
        
        return self.rules.nlargest(n, sort_by)
    
    def filter_rules(self, min_support: float = None, min_confidence: float = None,
                    min_lift: float = None, min_conviction: float = None):
        """
        Filter rules by multiple criteria.
        
        Args:
            min_support: Minimum support
            min_confidence: Minimum confidence
            min_lift: Minimum lift
            min_conviction: Minimum conviction
            
        Returns:
            Filtered DataFrame of rules
        """
        if self.rules is None:
            raise ValueError("Must generate rules first")
        
        filtered = self.rules.copy()
        
        if min_support:
            filtered = filtered[filtered['support'] >= min_support]
        if min_confidence:
            filtered = filtered[filtered['confidence'] >= min_confidence]
        if min_lift:
            filtered = filtered[filtered['lift'] >= min_lift]
        if min_conviction:
            filtered = filtered[filtered['conviction'] >= min_conviction]
        
        print(f"Filtered to {len(filtered)} rules")
        return filtered
    
    def get_rules_for_item(self, item: str, as_antecedent: bool = True):
        """
        Get rules containing a specific item.
        
        Args:
            item: Item name
            as_antecedent: If True, get rules where item is in antecedent,
                          otherwise in consequent
            
        Returns:
            DataFrame of rules containing the item
        """
        if self.rules is None:
            raise ValueError("Must generate rules first")
        
        if as_antecedent:
            mask = self.rules['antecedents'].apply(lambda x: item in x)
        else:
            mask = self.rules['consequents'].apply(lambda x: item in x)
        
        return self.rules[mask]
    
    def print_rule_summary(self):
        """Print summary statistics of the rules."""
        if self.rules is None:
            raise ValueError("Must generate rules first")
        
        print("\n" + "="*80)
        print("ASSOCIATION RULES SUMMARY")
        print("="*80)
        
        print(f"\nTotal Rules: {len(self.rules)}")
        
        print(f"\nSupport Statistics:")
        print(f"  Min: {self.rules['support'].min():.4f}")
        print(f"  Max: {self.rules['support'].max():.4f}")
        print(f"  Mean: {self.rules['support'].mean():.4f}")
        
        print(f"\nConfidence Statistics:")
        print(f"  Min: {self.rules['confidence'].min():.4f}")
        print(f"  Max: {self.rules['confidence'].max():.4f}")
        print(f"  Mean: {self.rules['confidence'].mean():.4f}")
        
        print(f"\nLift Statistics:")
        print(f"  Min: {self.rules['lift'].min():.4f}")
        print(f"  Max: {self.rules['lift'].max():.4f}")
        print(f"  Mean: {self.rules['lift'].mean():.4f}")
        
        print(f"\nTop 10 Rules by Lift:")
        top_rules = self.get_top_rules(10, 'lift')
        for idx, row in top_rules.iterrows():
            print(f"\n  {row['antecedents_str']} => {row['consequents_str']}")
            print(f"    Support: {row['support']:.4f}, Confidence: {row['confidence']:.4f}, Lift: {row['lift']:.4f}")
    
    def save_rules(self, output_dir: str):
        """
        Save rules to CSV files.
        
        Args:
            output_dir: Directory to save rules
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save all rules
        rules_file = output_path / 'association_rules.csv'
        self.rules.to_csv(rules_file, index=False)
        print(f"\nSaved all rules to {rules_file}")
        
        # Save top rules by different metrics
        for metric in ['lift', 'confidence', 'support']:
            top_rules = self.get_top_rules(100, metric)
            top_file = output_path / f'top_rules_by_{metric}.csv'
            top_rules.to_csv(top_file, index=False)
            print(f"Saved top rules by {metric} to {top_file}")
        
        # Save frequent itemsets
        itemsets_file = output_path / 'frequent_itemsets.csv'
        self.frequent_itemsets.to_csv(itemsets_file, index=False)
        print(f"Saved frequent itemsets to {itemsets_file}")
        
        return self


def main():
    """Main execution function."""
    # Define paths
    data_dir = Path(__file__).parent.parent / 'data'
    matrix_file = data_dir / 'transactions' / 'basket_matrix.csv'
    models_dir = Path(__file__).parent.parent / 'models'
    
    # Mine association rules
    miner = AssociationRuleMiner()
    miner.load_basket_matrix(matrix_file)
    
    # Mine frequent itemsets with 0.5% support - OPTIMAL FOR PORTFOLIO
    # 0.005 = 0.5% of transactions (about 93 transactions out of 18,597)
    # This provides a great balance: fast execution (~1-2 mins) and solid coverage of popular items
    print("\n" + "="*80)
    print("MINING WITH 0.5% SUPPORT (PORTFOLIO OPTIMAL)")
    print("="*80)
    miner.mine_frequent_itemsets(min_support=0.005, use_fpgrowth=True)
    
    # Generate rules with low lift threshold to get more rules
    miner.generate_rules(metric='lift', min_threshold=1.0)
    
    # Print summary
    miner.print_rule_summary()
    
    # Save rules
    miner.save_rules(models_dir)
    
    print("\n" + "="*80)
    print("ASSOCIATION RULE MINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
