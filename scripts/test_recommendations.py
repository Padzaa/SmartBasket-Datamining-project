"""Deep test of recommendation engine"""
import pandas as pd
import sys
from pathlib import Path

sys.path.append('src')
from recommendations import RecommendationEngine

# Load engine
engine = RecommendationEngine()
engine.load_rules('models/association_rules.csv')

# Load actual product names from data
df = pd.read_csv('data/processed/transactions_clean.csv')
actual_products = df['Itemname'].unique()

print(f"Total products in data: {len(actual_products)}")
print(f"Sample products: {actual_products[:5].tolist()}\n")

# Test with actual products from the data
test_baskets = [
    ['WHITE HANGING HEART T-LIGHT HOLDER'],
    ['WHITE HANGING HEART T-LIGHT HOLDER', 'WHITE METAL LANTERN'],
    ['REGENCY CAKESTAND 3 TIER'],
    actual_products[:3].tolist(),  # First 3 products
    actual_products[10:13].tolist(),  # Different 3 products
]

for i, basket in enumerate(test_baskets, 1):
    print(f"\n{'='*80}")
    print(f"Test {i}: Basket = {basket}")
    print('='*80)
    
    recs = engine.recommend_products(basket, n=5, min_confidence=0.1)
    print(f"Recommendations found: {len(recs)}")
    
    if not recs.empty:
        print("\nTop 5 recommendations:")
        print(recs[['Product', 'Confidence', 'Lift', 'Based_On']].to_string())
    else:
        print("NO RECOMMENDATIONS FOUND")
        
        # Debug: Check if products are in rules
        basket_upper = [p.upper().strip() for p in basket]
        print(f"\nDebug - Basket (uppercase, stripped): {basket_upper}")
        
        # Check how many rules have ANY of these products
        matching_rules = 0
        for idx, rule in engine.rules_df.head(100).iterrows():
            antecedents = rule['antecedents']
            if any(item in antecedents for item in basket_upper):
                matching_rules += 1
                if matching_rules <= 3:
                    print(f"  Found rule: {antecedents} => {rule['consequents']}")
        
        print(f"\nTotal rules with basket items in first 100 rules: {matching_rules}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print('='*80)
