"""Check product name matching issue"""
import pandas as pd

# Load rules
rules = pd.read_csv('models/association_rules.csv')
print("="*80)
print("CHECKING PRODUCT NAME MATCHING")
print("="*80)

# Get sample products from rules
print("\n1. Sample products in RULES (from antecedents):")
sample_products = set()
for ant_str in rules['antecedents_str'].head(20):
    products = ant_str.split(', ')
    sample_products.update(products)

for i, p in enumerate(list(sample_products)[:10], 1):
    print(f"   {i}. {p}")

# Load transaction data
df = pd.read_csv('data/processed/transactions_clean.csv')
print(f"\n2. Total unique products in TRANSACTION DATA: {df['Itemname'].nunique()}")
print("   Sample products from data:")
for i, p in enumerate(df['Itemname'].unique()[:10], 1):
    print(f"   {i}. {p}")

# Test the specific products from user's basket
test_products = [
    '12 EGG HOUSE PAINTED WOOD',
    '12 DAISY PEGS IN WOOD BOX',
    '10 COLOUR SPACEBOY PEN'
]

print(f"\n3. Testing if USER'S BASKET products exist in transaction data:")
for p in test_products:
    exists = p in df['Itemname'].values
    print(f"   '{p}': {exists}")

# Check if these products exist in rules
print(f"\n4. Testing if USER'S BASKET products exist in RULES:")
all_rule_products = set()
for ant_str in rules['antecedents_str']:
    all_rule_products.update(ant_str.split(', '))
for cons_str in rules['consequents_str']:
    all_rule_products.update(cons_str.split(', '))

print(f"   Total unique products in rules: {len(all_rule_products)}")
for p in test_products:
    exists = p in all_rule_products
    print(f"   '{p}': {exists}")

# Find similar products
print(f"\n5. Looking for SIMILAR product names in rules:")
for test_p in test_products[:2]:  # Just check first 2
    print(f"\n   Searching for products containing '{test_p[:15]}'...")
    similar = [p for p in all_rule_products if test_p[:10] in p or p[:10] in test_p]
    if similar:
        for s in similar[:5]:
            print(f"      - {s}")
    else:
        print(f"      No similar products found")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)
if test_products[0] in df['Itemname'].values and test_products[0] not in all_rule_products:
    print("❌ PROBLEM FOUND: Products exist in transaction data but NOT in rules!")
    print("   This means the association rule mining didn't include these products.")
    print("   Likely cause: Products don't meet minimum support threshold.")
elif test_products[0] not in df['Itemname'].values:
    print("❌ PROBLEM FOUND: Products don't exist in transaction data!")
    print("   The multiselect is showing products that aren't in the processed data.")
else:
    print("✅ Products exist in both data and rules - issue is elsewhere")
