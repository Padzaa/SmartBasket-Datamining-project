import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk") # Larger font for social media

def generate_plots():
    # Define paths
    base_dir = Path(__file__).parent
    data_file = base_dir / 'data' / 'transactions' / 'basket_matrix.csv' # Actually need raw data for top products, or just sum the matrix
    rules_file = base_dir / 'models' / 'association_rules.csv'
    output_dir = base_dir / 'social_media'
    
    output_dir.mkdir(exist_ok=True)
    
    print("Loading data...")
    # Load rules
    rules = pd.read_csv(rules_file)
    
    # 1. Rules Scatter Plot (Support vs Confidence colored by Lift)
    print("Generating Rules Scatter Plot...")
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        rules['support'], 
        rules['confidence'], 
        c=rules['lift'], 
        cmap='viridis', 
        alpha=0.6,
        s=50
    )
    plt.colorbar(scatter, label='Lift')
    plt.title('Association Rules: Support vs Confidence (Colored by Lift)', fontsize=16, pad=20)
    plt.xlabel('Support', fontsize=14)
    plt.ylabel('Confidence', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add annotation for the "Sweet Spot"
    plt.annotate('High Value Rules', 
                 xy=(rules['support'].mean(), rules['confidence'].max()), 
                 xytext=(rules['support'].mean() + 0.005, rules['confidence'].max() - 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05))
                 
    plt.tight_layout()
    plt.savefig(output_dir / 'rules_scatter_plot.png', dpi=300)
    print(f"Saved {output_dir / 'rules_scatter_plot.png'}")
    plt.close()
    
    # 2. Top Products Bar Chart
    # Since we don't want to parse the huge raw CSV again, let's use the rules antecedents/consequents to find popular items in rules
    # Or better, load the matrix column sums if possible. 
    # Let's try loading the matrix column sums.
    
    print("Generating Top Products Plot...")
    # Reading just the header to get product names if matrix is too big? 
    # No, let's read the matrix, it's 18k rows, should be fine.
    matrix_file = base_dir / 'data' / 'transactions' / 'basket_matrix.csv'
    
    # We can use a chunked read or just read it if memory allows. 
    # The user has 18k transactions, 4k products. 18000 * 4000 * 1 byte ~ 72MB. It fits easily.
    df_matrix = pd.read_csv(matrix_file, index_col=0)
    
    # Calculate frequency
    item_counts = df_matrix.sum().sort_values(ascending=False).head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=item_counts.values, y=item_counts.index, palette='viridis')
    plt.title('Top 15 Most Frequent Products', fontsize=16, pad=20)
    plt.xlabel('Number of Transactions', fontsize=14)
    plt.ylabel('Product', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'top_products_plot.png', dpi=300)
    print(f"Saved {output_dir / 'top_products_plot.png'}")
    plt.close()

if __name__ == "__main__":
    generate_plots()
