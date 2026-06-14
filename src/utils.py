"""
Utility functions for Market Basket Analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Set style for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_transaction_data(filepath: str, sep: str = ';') -> pd.DataFrame:
    """
    Load transaction data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        sep: Separator used in the CSV file
        
    Returns:
        DataFrame with transaction data
    """
    df = pd.read_csv(filepath, sep=sep, encoding='utf-8', low_memory=False)
    return df


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Get comprehensive summary statistics of the dataset.
    
    Args:
        df: Transaction DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_records': len(df),
        'total_transactions': df['BillNo'].nunique(),
        'total_products': df['Itemname'].nunique(),
        'total_customers': df['CustomerID'].nunique(),
        'total_countries': df['Country'].nunique(),
        'date_range': (df['Date'].min(), df['Date'].max()),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    return summary


def format_currency(value: float, currency: str = '$') -> str:
    """Format value as currency."""
    return f"{currency}{value:,.2f}"


def format_number(value: int) -> str:
    """Format large numbers with commas."""
    return f"{value:,}"


def create_transaction_matrix(df: pd.DataFrame, 
                              transaction_col: str = 'BillNo',
                              item_col: str = 'Itemname') -> pd.DataFrame:
    """
    Create a transaction-item matrix (one-hot encoded).
    
    Args:
        df: Transaction DataFrame
        transaction_col: Column name for transaction ID
        item_col: Column name for item name
        
    Returns:
        Binary matrix where rows are transactions and columns are items
    """
    # Create a binary indicator
    basket = df.groupby([transaction_col, item_col])['Quantity'].sum().unstack().reset_index().fillna(0).set_index(transaction_col)
    
    # Convert to binary (1 if item purchased, 0 otherwise)
    basket_binary = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    return basket_binary


def get_top_items(df: pd.DataFrame, 
                  item_col: str = 'Itemname',
                  n: int = 20) -> pd.DataFrame:
    """
    Get top N items by frequency.
    
    Args:
        df: Transaction DataFrame
        item_col: Column name for item name
        n: Number of top items to return
        
    Returns:
        DataFrame with top items and their counts
    """
    top_items = df[item_col].value_counts().head(n).reset_index()
    top_items.columns = ['Item', 'Frequency']
    return top_items


def calculate_basket_metrics(df: pd.DataFrame,
                            transaction_col: str = 'BillNo',
                            price_col: str = 'Price',
                            quantity_col: str = 'Quantity') -> pd.DataFrame:
    """
    Calculate basket-level metrics.
    
    Args:
        df: Transaction DataFrame
        transaction_col: Column name for transaction ID
        price_col: Column name for price
        quantity_col: Column name for quantity
        
    Returns:
        DataFrame with basket metrics
    """
    basket_metrics = df.groupby(transaction_col).agg({
        'Itemname': 'count',  # Number of items
        quantity_col: 'sum',   # Total quantity
        price_col: 'sum'       # Total value
    }).rename(columns={
        'Itemname': 'basket_size',
        quantity_col: 'total_quantity',
        price_col: 'basket_value'
    })
    
    return basket_metrics


def plot_top_items(df: pd.DataFrame, n: int = 20, title: str = "Top Products by Frequency"):
    """
    Create a bar plot of top items.
    
    Args:
        df: Transaction DataFrame
        n: Number of top items to display
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    top_items = get_top_items(df, n=n)
    
    fig = px.bar(top_items, 
                 x='Frequency', 
                 y='Item',
                 orientation='h',
                 title=title,
                 labels={'Frequency': 'Number of Transactions', 'Item': 'Product'},
                 color='Frequency',
                 color_continuous_scale='Viridis')
    
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    return fig


def plot_basket_distribution(basket_metrics: pd.DataFrame):
    """
    Create distribution plots for basket metrics.
    
    Args:
        basket_metrics: DataFrame with basket metrics
        
    Returns:
        Plotly figure object
    """
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Basket Size Distribution', 
                       'Total Quantity Distribution',
                       'Basket Value Distribution')
    )
    
    # Basket size
    fig.add_trace(
        go.Histogram(x=basket_metrics['basket_size'], name='Basket Size', marker_color='#636EFA'),
        row=1, col=1
    )
    
    # Total quantity
    fig.add_trace(
        go.Histogram(x=basket_metrics['total_quantity'], name='Total Quantity', marker_color='#EF553B'),
        row=1, col=2
    )
    
    # Basket value
    fig.add_trace(
        go.Histogram(x=basket_metrics['basket_value'], name='Basket Value', marker_color='#00CC96'),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False, title_text="Basket Metrics Distributions")
    
    return fig


def save_plot(fig, filepath: str, format: str = 'png'):
    """
    Save plotly figure to file.
    
    Args:
        fig: Plotly figure object
        filepath: Path to save the figure
        format: File format (png, html, etc.)
    """
    if format == 'html':
        fig.write_html(filepath)
    else:
        fig.write_image(filepath, format=format)
    print(f"Plot saved to {filepath}")


def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")
