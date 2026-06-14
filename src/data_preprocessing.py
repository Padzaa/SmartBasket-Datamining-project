"""
Data Preprocessing Module for Market Basket Analysis

This module handles data cleaning, transformation, and feature engineering
for transaction data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class TransactionDataProcessor:
    """
    Process and clean transaction data for market basket analysis.
    """
    
    def __init__(self, filepath: str, sep: str = ';'):
        """
        Initialize the processor with data file.
        
        Args:
            filepath: Path to the transaction data CSV
            sep: Separator used in the CSV file
        """
        self.filepath = filepath
        self.sep = sep
        self.df = None
        self.df_clean = None
        
    def load_data(self):
        """Load raw transaction data."""
        print(f"Loading data from {self.filepath}...")
        self.df = pd.read_csv(self.filepath, sep=self.sep, encoding='utf-8', low_memory=False)
        print(f"Loaded {len(self.df):,} records")
        return self
    
    def clean_data(self):
        """
        Clean the transaction data:
        - Parse dates
        - Handle data types
        - Remove cancelled transactions
        - Handle missing values
        - Remove duplicates
        """
        print("\nCleaning data...")
        df = self.df.copy()
        
        # Initial shape
        initial_count = len(df)
        print(f"Initial records: {initial_count:,}")
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M', errors='coerce')
        
        # Convert Price to numeric (handle comma as decimal separator if needed)
        if df['Price'].dtype == 'object':
            df['Price'] = df['Price'].str.replace(',', '.').astype(float)
        
        # Remove records with missing critical fields
        df = df.dropna(subset=['BillNo', 'Itemname', 'Date'])
        print(f"After removing missing critical fields: {len(df):,} records ({initial_count - len(df):,} removed)")
        
        # Remove cancelled transactions (negative quantities or BillNo starting with 'C')
        df = df[df['Quantity'] > 0]
        df = df[~df['BillNo'].astype(str).str.startswith('C')]
        print(f"After removing cancelled transactions: {len(df):,} records")
        
        # Remove transactions with invalid prices
        df = df[df['Price'] > 0]
        print(f"After removing invalid prices: {len(df):,} records")
        
        # Clean item names (strip whitespace, convert to uppercase for consistency)
        df['Itemname'] = df['Itemname'].str.strip().str.upper()
        
        # Remove generic/non-product items
        exclude_keywords = ['POSTAGE', 'MANUAL', 'BANK CHARGES', 'DISCOUNT', 'ADJUST', 'DOTCOM']
        for keyword in exclude_keywords:
            df = df[~df['Itemname'].str.contains(keyword, na=False)]
        print(f"After removing non-product items: {len(df):,} records")
        
        # Remove duplicates
        df = df.drop_duplicates()
        print(f"After removing duplicates: {len(df):,} records")
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        self.df_clean = df
        print(f"\nCleaning complete. Final records: {len(df):,}")
        print(f"Total removed: {initial_count - len(df):,} ({(initial_count - len(df))/initial_count*100:.2f}%)")
        
        return self
    
    def engineer_features(self):
        """
        Create additional features for analysis:
        - Transaction value
        - Time-based features
        - Customer features
        """
        print("\nEngineering features...")
        df = self.df_clean.copy()
        
        # Transaction value
        df['TransactionValue'] = df['Quantity'] * df['Price']
        
        # Time-based features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Hour'] = df['Date'].dt.hour
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        
        # Day name
        df['DayName'] = df['Date'].dt.day_name()
        
        # Month name
        df['MonthName'] = df['Date'].dt.month_name()
        
        # Weekend flag
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        self.df_clean = df
        print("Feature engineering complete")
        
        return self
    
    def get_summary_stats(self):
        """Print summary statistics of the cleaned data."""
        df = self.df_clean
        
        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        
        print(f"\nTotal Records: {len(df):,}")
        print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Duration: {(df['Date'].max() - df['Date'].min()).days} days")
        
        print(f"\nUnique Values:")
        print(f"  Transactions: {df['BillNo'].nunique():,}")
        print(f"  Products: {df['Itemname'].nunique():,}")
        print(f"  Customers: {df['CustomerID'].nunique():,}")
        print(f"  Countries: {df['Country'].nunique():,}")
        
        print(f"\nTransaction Statistics:")
        print(f"  Total Revenue: ${df['TransactionValue'].sum():,.2f}")
        print(f"  Average Transaction Value: ${df.groupby('BillNo')['TransactionValue'].sum().mean():,.2f}")
        print(f"  Average Basket Size: {df.groupby('BillNo')['Itemname'].count().mean():.2f} items")
        
        print(f"\nTop 5 Countries by Transactions:")
        print(df['Country'].value_counts().head())
        
        print(f"\nTop 10 Products:")
        print(df['Itemname'].value_counts().head(10))
        
        return self
    
    def create_transaction_matrix(self, min_support: float = 0.001):
        """
        Create transaction-item matrix for association rule mining.
        
        Args:
            min_support: Minimum support threshold to filter items
            
        Returns:
            Binary transaction-item matrix
        """
        print(f"\nCreating transaction-item matrix (min_support={min_support})...")
        
        # Calculate item frequencies
        item_freq = self.df_clean['Itemname'].value_counts()
        total_transactions = self.df_clean['BillNo'].nunique()
        item_support = item_freq / total_transactions
        
        # Filter items by minimum support
        frequent_items = item_support[item_support >= min_support].index.tolist()
        print(f"Items meeting min_support threshold: {len(frequent_items)} out of {len(item_freq)}")
        
        # Filter dataframe to only include frequent items
        df_filtered = self.df_clean[self.df_clean['Itemname'].isin(frequent_items)]
        
        # Create transaction-item matrix
        basket = df_filtered.groupby(['BillNo', 'Itemname'])['Quantity'].sum().unstack().fillna(0)
        
        # Convert to binary
        basket_binary = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        print(f"Transaction matrix shape: {basket_binary.shape}")
        print(f"Sparsity: {(1 - basket_binary.sum().sum() / (basket_binary.shape[0] * basket_binary.shape[1]))*100:.2f}%")
        
        return basket_binary
    
    def save_processed_data(self, output_dir: str):
        """
        Save processed data to files.
        
        Args:
            output_dir: Directory to save processed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned data
        clean_file = output_path / 'transactions_clean.csv'
        self.df_clean.to_csv(clean_file, index=False)
        print(f"\nSaved cleaned data to {clean_file}")
        
        # Save summary statistics
        summary_file = output_path / 'data_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Dataset Summary\n")
            f.write(f"="*80 + "\n\n")
            f.write(f"Total Records: {len(self.df_clean):,}\n")
            f.write(f"Date Range: {self.df_clean['Date'].min()} to {self.df_clean['Date'].max()}\n")
            f.write(f"Unique Transactions: {self.df_clean['BillNo'].nunique():,}\n")
            f.write(f"Unique Products: {self.df_clean['Itemname'].nunique():,}\n")
            f.write(f"Unique Customers: {self.df_clean['CustomerID'].nunique():,}\n")
            f.write(f"Unique Countries: {self.df_clean['Country'].nunique():,}\n")
        print(f"Saved summary to {summary_file}")
        
        return self


def main():
    """Main execution function."""
    # Define paths
    data_dir = Path(__file__).parent.parent / 'data'
    raw_file = data_dir / 'raw' / 'transactions.csv'
    processed_dir = data_dir / 'processed'
    
    # Process data
    processor = TransactionDataProcessor(raw_file)
    processor.load_data()
    processor.clean_data()
    processor.engineer_features()
    processor.get_summary_stats()
    processor.save_processed_data(processed_dir)
    
    # Create and save transaction matrix
    basket_matrix = processor.create_transaction_matrix(min_support=0.01)
    matrix_file = data_dir / 'transactions' / 'basket_matrix.csv'
    basket_matrix.to_csv(matrix_file)
    print(f"\nSaved transaction matrix to {matrix_file}")
    
    print("\n" + "="*80)
    print("DATA PREPROCESSING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
