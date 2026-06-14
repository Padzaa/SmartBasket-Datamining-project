"""
Production-Grade Customer Segmentation Module

Implements advanced RFM analysis, multiple clustering algorithms,
customer lifetime value estimation, and comprehensive segment profiling.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class CustomerSegmentation:
    """
    Production-grade customer segmentation with advanced analytics.
    """
    
    def __init__(self, transactions_df: pd.DataFrame = None):
        """
        Initialize segmentation engine.
        
        Args:
            transactions_df: DataFrame with transaction data
        """
        self.transactions_df = transactions_df
        self.rfm_df = None
        self.customer_features = None
        self.scaler = None
        self.clustering_model = None
        self.clustering_metrics = {}
        
    def load_data(self, filepath: str):
        """Load transaction data from file."""
        print(f"Loading transaction data from {filepath}...")
        self.transactions_df = pd.read_csv(filepath, parse_dates=['Date'])
        print(f"Loaded {len(self.transactions_df):,} transactions")
        return self
    
    def calculate_rfm(self, reference_date: str = None):
        """
        Calculate comprehensive RFM (Recency, Frequency, Monetary) metrics.
        
        Args:
            reference_date: Reference date for recency calculation
            
        Returns:
            DataFrame with RFM metrics
        """
        print("\nCalculating RFM metrics...")
        
        # Filter for valid CustomerIDs
        df_valid = self.transactions_df[self.transactions_df['CustomerID'].notna()].copy()
        print(f"Transactions with valid CustomerID: {len(df_valid):,}")
        print(f"Unique customers: {df_valid['CustomerID'].nunique():,}")
        
        # Set reference date
        if reference_date is None:
            reference_date = df_valid['Date'].max() + pd.Timedelta(days=1)
        else:
            reference_date = pd.to_datetime(reference_date)
        
        print(f"Reference date: {reference_date}")
        
        # Calculate comprehensive customer metrics
        customer_metrics = []
        
        for customer_id, group in df_valid.groupby('CustomerID'):
            # Basic RFM
            recency = (reference_date - group['Date'].max()).days
            frequency = group['BillNo'].nunique()
            monetary = group['TransactionValue'].sum()
            
            # Additional metrics
            avg_transaction_value = monetary / frequency if frequency > 0 else 0
            total_items = group['Quantity'].sum()
            avg_basket_size = total_items / frequency if frequency > 0 else 0
            unique_products = group['Itemname'].nunique()
            
            # Time-based metrics
            first_purchase = group['Date'].min()
            last_purchase = group['Date'].max()
            customer_lifetime = (last_purchase - first_purchase).days
            
            # Purchase frequency (transactions per day)
            purchase_rate = frequency / max(customer_lifetime, 1) if customer_lifetime > 0 else frequency
            
            customer_metrics.append({
                'CustomerID': customer_id,
                'Recency': recency,
                'Frequency': frequency,
                'Monetary': monetary,
                'AvgTransactionValue': avg_transaction_value,
                'TotalItems': total_items,
                'AvgBasketSize': avg_basket_size,
                'UniqueProducts': unique_products,
                'CustomerLifetime': customer_lifetime,
                'PurchaseRate': purchase_rate,
                'FirstPurchase': first_purchase,
                'LastPurchase': last_purchase
            })
        
        rfm = pd.DataFrame(customer_metrics)
        rfm = rfm.set_index('CustomerID')
        
        # Calculate RFM scores (1-5 scale)
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        
        # Convert to numeric
        rfm['R_Score'] = rfm['R_Score'].astype(int)
        rfm['F_Score'] = rfm['F_Score'].astype(int)
        rfm['M_Score'] = rfm['M_Score'].astype(int)
        
        # Calculate composite scores
        rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
        rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        # Calculate Customer Lifetime Value (CLV) estimate
        # Simple CLV = Avg Transaction Value * Purchase Frequency * Customer Lifetime (in years)
        rfm['CLV_Estimate'] = rfm['AvgTransactionValue'] * rfm['PurchaseRate'] * 365
        
        self.rfm_df = rfm
        
        print(f"\nCalculated RFM for {len(rfm):,} customers")
        print(f"\nRFM Score Distribution:")
        print(rfm['RFM_Score'].describe())
        print(f"\nCLV Distribution:")
        print(rfm['CLV_Estimate'].describe())
        
        return rfm
    
    def create_customer_segments(self, n_clusters: int = 5, method: str = 'kmeans',
                                use_pca: bool = False, n_components: int = 3):
        """
        Create customer segments using advanced clustering.
        
        Args:
            n_clusters: Number of clusters
            method: Clustering method ('kmeans', 'hierarchical', 'dbscan')
            use_pca: Whether to use PCA for dimensionality reduction
            n_components: Number of PCA components
            
        Returns:
            DataFrame with customer segments
        """
        if self.rfm_df is None:
            raise ValueError("Must calculate RFM first")
        
        print(f"\nCreating {n_clusters} customer segments using {method.upper()}...")
        
        # Select features for clustering
        feature_cols = ['Recency', 'Frequency', 'Monetary', 'AvgTransactionValue',
                       'AvgBasketSize', 'UniqueProducts', 'CustomerLifetime', 'PurchaseRate']
        
        X = self.rfm_df[feature_cols].copy()
        
        # Handle any infinite or missing values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Standardize features using RobustScaler (better for outliers)
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Optional PCA
        if use_pca:
            print(f"Applying PCA with {n_components} components...")
            pca = PCA(n_components=n_components)
            X_scaled = pca.fit_transform(X_scaled)
            print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        
        # Perform clustering
        if method == 'kmeans':
            self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42, 
                                          n_init=20, max_iter=500)
            clusters = self.clustering_model.fit_predict(X_scaled)
        elif method == 'hierarchical':
            self.clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
            clusters = self.clustering_model.fit_predict(X_scaled)
        elif method == 'dbscan':
            self.clustering_model = DBSCAN(eps=0.5, min_samples=5)
            clusters = self.clustering_model.fit_predict(X_scaled)
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            print(f"DBSCAN found {n_clusters} clusters")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.rfm_df['Cluster'] = clusters
        
        # Calculate clustering quality metrics
        if len(set(clusters)) > 1:
            self.clustering_metrics['silhouette_score'] = silhouette_score(X_scaled, clusters)
            self.clustering_metrics['davies_bouldin_score'] = davies_bouldin_score(X_scaled, clusters)
            print(f"\nClustering Quality Metrics:")
            print(f"  Silhouette Score: {self.clustering_metrics['silhouette_score']:.3f} (higher is better)")
            print(f"  Davies-Bouldin Score: {self.clustering_metrics['davies_bouldin_score']:.3f} (lower is better)")
        
        # Assign intelligent segment names based on cluster characteristics
        self._assign_segment_names()
        
        # Calculate segment statistics
        self._calculate_segment_stats()
        
        return self.rfm_df
    
    def _assign_segment_names(self):
        """Assign meaningful names to segments based on RFM characteristics."""
        cluster_summary = self.rfm_df.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'RFM_Score': 'mean',
            'CLV_Estimate': 'mean'
        }).round(2)
        
        # Calculate percentiles for thresholds
        r_median = self.rfm_df['Recency'].median()
        f_75 = self.rfm_df['Frequency'].quantile(0.75)
        f_50 = self.rfm_df['Frequency'].quantile(0.50)
        m_75 = self.rfm_df['Monetary'].quantile(0.75)
        m_50 = self.rfm_df['Monetary'].quantile(0.50)
        
        segment_names = {}
        for cluster in cluster_summary.index:
            r = cluster_summary.loc[cluster, 'Recency']
            f = cluster_summary.loc[cluster, 'Frequency']
            m = cluster_summary.loc[cluster, 'Monetary']
            rfm_score = cluster_summary.loc[cluster, 'RFM_Score']
            
            # Intelligent naming based on multiple criteria
            if r < r_median and f > f_75 and m > m_75:
                segment_names[cluster] = 'Champions'
            elif r < r_median and f > f_50 and m > m_50:
                segment_names[cluster] = 'Loyal Customers'
            elif r < r_median and m > m_75:
                segment_names[cluster] = 'Big Spenders'
            elif r < r_median:
                segment_names[cluster] = 'Potential Loyalists'
            elif r > r_median * 2 and f < f_50:
                segment_names[cluster] = 'At Risk'
            elif r > r_median * 2:
                segment_names[cluster] = 'Hibernating'
            elif f < f_50 and m < m_50:
                segment_names[cluster] = 'Need Attention'
            else:
                segment_names[cluster] = 'Promising'
        
        self.rfm_df['Segment_Name'] = self.rfm_df['Cluster'].map(segment_names)
        
        print(f"\nSegment Distribution:")
        segment_counts = self.rfm_df['Segment_Name'].value_counts()
        for segment, count in segment_counts.items():
            pct = count / len(self.rfm_df) * 100
            print(f"  {segment}: {count:,} customers ({pct:.1f}%)")
    
    def _calculate_segment_stats(self):
        """Calculate comprehensive statistics for each segment."""
        print(f"\nSegment Characteristics:")
        
        segment_stats = self.rfm_df.groupby('Segment_Name').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'AvgTransactionValue': 'mean',
            'CLV_Estimate': 'mean',
            'RFM_Score': 'mean'
        }).round(2)
        
        # Add customer count
        segment_stats['CustomerCount'] = self.rfm_df.groupby('Segment_Name').size()
        
        # Add revenue contribution
        total_revenue = self.rfm_df['Monetary'].sum()
        segment_stats['RevenueContribution%'] = (
            self.rfm_df.groupby('Segment_Name')['Monetary'].sum() / total_revenue * 100
        ).round(2)
        
        print(segment_stats.to_string())
        
        self.customer_features = segment_stats
    
    def get_segment_profile(self, segment_name: str = None):
        """
        Get detailed profile of a customer segment.
        
        Args:
            segment_name: Name of the segment (if None, returns all segments)
            
        Returns:
            DataFrame with segment profile
        """
        if self.rfm_df is None:
            raise ValueError("Must create segments first")
        
        if segment_name:
            segment_data = self.rfm_df[self.rfm_df['Segment_Name'] == segment_name]
        else:
            segment_data = self.rfm_df
        
        profile = segment_data.agg({
            'Recency': ['mean', 'median', 'min', 'max', 'std'],
            'Frequency': ['mean', 'median', 'min', 'max', 'std'],
            'Monetary': ['mean', 'median', 'min', 'max', 'std'],
            'CLV_Estimate': ['mean', 'median', 'min', 'max'],
            'RFM_Score': ['mean', 'median']
        }).round(2)
        
        return profile
    
    def get_top_customers(self, n: int = 100, by: str = 'CLV_Estimate'):
        """
        Get top N customers by a metric.
        
        Args:
            n: Number of customers to return
            by: Metric to sort by
            
        Returns:
            DataFrame of top customers
        """
        if self.rfm_df is None:
            raise ValueError("Must calculate RFM first")
        
        if by == 'Recency':
            return self.rfm_df.nsmallest(n, by)
        else:
            return self.rfm_df.nlargest(n, by)
    
    def get_segment_recommendations(self, segment_name: str):
        """
        Get business recommendations for a specific segment.
        
        Args:
            segment_name: Name of the segment
            
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            'Champions': {
                'strategy': 'Reward and Retain',
                'actions': [
                    'VIP loyalty programs',
                    'Early access to new products',
                    'Exclusive discounts',
                    'Personalized thank you messages',
                    'Request reviews and referrals'
                ],
                'priority': 'High'
            },
            'Loyal Customers': {
                'strategy': 'Upsell and Cross-sell',
                'actions': [
                    'Recommend premium products',
                    'Bundle offers',
                    'Loyalty point bonuses',
                    'Birthday/anniversary rewards'
                ],
                'priority': 'High'
            },
            'Big Spenders': {
                'strategy': 'Increase Frequency',
                'actions': [
                    'Subscription programs',
                    'Frequent buyer rewards',
                    'Personalized recommendations',
                    'Limited-time offers'
                ],
                'priority': 'Medium-High'
            },
            'Potential Loyalists': {
                'strategy': 'Nurture and Engage',
                'actions': [
                    'Onboarding campaigns',
                    'Product education',
                    'Loyalty program enrollment',
                    'Engagement incentives'
                ],
                'priority': 'Medium'
            },
            'At Risk': {
                'strategy': 'Re-engage',
                'actions': [
                    'Win-back campaigns',
                    'Special discounts',
                    'Survey for feedback',
                    'Personalized outreach'
                ],
                'priority': 'High'
            },
            'Hibernating': {
                'strategy': 'Reactivation',
                'actions': [
                    'Aggressive win-back offers',
                    'New product announcements',
                    'Brand refresh campaigns',
                    'Survey and feedback'
                ],
                'priority': 'Medium'
            },
            'Need Attention': {
                'strategy': 'Build Relationship',
                'actions': [
                    'Targeted promotions',
                    'Educational content',
                    'Product recommendations',
                    'Engagement campaigns'
                ],
                'priority': 'Low-Medium'
            },
            'Promising': {
                'strategy': 'Develop Potential',
                'actions': [
                    'Welcome series',
                    'Product discovery',
                    'First purchase incentives',
                    'Engagement tracking'
                ],
                'priority': 'Medium'
            }
        }
        
        return recommendations.get(segment_name, {
            'strategy': 'Monitor and Analyze',
            'actions': ['Collect more data', 'Track behavior'],
            'priority': 'Low'
        })
    
    def save_segments(self, output_dir: str):
        """
        Save customer segments and analysis to files.
        
        Args:
            output_dir: Directory to save segments
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive RFM data
        rfm_file = output_path / 'customer_rfm.csv'
        self.rfm_df.to_csv(rfm_file)
        print(f"\nSaved RFM data to {rfm_file}")
        
        # Save segment summary with all metrics
        segment_summary = self.rfm_df.groupby('Segment_Name').agg({
            'Recency': ['mean', 'median'],
            'Frequency': ['mean', 'median'],
            'Monetary': ['mean', 'median', 'sum'],
            'CLV_Estimate': ['mean', 'median', 'sum'],
            'RFM_Score': 'mean'
        }).round(2)
        
        # Add customer count
        segment_summary['CustomerCount'] = self.rfm_df.groupby('Segment_Name').size()
        
        summary_file = output_path / 'segment_summary.csv'
        segment_summary.to_csv(summary_file)
        print(f"Saved segment summary to {summary_file}")
        
        # Save segment recommendations
        recommendations = []
        for segment in self.rfm_df['Segment_Name'].unique():
            rec = self.get_segment_recommendations(segment)
            recommendations.append({
                'Segment': segment,
                'Strategy': rec.get('strategy', ''),
                'Priority': rec.get('priority', ''),
                'Actions': '; '.join(rec.get('actions', []))
            })
        
        rec_df = pd.DataFrame(recommendations)
        rec_file = output_path / 'segment_recommendations.csv'
        rec_df.to_csv(rec_file, index=False)
        print(f"Saved segment recommendations to {rec_file}")
        
        # Save clustering metrics
        if self.clustering_metrics:
            metrics_file = output_path / 'clustering_metrics.txt'
            with open(metrics_file, 'w') as f:
                f.write("Clustering Quality Metrics\n")
                f.write("="*50 + "\n\n")
                for metric, value in self.clustering_metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
            print(f"Saved clustering metrics to {metrics_file}")
        
        return self


def main():
    """Main execution function."""
    # Define paths
    data_dir = Path(__file__).parent.parent / 'data'
    processed_file = data_dir / 'processed' / 'transactions_clean.csv'
    models_dir = Path(__file__).parent.parent / 'models'
    
    # Perform segmentation
    segmenter = CustomerSegmentation()
    segmenter.load_data(str(processed_file))
    segmenter.calculate_rfm()
    segmenter.create_customer_segments(n_clusters=6, method='kmeans')
    
    # Print segment profiles
    print("\n" + "="*80)
    print("CUSTOMER SEGMENT PROFILES")
    print("="*80)
    
    for segment in sorted(segmenter.rfm_df['Segment_Name'].unique()):
        print(f"\n{segment}:")
        print("-" * 40)
        profile = segmenter.get_segment_profile(segment)
        print(profile)
        
        # Print recommendations
        rec = segmenter.get_segment_recommendations(segment)
        print(f"\nStrategy: {rec.get('strategy', 'N/A')}")
        print(f"Priority: {rec.get('priority', 'N/A')}")
        print("Actions:")
        for action in rec.get('actions', []):
            print(f"  - {action}")
    
    # Save segments
    segmenter.save_segments(str(models_dir))
    
    print("\n" + "="*80)
    print("CUSTOMER SEGMENTATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
