"""
Data Utilities for Fraud Detection Analysis
Comprehensive data loading, preprocessing, and synthetic data generation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
import random
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')


def load_fraud_data(data_path: str) -> pd.DataFrame:
    """
    Load fraud detection dataset from various formats
    
    Args:
        data_path: Path to the data file
        
    Returns:
        DataFrame with fraud detection data
    """
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path, parse_dates=['timestamp'])
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
            
        print(f"✅ Data loaded successfully: {df.shape}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        raise


def clean_data(df: pd.DataFrame, 
               remove_duplicates: bool = True,
               handle_missing: str = 'median',
               remove_outliers: bool = False) -> pd.DataFrame:
    """
    Clean and preprocess fraud detection data
    
    Args:
        df: Input DataFrame
        remove_duplicates: Whether to remove duplicate rows
        handle_missing: Method to handle missing values ('median', 'mean', 'drop', 'forward')
        remove_outliers: Whether to remove outliers using IQR method
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    print(f"🧹 Starting data cleaning...")
    print(f"   Initial shape: {df_clean.shape}")
    
    # Remove duplicates
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_dupes = initial_rows - len(df_clean)
        if removed_dupes > 0:
            print(f"   Removed {removed_dupes} duplicate rows")
    
    # Handle missing values
    if handle_missing != 'none':
        missing_before = df_clean.isnull().sum().sum()
        
        if handle_missing == 'median':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            df_clean[categorical_cols] = df_clean[categorical_cols].fillna(df_clean[categorical_cols].mode().iloc[0])
            
        elif handle_missing == 'mean':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
            
        elif handle_missing == 'drop':
            df_clean = df_clean.dropna()
            
        elif handle_missing == 'forward':
            df_clean = df_clean.fillna(method='ffill')
            
        missing_after = df_clean.isnull().sum().sum()
        print(f"   Handled {missing_before - missing_after} missing values")
    
    # Remove outliers
    if remove_outliers:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        initial_rows = len(df_clean)
        
        for col in numeric_cols:
            if col != 'is_fraud':  # Don't remove outliers from target variable
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & 
                    (df_clean[col] <= upper_bound)
                ]
        
        removed_outliers = initial_rows - len(df_clean)
        if removed_outliers > 0:
            print(f"   Removed {removed_outliers} outlier rows")
    
    print(f"   Final shape: {df_clean.shape}")
    print(f"✅ Data cleaning completed")
    
    return df_clean


def generate_synthetic_fraud_data(n_transactions: int = 100000,
                                 n_users: int = 10000,
                                 n_merchants: int = 1000,
                                 fraud_rate: float = 0.02,
                                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic fraud detection dataset for testing and development
    
    Args:
        n_transactions: Number of transactions to generate
        n_users: Number of unique users
        n_merchants: Number of unique merchants
        fraud_rate: Target fraud rate (0-1)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (transactions_df, users_df, merchants_df)
    """
    np.random.seed(random_state)
    random.seed(random_state)
    
    print(f"🎲 Generating synthetic fraud data...")
    print(f"   Transactions: {n_transactions:,}")
    print(f"   Users: {n_users:,}")
    print(f"   Merchants: {n_merchants:,}")
    print(f"   Target fraud rate: {fraud_rate:.2%}")
    
    # Generate users
    users_data = []
    for i in range(n_users):
        user_risk = np.random.choice(['low', 'medium', 'high'], p=[0.7, 0.2, 0.1])
        users_data.append({
            'user_id': f'USER_{i:06d}',
            'user_risk_level': user_risk,
            'registration_date': datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1500)),
            'country': np.random.choice(['US', 'UK', 'CA', 'DE', 'FR', 'AU', 'JP', 'IN'], 
                                      p=[0.3, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05, 0.15]),
            'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], 
                                        p=[0.2, 0.3, 0.25, 0.15, 0.1])
        })
    
    users_df = pd.DataFrame(users_data)
    
    # Generate merchants
    merchants_data = []
    merchant_categories = ['electronics', 'clothing', 'food_beverage', 'travel', 
                          'entertainment', 'healthcare', 'automotive', 'home_garden',
                          'sports_outdoors', 'books_media']
    
    for i in range(n_merchants):
        category = np.random.choice(merchant_categories)
        risk_level = np.random.choice(['low', 'medium', 'high'], p=[0.6, 0.3, 0.1])
        
        merchants_data.append({
            'merchant_id': f'MERCHANT_{i:05d}',
            'merchant_name': f'Merchant {i}',
            'category': category,
            'risk_level': risk_level,
            'country': np.random.choice(['US', 'UK', 'CA', 'DE', 'FR'], p=[0.4, 0.2, 0.15, 0.15, 0.1]),
            'established_date': datetime(2010, 1, 1) + timedelta(days=np.random.randint(0, 4000))
        })
    
    merchants_df = pd.DataFrame(merchants_data)
    
    # Generate transactions
    transactions_data = []
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 10, 17)
    
    payment_methods = ['credit_card', 'debit_card', 'paypal', 'apple_pay', 'google_pay', 
                      'bank_transfer', 'cryptocurrency']
    payment_probs = [0.4, 0.25, 0.15, 0.08, 0.07, 0.03, 0.02]
    
    for i in range(n_transactions):
        # Select user and merchant
        user_idx = np.random.randint(0, n_users)
        merchant_idx = np.random.randint(0, n_merchants)
        
        user_info = users_data[user_idx]
        merchant_info = merchants_data[merchant_idx]
        
        # Generate timestamp with realistic patterns
        # Higher fraud probability during night hours and weekends
        hour = np.random.choice(range(24), p=[0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.04,
                                             0.05, 0.06, 0.06, 0.07, 0.07, 0.07, 0.06, 0.05,
                                             0.05, 0.06, 0.06, 0.07, 0.08, 0.07, 0.05, 0.03])
        
        days_diff = (end_date - start_date).days
        random_day = np.random.randint(0, days_diff)
        timestamp = start_date + timedelta(days=random_day, hours=hour, 
                                          minutes=np.random.randint(0, 60),
                                          seconds=np.random.randint(0, 60))
        
        # Generate amount with realistic distribution
        if merchant_info['category'] in ['electronics', 'automotive']:
            amount_base = np.random.lognormal(mean=5, sigma=1)  # Higher amounts
        elif merchant_info['category'] in ['food_beverage', 'entertainment']:
            amount_base = np.random.lognormal(mean=3, sigma=0.8)  # Lower amounts
        else:
            amount_base = np.random.lognormal(mean=4, sigma=1)  # Medium amounts
        
        amount = max(1.0, round(amount_base, 2))
        
        # Determine fraud probability based on various factors
        fraud_prob = fraud_rate
        
        # Risk factors that increase fraud probability
        if user_info['user_risk_level'] == 'high':
            fraud_prob *= 3
        elif user_info['user_risk_level'] == 'medium':
            fraud_prob *= 1.5
            
        if merchant_info['risk_level'] == 'high':
            fraud_prob *= 2.5
        elif merchant_info['risk_level'] == 'medium':
            fraud_prob *= 1.3
            
        # Time-based risk factors
        if hour >= 23 or hour <= 6:  # Night hours
            fraud_prob *= 1.8
            
        if timestamp.weekday() >= 5:  # Weekend
            fraud_prob *= 1.2
            
        # Amount-based risk
        if amount > 1000:
            fraud_prob *= 1.5
        elif amount > 5000:
            fraud_prob *= 2.0
            
        # Payment method risk
        payment_method = np.random.choice(payment_methods, p=payment_probs)
        if payment_method == 'cryptocurrency':
            fraud_prob *= 3.0
        elif payment_method in ['apple_pay', 'google_pay']:
            fraud_prob *= 0.7  # Lower fraud risk
            
        # Ensure fraud_prob doesn't exceed 1
        fraud_prob = min(fraud_prob, 0.8)
        
        # Determine if transaction is fraud
        is_fraud = np.random.random() < fraud_prob
        
        transaction = {
            'transaction_id': f'TXN_{i:08d}',
            'user_id': user_info['user_id'],
            'merchant_id': merchant_info['merchant_id'],
            'amount': amount,
            'currency': 'USD',
            'payment_method': payment_method,
            'timestamp': timestamp,
            'country': user_info['country'],
            'merchant_category': merchant_info['category'],
            'is_fraud': int(is_fraud)
        }
        
        # Add some additional realistic features
        if np.random.random() < 0.1:  # 10% chance of having device_id
            transaction['device_id'] = f'DEVICE_{np.random.randint(1000, 9999)}'
        
        if np.random.random() < 0.3:  # 30% chance of having IP address
            transaction['ip_address'] = f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        
        transactions_data.append(transaction)
    
    transactions_df = pd.DataFrame(transactions_data)
    
    # Calculate actual fraud rate
    actual_fraud_rate = transactions_df['is_fraud'].mean()
    
    print(f"✅ Synthetic data generated successfully!")
    print(f"   Actual fraud rate: {actual_fraud_rate:.2%}")
    print(f"   Date range: {transactions_df['timestamp'].min()} to {transactions_df['timestamp'].max()}")
    
    return transactions_df, users_df, merchants_df


def validate_data_quality(df: pd.DataFrame, 
                         target_col: str = 'is_fraud') -> Dict[str, Any]:
    """
    Comprehensive data quality assessment
    
    Args:
        df: DataFrame to validate
        target_col: Name of target column
        
    Returns:
        Dictionary with data quality metrics
    """
    quality_report = {}
    
    # Basic statistics
    quality_report['shape'] = df.shape
    quality_report['memory_usage_mb'] = df.memory_usage(deep=True).sum() / (1024**2)
    
    # Missing values
    missing_values = df.isnull().sum()
    quality_report['missing_values'] = missing_values[missing_values > 0].to_dict()
    quality_report['missing_percentage'] = (missing_values / len(df) * 100)[missing_values > 0].to_dict()
    
    # Data types
    quality_report['data_types'] = df.dtypes.to_dict()
    
    # Duplicates
    quality_report['duplicate_rows'] = df.duplicated().sum()
    
    # Target variable analysis
    if target_col in df.columns:
        quality_report['target_distribution'] = df[target_col].value_counts().to_dict()
        quality_report['target_fraud_rate'] = df[target_col].mean()
    
    # Numerical columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    quality_report['numeric_columns'] = len(numeric_cols)
    
    if numeric_cols:
        # Check for infinite values
        inf_values = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_values[col] = inf_count
        quality_report['infinite_values'] = inf_values
        
        # Check for negative values in amount columns
        negative_values = {}
        for col in numeric_cols:
            if 'amount' in col.lower():
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    negative_values[col] = neg_count
        quality_report['negative_amounts'] = negative_values
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    quality_report['categorical_columns'] = len(categorical_cols)
    
    if categorical_cols:
        cardinality = {}
        for col in categorical_cols:
            cardinality[col] = df[col].nunique()
        quality_report['cardinality'] = cardinality
    
    # Date columns analysis
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if date_cols:
        date_ranges = {}
        for col in date_cols:
            date_ranges[col] = {
                'min_date': df[col].min(),
                'max_date': df[col].max(),
                'date_range_days': (df[col].max() - df[col].min()).days
            }
        quality_report['date_ranges'] = date_ranges
    
    return quality_report


def sample_data_for_development(df: pd.DataFrame, 
                               sample_size: int = 10000,
                               stratify_col: Optional[str] = 'is_fraud',
                               random_state: int = 42) -> pd.DataFrame:
    """
    Create a representative sample of the data for development and testing
    
    Args:
        df: Input DataFrame
        sample_size: Number of samples to extract
        stratify_col: Column to stratify sampling (maintains class distribution)
        random_state: Random seed for reproducibility
        
    Returns:
        Sampled DataFrame
    """
    if len(df) <= sample_size:
        print(f"⚠️ Dataset size ({len(df)}) <= sample size ({sample_size}), returning full dataset")
        return df.copy()
    
    if stratify_col and stratify_col in df.columns:
        # Stratified sampling to maintain class distribution
        sample_df = df.groupby(stratify_col, group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size * len(x) // len(df) + 1), 
                             random_state=random_state)
        ).reset_index(drop=True)
        
        # If we have too many samples, randomly remove excess
        if len(sample_df) > sample_size:
            sample_df = sample_df.sample(sample_size, random_state=random_state).reset_index(drop=True)
            
        print(f"📊 Stratified sample created: {len(sample_df):,} rows")
        if stratify_col in sample_df.columns:
            original_dist = df[stratify_col].value_counts(normalize=True)
            sample_dist = sample_df[stratify_col].value_counts(normalize=True)
            print(f"   Original distribution: {original_dist.to_dict()}")
            print(f"   Sample distribution: {sample_dist.to_dict()}")
            
    else:
        # Random sampling
        sample_df = df.sample(sample_size, random_state=random_state).reset_index(drop=True)
        print(f"📊 Random sample created: {len(sample_df):,} rows")
    
    return sample_df


def export_data_summary(df: pd.DataFrame, 
                       output_path: str,
                       include_sample: bool = True,
                       sample_size: int = 100) -> None:
    """
    Export comprehensive data summary to file
    
    Args:
        df: DataFrame to summarize
        output_path: Path to save summary
        include_sample: Whether to include data sample
        sample_size: Size of sample to include
    """
    quality_report = validate_data_quality(df)
    
    summary_content = f"""
FRAUD DETECTION DATASET SUMMARY
===============================

Dataset Overview:
- Shape: {quality_report['shape']}
- Memory usage: {quality_report['memory_usage_mb']:.2f} MB
- Duplicate rows: {quality_report['duplicate_rows']:,}

Data Types:
- Numeric columns: {quality_report['numeric_columns']}
- Categorical columns: {quality_report['categorical_columns']}

Target Variable Analysis:
- Fraud rate: {quality_report.get('target_fraud_rate', 'N/A'):.2%}
- Distribution: {quality_report.get('target_distribution', {})}

Data Quality Issues:
- Missing values: {len(quality_report['missing_values'])} columns affected
- Infinite values: {len(quality_report.get('infinite_values', {}))} columns affected
- Negative amounts: {len(quality_report.get('negative_amounts', {}))} columns affected

Missing Values Details:
{chr(10).join([f"  - {col}: {count} ({quality_report['missing_percentage'].get(col, 0):.1f}%)" 
               for col, count in quality_report['missing_values'].items()])}

High Cardinality Columns:
{chr(10).join([f"  - {col}: {count:,} unique values" 
               for col, count in quality_report.get('cardinality', {}).items() 
               if count > 1000])}
"""
    
    if include_sample and sample_size > 0:
        sample_df = df.head(sample_size)
        summary_content += f"\n\nDATA SAMPLE (first {sample_size} rows):\n"
        summary_content += "=" * 50 + "\n"
        summary_content += sample_df.to_string(max_rows=sample_size, max_cols=20)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print(f"📄 Data summary exported to: {output_path}")


# Utility functions for specific fraud detection tasks
def detect_velocity_anomalies(df: pd.DataFrame,
                             user_col: str = 'user_id',
                             timestamp_col: str = 'timestamp',
                             window_minutes: int = 60,
                             threshold: int = 5) -> pd.DataFrame:
    """
    Detect users with high transaction velocity (potential fraud indicator)
    
    Args:
        df: Transaction DataFrame
        user_col: Column name for user ID
        timestamp_col: Column name for timestamp
        window_minutes: Time window for velocity calculation
        threshold: Minimum transactions per window to flag
        
    Returns:
        DataFrame with users flagged for high velocity
    """
    df_sorted = df.sort_values([user_col, timestamp_col])
    
    # Calculate rolling transaction count
    df_sorted['rolling_count'] = (
        df_sorted.groupby(user_col)[timestamp_col]
        .rolling(f'{window_minutes}T', on=timestamp_col)
        .count()
        .reset_index(level=0, drop=True)
    )
    
    # Identify high velocity transactions
    high_velocity = df_sorted[df_sorted['rolling_count'] >= threshold]
    
    velocity_summary = high_velocity.groupby(user_col).agg({
        'rolling_count': ['max', 'mean'],
        timestamp_col: ['min', 'max', 'count']
    }).round(2)
    
    velocity_summary.columns = [
        'max_velocity', 'avg_velocity', 
        'first_transaction', 'last_transaction', 'total_transactions'
    ]
    
    return velocity_summary.reset_index()


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing data utilities...")
    
    # Generate synthetic data
    transactions_df, users_df, merchants_df = generate_synthetic_fraud_data(
        n_transactions=1000,
        n_users=100,
        n_merchants=50,
        fraud_rate=0.05
    )
    
    # Validate data quality
    quality_report = validate_data_quality(transactions_df)
    print(f"\n📊 Data quality report:")
    for key, value in quality_report.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Data utilities test completed!")
