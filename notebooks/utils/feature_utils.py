"""
Feature Engineering Utilities for Fraud Detection
Advanced feature creation, selection, and transformation functions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from typing import List, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


def create_velocity_features(df: pd.DataFrame,
                           user_col: str = 'user_id',
                           timestamp_col: str = 'timestamp',
                           amount_col: str = 'amount',
                           windows: List[str] = ['1H', '24H', '7D']) -> pd.DataFrame:
    """
    Create velocity-based features for fraud detection
    
    Args:
        df: Input DataFrame
        user_col: User identifier column
        timestamp_col: Timestamp column
        amount_col: Transaction amount column
        windows: List of time windows for velocity calculation
        
    Returns:
        DataFrame with velocity features added
    """
    df_sorted = df.sort_values([user_col, timestamp_col]).copy()
    
    for window in windows:
        window_suffix = window.replace('H', 'h').replace('D', 'd')
        
        # Transaction count velocity
        df_sorted[f'velocity_count_{window_suffix}'] = (
            df_sorted.groupby(user_col)[timestamp_col]
            .rolling(window, on=timestamp_col)
            .count()
            .reset_index(level=0, drop=True)
        )
        
        # Amount velocity
        df_sorted[f'velocity_amount_{window_suffix}'] = (
            df_sorted.groupby(user_col)[amount_col]
            .rolling(window, on=timestamp_col)
            .sum()
            .reset_index(level=0, drop=True)
        )
        
        # Average amount velocity
        df_sorted[f'velocity_avg_amount_{window_suffix}'] = (
            df_sorted[f'velocity_amount_{window_suffix}'] / 
            df_sorted[f'velocity_count_{window_suffix}']
        ).fillna(0)
    
    return df_sorted


def create_behavioral_features(df: pd.DataFrame,
                             user_col: str = 'user_id',
                             merchant_col: str = 'merchant_id',
                             timestamp_col: str = 'timestamp',
                             amount_col: str = 'amount',
                             lookback_days: int = 30) -> pd.DataFrame:
    """
    Create behavioral features based on user patterns
    
    Args:
        df: Input DataFrame
        user_col: User identifier column
        merchant_col: Merchant identifier column
        timestamp_col: Timestamp column
        amount_col: Transaction amount column
        lookback_days: Number of days to look back for pattern analysis
        
    Returns:
        DataFrame with behavioral features added
    """
    df_features = df.copy()
    
    # Sort by user and timestamp
    df_sorted = df.sort_values([user_col, timestamp_col])
    
    # Time-based features
    df_features['hour'] = df_features[timestamp_col].dt.hour
    df_features['day_of_week'] = df_features[timestamp_col].dt.dayofweek
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
    df_features['is_business_hours'] = ((df_features['hour'] >= 9) & (df_features['hour'] <= 17)).astype(int)
    
    # User historical patterns (up to current transaction)
    df_features['user_transaction_count_historical'] = df_sorted.groupby(user_col).cumcount()
    df_features['user_total_amount_historical'] = df_sorted.groupby(user_col)[amount_col].cumsum()
    df_features['user_avg_amount_historical'] = (
        df_features['user_total_amount_historical'] / 
        (df_features['user_transaction_count_historical'] + 1)
    )
    
    # Days since first transaction
    df_features['user_first_transaction'] = df_sorted.groupby(user_col)[timestamp_col].transform('first')
    df_features['days_since_first_transaction'] = (
        df_features[timestamp_col] - df_features['user_first_transaction']
    ).dt.days
    
    # User-merchant relationship features
    user_merchant_counts = df.groupby([user_col, merchant_col]).size().reset_index(name='user_merchant_frequency')
    df_features = df_features.merge(user_merchant_counts, on=[user_col, merchant_col], how='left')
    df_features['user_merchant_frequency'] = df_features['user_merchant_frequency'].fillna(1)
    
    # Is new merchant for user
    user_merchant_first = df_sorted.groupby([user_col, merchant_col])[timestamp_col].transform('first')
    df_features['is_new_merchant'] = (df_features[timestamp_col] == user_merchant_first).astype(int)
    
    # Amount deviation from user's normal behavior
    user_amount_stats = df.groupby(user_col)[amount_col].agg(['mean', 'std']).reset_index()
    user_amount_stats.columns = [user_col, 'user_avg_amount_global', 'user_std_amount_global']
    df_features = df_features.merge(user_amount_stats, on=user_col, how='left')
    
    df_features['amount_z_score'] = (
        (df_features[amount_col] - df_features['user_avg_amount_global']) / 
        df_features['user_std_amount_global'].fillna(1)
    )
    
    # Cleanup temporary columns
    columns_to_drop = ['user_first_transaction', 'user_avg_amount_global', 'user_std_amount_global']
    df_features = df_features.drop(columns=columns_to_drop, errors='ignore')
    
    return df_features


def create_interaction_features(df: pd.DataFrame,
                              feature_pairs: List[Tuple[str, str]],
                              operations: List[str] = ['multiply', 'divide', 'add', 'subtract']) -> pd.DataFrame:
    """
    Create interaction features between existing features
    
    Args:
        df: Input DataFrame
        feature_pairs: List of tuples containing feature pairs to interact
        operations: List of operations to apply ('multiply', 'divide', 'add', 'subtract')
        
    Returns:
        DataFrame with interaction features added
    """
    df_interactions = df.copy()
    
    for feat1, feat2 in feature_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            
            if 'multiply' in operations:
                df_interactions[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
            
            if 'divide' in operations and (df[feat2] != 0).all():
                df_interactions[f'{feat1}_div_{feat2}'] = df[feat1] / df[feat2].replace(0, 1)
            
            if 'add' in operations:
                df_interactions[f'{feat1}_plus_{feat2}'] = df[feat1] + df[feat2]
            
            if 'subtract' in operations:
                df_interactions[f'{feat1}_minus_{feat2}'] = df[feat1] - df[feat2]
    
    return df_interactions


def create_aggregation_features(df: pd.DataFrame,
                               group_cols: List[str],
                               agg_cols: List[str],
                               agg_functions: List[str] = ['mean', 'sum', 'count', 'std', 'min', 'max'],
                               prefix: str = '') -> pd.DataFrame:
    """
    Create aggregation features for specified groupings
    
    Args:
        df: Input DataFrame
        group_cols: Columns to group by
        agg_cols: Columns to aggregate
        agg_functions: Aggregation functions to apply
        prefix: Prefix for new column names
        
    Returns:
        DataFrame with aggregation features added
    """
    agg_features = df.copy()
    
    for agg_col in agg_cols:
        if agg_col in df.columns:
            # Create aggregation dictionary
            agg_dict = {agg_col: agg_functions}
            
            # Compute aggregations
            grouped = df.groupby(group_cols).agg(agg_dict)
            
            # Flatten column names
            grouped.columns = [f'{prefix}{"_" if prefix else ""}{agg_col}_{func}_{"_".join(group_cols)}'
                              for func in agg_functions]
            
            # Merge back to original DataFrame
            agg_features = agg_features.merge(grouped, left_on=group_cols, right_index=True, how='left')
    
    return agg_features


def create_time_based_features(df: pd.DataFrame,
                              timestamp_col: str = 'timestamp',
                              cyclical_encoding: bool = True) -> pd.DataFrame:
    """
    Create comprehensive time-based features
    
    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column
        cyclical_encoding: Whether to create cyclical encodings
        
    Returns:
        DataFrame with time-based features added
    """
    df_time = df.copy()
    
    # Extract basic time features
    df_time['year'] = df_time[timestamp_col].dt.year
    df_time['month'] = df_time[timestamp_col].dt.month
    df_time['day'] = df_time[timestamp_col].dt.day
    df_time['hour'] = df_time[timestamp_col].dt.hour
    df_time['minute'] = df_time[timestamp_col].dt.minute
    df_time['day_of_week'] = df_time[timestamp_col].dt.dayofweek
    df_time['day_of_year'] = df_time[timestamp_col].dt.dayofyear
    df_time['week_of_year'] = df_time[timestamp_col].dt.isocalendar().week
    df_time['quarter'] = df_time[timestamp_col].dt.quarter
    
    # Boolean time features
    df_time['is_weekend'] = (df_time['day_of_week'] >= 5).astype(int)
    df_time['is_month_start'] = df_time[timestamp_col].dt.is_month_start.astype(int)
    df_time['is_month_end'] = df_time[timestamp_col].dt.is_month_end.astype(int)
    df_time['is_quarter_start'] = df_time[timestamp_col].dt.is_quarter_start.astype(int)
    df_time['is_quarter_end'] = df_time[timestamp_col].dt.is_quarter_end.astype(int)
    
    # Business time features
    df_time['is_business_hours'] = ((df_time['hour'] >= 9) & (df_time['hour'] <= 17)).astype(int)
    df_time['is_lunch_time'] = ((df_time['hour'] >= 12) & (df_time['hour'] <= 14)).astype(int)
    df_time['is_late_night'] = ((df_time['hour'] >= 22) | (df_time['hour'] <= 6)).astype(int)
    df_time['is_early_morning'] = ((df_time['hour'] >= 6) & (df_time['hour'] <= 9)).astype(int)
    
    # Cyclical encoding for periodic features
    if cyclical_encoding:
        # Hour (24-hour cycle)
        df_time['hour_sin'] = np.sin(2 * np.pi * df_time['hour'] / 24)
        df_time['hour_cos'] = np.cos(2 * np.pi * df_time['hour'] / 24)
        
        # Day of week (7-day cycle)
        df_time['day_of_week_sin'] = np.sin(2 * np.pi * df_time['day_of_week'] / 7)
        df_time['day_of_week_cos'] = np.cos(2 * np.pi * df_time['day_of_week'] / 7)
        
        # Day of year (365-day cycle)
        df_time['day_of_year_sin'] = np.sin(2 * np.pi * df_time['day_of_year'] / 365)
        df_time['day_of_year_cos'] = np.cos(2 * np.pi * df_time['day_of_year'] / 365)
        
        # Month (12-month cycle)
        df_time['month_sin'] = np.sin(2 * np.pi * df_time['month'] / 12)
        df_time['month_cos'] = np.cos(2 * np.pi * df_time['month'] / 12)
    
    return df_time


def select_features_by_importance(X: pd.DataFrame,
                                y: pd.Series,
                                method: str = 'mutual_info',
                                k: int = 20,
                                random_state: int = 42) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select top k features based on importance scores
    
    Args:
        X: Feature matrix
        y: Target variable
        method: Feature selection method ('mutual_info', 'chi2', 'f_classif')
        k: Number of features to select
        random_state: Random seed
        
    Returns:
        Tuple of (selected features DataFrame, list of selected feature names)
    """
    # Handle missing values
    X_clean = X.fillna(X.median())
    
    if method == 'mutual_info':
        selector = SelectKBest(mutual_info_classif, k=k)
    elif method == 'chi2':
        # Chi2 requires non-negative features
        X_clean = X_clean - X_clean.min() + 1
        selector = SelectKBest(chi2, k=k)
    elif method == 'f_classif':
        selector = SelectKBest(f_classif, k=k)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Fit selector and transform features
    X_selected = selector.fit_transform(X_clean, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Create DataFrame with selected features
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    # Get feature scores
    scores = selector.scores_
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'score': scores,
        'selected': selector.get_support()
    }).sort_values('score', ascending=False)
    
    print(f"✅ Selected {k} features using {method}")
    print(f"Top 10 selected features:")
    selected_top = feature_scores[feature_scores['selected']].head(10)
    for idx, row in selected_top.iterrows():
        print(f"   {row['feature']}: {row['score']:.4f}")
    
    return X_selected_df, selected_features


def create_polynomial_features(df: pd.DataFrame,
                              feature_cols: List[str],
                              degree: int = 2,
                              interaction_only: bool = False,
                              include_bias: bool = False) -> pd.DataFrame:
    """
    Create polynomial features
    
    Args:
        df: Input DataFrame
        feature_cols: Columns to create polynomial features for
        degree: Degree of polynomial features
        interaction_only: Whether to include only interaction terms
        include_bias: Whether to include bias column
        
    Returns:
        DataFrame with polynomial features added
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    # Select numeric features
    numeric_features = []
    for col in feature_cols:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            numeric_features.append(col)
    
    if not numeric_features:
        print("⚠️ No numeric features found for polynomial expansion")
        return df
    
    # Create polynomial features
    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=include_bias
    )
    
    # Fit and transform
    X_poly = poly.fit_transform(df[numeric_features].fillna(0))
    
    # Create feature names
    feature_names = poly.get_feature_names_out(numeric_features)
    
    # Create DataFrame
    poly_df = pd.DataFrame(X_poly, columns=feature_names, index=df.index)
    
    # Remove original features to avoid duplication
    original_features = [name for name in feature_names if name in numeric_features]
    poly_df = poly_df.drop(columns=original_features)
    
    # Combine with original DataFrame
    result_df = pd.concat([df, poly_df], axis=1)
    
    print(f"✅ Created {len(poly_df.columns)} polynomial features (degree={degree})")
    
    return result_df


def create_categorical_features(df: pd.DataFrame,
                               categorical_cols: List[str],
                               encoding_method: str = 'onehot',
                               max_categories: int = 10) -> pd.DataFrame:
    """
    Encode categorical features
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical columns to encode
        encoding_method: Encoding method ('onehot', 'label', 'target')
        max_categories: Maximum number of categories for one-hot encoding
        
    Returns:
        DataFrame with encoded categorical features
    """
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
            
        unique_count = df[col].nunique()
        
        if encoding_method == 'onehot' and unique_count <= max_categories:
            # One-hot encoding
            encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            encoded = encoder.fit_transform(df[[col]])
            
            # Create feature names
            categories = encoder.categories_[0][1:]  # Drop first category
            feature_names = [f'{col}_{cat}' for cat in categories]
            
            # Create DataFrame
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
            
            # Add to result
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), encoded_df], axis=1)
            
        elif encoding_method == 'label':
            # Label encoding
            encoder = LabelEncoder()
            df_encoded[f'{col}_encoded'] = encoder.fit_transform(df[col].astype(str))
            
        elif encoding_method == 'target':
            # Target encoding (requires target variable)
            print(f"⚠️ Target encoding not implemented for {col}")
    
    return df_encoded


def remove_correlated_features(df: pd.DataFrame,
                              threshold: float = 0.95,
                              target_col: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly correlated features
    
    Args:
        df: Input DataFrame
        threshold: Correlation threshold for removal
        target_col: Target column to exclude from correlation analysis
        
    Returns:
        Tuple of (DataFrame with removed features, list of removed features)
    """
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude target column
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if len(numeric_cols) < 2:
        return df, []
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Find pairs of highly correlated features
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to remove
    to_remove = [column for column in upper_triangle.columns 
                if any(upper_triangle[column] > threshold)]
    
    # Remove correlated features
    df_reduced = df.drop(columns=to_remove)
    
    print(f"🗑️ Removed {len(to_remove)} highly correlated features (threshold={threshold})")
    if to_remove:
        print(f"   Removed features: {to_remove[:10]}{'...' if len(to_remove) > 10 else ''}")
    
    return df_reduced, to_remove


if __name__ == "__main__":
    # Example usage
    print("🔧 Testing feature engineering utilities...")
    
    # Generate sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'user_id': np.random.choice(['U1', 'U2', 'U3'], 100),
        'merchant_id': np.random.choice(['M1', 'M2', 'M3'], 100),
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
        'amount': np.random.lognormal(3, 1, 100),
        'is_fraud': np.random.choice([0, 1], 100, p=[0.9, 0.1])
    })
    
    # Test velocity features
    sample_with_velocity = create_velocity_features(sample_data)
    print(f"   Velocity features added: {sample_with_velocity.shape}")
    
    # Test behavioral features
    sample_with_behavioral = create_behavioral_features(sample_with_velocity)
    print(f"   Behavioral features added: {sample_with_behavioral.shape}")
    
    # Test time features
    sample_with_time = create_time_based_features(sample_with_behavioral)
    print(f"   Time features added: {sample_with_time.shape}")
    
    print("\n✅ Feature engineering utilities test completed!")
