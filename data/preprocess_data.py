"""
Advanced Data Preprocessing Pipeline for Fraud Detection
Handles data cleaning, feature engineering, and preparation for ML models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

class FraudDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for fraud detection
    Handles missing values, outliers, feature engineering, and data transformation
    """
    
    def __init__(self):
        """Initialize the data preprocessor"""
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_stats = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def clean_transaction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw transaction data"""
        df_clean = df.copy()
        
        self.logger.info(f"Starting data cleaning for {len(df_clean)} transactions")
        
        # Convert timestamp to datetime
        if 'timestamp' in df_clean.columns:
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
        
        # Remove duplicates based on transaction_id
        if 'transaction_id' in df_clean.columns:
            initial_count = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset=['transaction_id'])
            duplicates_removed = initial_count - len(df_clean)
            if duplicates_removed > 0:
                self.logger.warning(f"Removed {duplicates_removed} duplicate transactions")
        
        # Handle negative amounts
        if 'amount' in df_clean.columns:
            negative_amounts = (df_clean['amount'] < 0).sum()
            if negative_amounts > 0:
                self.logger.warning(f"Found {negative_amounts} negative amounts, setting to absolute value")
                df_clean['amount'] = df_clean['amount'].abs()
        
        # Remove transactions with missing critical fields
        critical_fields = ['user_id', 'merchant_id', 'amount', 'timestamp']
        missing_critical = df_clean[critical_fields].isnull().any(axis=1).sum()
        if missing_critical > 0:
            self.logger.warning(f"Removing {missing_critical} transactions with missing critical fields")
            df_clean = df_clean.dropna(subset=critical_fields)
        
        # Handle outliers in amount
        df_clean = self._handle_amount_outliers(df_clean)
        
        # Standardize categorical values
        df_clean = self._standardize_categorical_values(df_clean)
        
        self.logger.info(f"Data cleaning completed. Final count: {len(df_clean)} transactions")
        
        return df_clean
    
    def _handle_amount_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in transaction amounts using IQR method"""
        if 'amount' not in df.columns:
            return df
        
        df_clean = df.copy()
        
        # Calculate IQR
        Q1 = df_clean['amount'].quantile(0.25)
        Q3 = df_clean['amount'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 3 * IQR  # Use 3*IQR for extreme outliers
        upper_bound = Q3 + 3 * IQR
        
        # Count outliers
        outliers = ((df_clean['amount'] < lower_bound) | (df_clean['amount'] > upper_bound)).sum()
        
        if outliers > 0:
            self.logger.info(f"Found {outliers} amount outliers. Capping values.")
            
            # Cap outliers instead of removing (to preserve data)
            df_clean['amount'] = np.clip(df_clean['amount'], lower_bound, upper_bound)
        
        return df_clean
    
    def _standardize_categorical_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize categorical values"""
        df_clean = df.copy()
        
        # Standardize country codes (uppercase)
        if 'country' in df_clean.columns:
            df_clean['country'] = df_clean['country'].str.upper()
        
        # Standardize currency codes (uppercase)  
        if 'currency' in df_clean.columns:
            df_clean['currency'] = df_clean['currency'].str.upper()
        
        # Standardize payment methods (lowercase, replace spaces)
        if 'payment_method' in df_clean.columns:
            df_clean['payment_method'] = df_clean['payment_method'].str.lower().str.replace(' ', '_')
        
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for fraud detection"""
        df_features = df.copy()
        
        self.logger.info("Starting feature engineering...")
        
        # Time-based features
        df_features = self._create_time_features(df_features)
        
        # Amount-based features
        df_features = self._create_amount_features(df_features)
        
        # Velocity features (transaction frequency)
        df_features = self._create_velocity_features(df_features)
        
        # Categorical features
        df_features = self._create_categorical_features(df_features)
        
        # Behavioral features
        df_features = self._create_behavioral_features(df_features)
        
        # Risk-based features
        df_features = self._create_risk_features(df_features)
        
        self.logger.info(f"Feature engineering completed. Total features: {len(df_features.columns)}")
        
        return df_features
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        if 'timestamp' not in df.columns:
            return df
        
        df_time = df.copy()
        
        # Ensure timestamp is datetime
        df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
        
        # Basic time features
        df_time['hour'] = df_time['timestamp'].dt.hour
        df_time['day_of_week'] = df_time['timestamp'].dt.dayofweek
        df_time['day_of_month'] = df_time['timestamp'].dt.day
        df_time['month'] = df_time['timestamp'].dt.month
        df_time['is_weekend'] = (df_time['day_of_week'] >= 5).astype(int)
        
        # Business hours (9 AM to 6 PM)
        df_time['is_business_hours'] = ((df_time['hour'] >= 9) & (df_time['hour'] <= 18)).astype(int)
        
        # Late night transactions (11 PM to 6 AM)
        df_time['is_late_night'] = ((df_time['hour'] >= 23) | (df_time['hour'] <= 6)).astype(int)
        
        # Time since epoch (for sorting and intervals)
        df_time['timestamp_epoch'] = df_time['timestamp'].astype('int64') // 10**9
        
        return df_time
    
    def _create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create amount-based features"""
        if 'amount' not in df.columns:
            return df
        
        df_amount = df.copy()
        
        # Log transformation of amount (helps with skewed distribution)
        df_amount['amount_log'] = np.log1p(df_amount['amount'])
        
        # Amount categories
        df_amount['amount_category'] = pd.cut(df_amount['amount'], 
                                            bins=[0, 10, 50, 100, 500, 1000, float('inf')],
                                            labels=['micro', 'small', 'medium', 'large', 'very_large', 'extreme'])
        
        # Round amounts (potential indicator of manual entry)
        df_amount['is_round_amount'] = (df_amount['amount'] % 1 == 0).astype(int)
        df_amount['is_very_round'] = (df_amount['amount'] % 100 == 0).astype(int)
        
        return df_amount
    
    def _create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create velocity-based features (transaction frequency)"""
        if 'user_id' not in df.columns or 'timestamp' not in df.columns:
            return df
        
        df_velocity = df.copy()
        df_velocity = df_velocity.sort_values(['user_id', 'timestamp'])
        
        # Time between transactions for same user
        df_velocity['time_since_last_txn'] = df_velocity.groupby('user_id')['timestamp'].diff().dt.total_seconds()
        
        # Count of transactions in different time windows
        df_velocity['txn_count_1h'] = self._calculate_transaction_count(df_velocity, window_hours=1)
        df_velocity['txn_count_24h'] = self._calculate_transaction_count(df_velocity, window_hours=24)
        df_velocity['txn_count_7d'] = self._calculate_transaction_count(df_velocity, window_hours=168)  # 7 days
        
        # Amount velocity features
        df_velocity['amount_sum_1h'] = self._calculate_amount_sum(df_velocity, window_hours=1)
        df_velocity['amount_sum_24h'] = self._calculate_amount_sum(df_velocity, window_hours=24)
        
        # Unique merchants in time windows
        df_velocity['unique_merchants_24h'] = self._calculate_unique_merchants(df_velocity, window_hours=24)
        
        return df_velocity
    
    def _calculate_transaction_count(self, df: pd.DataFrame, window_hours: int) -> pd.Series:
        """Calculate transaction count in rolling window"""
        result = []
        
        for idx, row in df.iterrows():
            user_id = row['user_id']
            timestamp = row['timestamp']
            window_start = timestamp - timedelta(hours=window_hours)
            
            user_txns = df[(df['user_id'] == user_id) & 
                          (df['timestamp'] >= window_start) & 
                          (df['timestamp'] < timestamp)]
            
            result.append(len(user_txns))
        
        return pd.Series(result, index=df.index)
    
    def _calculate_amount_sum(self, df: pd.DataFrame, window_hours: int) -> pd.Series:
        """Calculate amount sum in rolling window"""
        result = []
        
        for idx, row in df.iterrows():
            user_id = row['user_id']
            timestamp = row['timestamp']
            window_start = timestamp - timedelta(hours=window_hours)
            
            user_txns = df[(df['user_id'] == user_id) & 
                          (df['timestamp'] >= window_start) & 
                          (df['timestamp'] < timestamp)]
            
            result.append(user_txns['amount'].sum() if len(user_txns) > 0 else 0)
        
        return pd.Series(result, index=df.index)
    
    def _calculate_unique_merchants(self, df: pd.DataFrame, window_hours: int) -> pd.Series:
        """Calculate unique merchants in rolling window"""
        result = []
        
        for idx, row in df.iterrows():
            user_id = row['user_id']
            timestamp = row['timestamp']
            window_start = timestamp - timedelta(hours=window_hours)
            
            user_txns = df[(df['user_id'] == user_id) & 
                          (df['timestamp'] >= window_start) & 
                          (df['timestamp'] < timestamp)]
            
            result.append(user_txns['merchant_id'].nunique() if len(user_txns) > 0 else 0)
        
        return pd.Series(result, index=df.index)
    
    def _create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from categorical variables"""
        df_cat = df.copy()
        
        # Payment method risk scores (based on typical fraud rates)
        payment_risk = {
            'credit_card': 0.3,
            'debit_card': 0.2,
            'wire_transfer': 0.8,
            'bank_transfer': 0.1,
            'paypal': 0.15,
            'apple_pay': 0.05,
            'google_pay': 0.05
        }
        
        if 'payment_method' in df_cat.columns:
            df_cat['payment_method_risk'] = df_cat['payment_method'].map(payment_risk).fillna(0.5)
        
        # Country risk scores (simplified)
        country_risk = {
            'US': 0.2, 'CA': 0.1, 'UK': 0.15, 'DE': 0.1, 'FR': 0.12,
            'AU': 0.1, 'JP': 0.08, 'RU': 0.8, 'CN': 0.6, 'NG': 0.9
        }
        
        if 'country' in df_cat.columns:
            df_cat['country_risk'] = df_cat['country'].map(country_risk).fillna(0.5)
        
        return df_cat
    
    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral features"""
        df_behavior = df.copy()
        
        if 'user_id' not in df_behavior.columns:
            return df_behavior
        
        # User transaction statistics
        user_stats = df_behavior.groupby('user_id').agg({
            'amount': ['mean', 'std', 'min', 'max', 'count'],
            'timestamp': ['min', 'max']
        }).round(2)
        
        # Flatten column names
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
        user_stats = user_stats.add_prefix('user_')
        
        # Merge back to main dataframe
        df_behavior = df_behavior.merge(user_stats, left_on='user_id', right_index=True, how='left')
        
        # Deviation from user's normal behavior
        if 'amount' in df_behavior.columns:
            df_behavior['amount_deviation'] = abs(df_behavior['amount'] - df_behavior['user_amount_mean']) / (df_behavior['user_amount_std'] + 1)
        
        return df_behavior
    
    def _create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk-based features"""
        df_risk = df.copy()
        
        # High-risk combinations
        if 'amount' in df_risk.columns and 'is_late_night' in df_risk.columns:
            df_risk['high_amount_late_night'] = ((df_risk['amount'] > 1000) & (df_risk['is_late_night'] == 1)).astype(int)
        
        if 'payment_method' in df_risk.columns and 'amount' in df_risk.columns:
            df_risk['wire_transfer_high_amount'] = ((df_risk['payment_method'] == 'wire_transfer') & (df_risk['amount'] > 5000)).astype(int)
        
        # Risk score combination
        risk_features = []
        if 'payment_method_risk' in df_risk.columns:
            risk_features.append('payment_method_risk')
        if 'country_risk' in df_risk.columns:
            risk_features.append('country_risk')
        
        if risk_features:
            df_risk['combined_risk_score'] = df_risk[risk_features].mean(axis=1)
        
        return df_risk
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features for machine learning"""
        df_encoded = df.copy()
        
        # Categorical columns to encode
        categorical_columns = [
            'payment_method', 'country', 'amount_category'
        ]
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if fit:
                    # Create and fit encoder
                    encoder = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = encoder.fit_transform(df_encoded[col].fillna('unknown'))
                    self.encoders[col] = encoder
                else:
                    # Use existing encoder
                    if col in self.encoders:
                        # Handle unseen categories
                        unique_values = set(df_encoded[col].fillna('unknown').unique())
                        known_values = set(self.encoders[col].classes_)
                        unseen_values = unique_values - known_values
                        
                        if unseen_values:
                            self.logger.warning(f"Unseen categories in {col}: {unseen_values}")
                            df_encoded[col] = df_encoded[col].replace(
                                {val: 'unknown' for val in unseen_values}
                            )
                        
                        df_encoded[f'{col}_encoded'] = self.encoders[col].transform(df_encoded[col].fillna('unknown'))
        
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        df_scaled = df.copy()
        
        # Numerical columns to scale
        numerical_columns = [
            'amount', 'amount_log', 'hour', 'day_of_week',
            'txn_count_1h', 'txn_count_24h', 'amount_sum_1h', 'amount_sum_24h',
            'payment_method_risk', 'country_risk', 'combined_risk_score'
        ]
        
        # Filter existing columns
        numerical_columns = [col for col in numerical_columns if col in df_scaled.columns]
        
        if numerical_columns:
            if fit:
                # Create and fit scaler
                scaler = StandardScaler()
                df_scaled[numerical_columns] = scaler.fit_transform(df_scaled[numerical_columns].fillna(0))
                self.scalers['numerical'] = scaler
            else:
                # Use existing scaler
                if 'numerical' in self.scalers:
                    df_scaled[numerical_columns] = self.scalers['numerical'].transform(df_scaled[numerical_columns].fillna(0))
        
        return df_scaled
    
    def prepare_model_data(self, df: pd.DataFrame, target_column: str = 'is_fraud') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for machine learning models"""
        df_model = df.copy()
        
        # Features to exclude from model training
        exclude_features = [
            'transaction_id', 'user_id', 'merchant_id', 'timestamp',
            'payment_method', 'country', 'amount_category',  # Keep encoded versions
            'ip_address', 'device_id'  # Remove identifiers
        ]
        
        # Remove excluded features
        feature_columns = [col for col in df_model.columns 
                          if col not in exclude_features and col != target_column]
        
        X = df_model[feature_columns]
        y = df_model[target_column] if target_column in df_model.columns else None
        
        # Handle missing values in features
        X = X.fillna(0)
        
        self.logger.info(f"Model data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def full_preprocessing_pipeline(self, df: pd.DataFrame, fit: bool = True, target_column: str = 'is_fraud') -> Tuple[pd.DataFrame, pd.Series]:
        """Run the complete preprocessing pipeline"""
        self.logger.info("Starting full preprocessing pipeline...")
        
        # Step 1: Clean data
        df_clean = self.clean_transaction_data(df)
        
        # Step 2: Engineer features
        df_features = self.engineer_features(df_clean)
        
        # Step 3: Encode categorical features
        df_encoded = self.encode_categorical_features(df_features, fit=fit)
        
        # Step 4: Scale numerical features
        df_scaled = self.scale_numerical_features(df_encoded, fit=fit)
        
        # Step 5: Prepare model data
        X, y = self.prepare_model_data(df_scaled, target_column)
        
        self.logger.info("Preprocessing pipeline completed successfully!")
        
        return X, y

def main():
    """Main preprocessing function"""
    # Initialize preprocessor
    preprocessor = FraudDataPreprocessor()
    
    # Load raw transaction data
    print("Loading transaction data...")
    transactions_df = pd.read_csv('raw/transactions.csv')
    
    print(f"Loaded {len(transactions_df)} transactions")
    print(f"Fraud rate: {transactions_df['is_fraud'].mean()*100:.2f}%")
    
    # Run preprocessing pipeline
    print("Running preprocessing pipeline...")
    X, y = preprocessor.full_preprocessing_pipeline(transactions_df, fit=True)
    
    print(f"Preprocessed data shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    # Save processed data
    print("Saving processed data...")
    
    # Combine features and target
    processed_df = X.copy()
    if y is not None:
        processed_df['is_fraud'] = y
    
    processed_df.to_parquet('processed/processed_transactions.parquet', index=False)
    
    # Save feature statistics
    feature_stats = {
        'feature_count': len(X.columns),
        'feature_names': list(X.columns),
        'data_shape': X.shape,
        'fraud_rate': y.mean() if y is not None else None
    }
    
    import json
    with open('processed/feature_stats.json', 'w') as f:
        json.dump(feature_stats, f, indent=2)
    
    print("Preprocessing completed successfully!")
    print(f"Processed data saved to: processed/processed_transactions.parquet")

if __name__ == "__main__":
    main()
