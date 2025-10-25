#!/usr/bin/env python
"""
Transaction Data Processing Pipeline
ETL pipeline for processing raw transaction data
"""

import pandas as pd
import numpy as np
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import asyncio
import aiofiles
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TransactionProcessor:
    """Process raw transaction data for fraud detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.input_path = Path(config['input_path'])
        self.output_path = Path(config['output_path'])
        self.batch_size = config.get('batch_size', 10000)
        self.parallel_workers = config.get('parallel_workers', 4)
        
        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def validate_input_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate input data quality"""
        issues = []
        
        # Check required columns
        required_columns = [
            'transaction_id', 'user_id', 'merchant_id', 'amount', 
            'timestamp', 'payment_method'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        if 'amount' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['amount']):
                issues.append("Amount column is not numeric")
            
            if (df['amount'] <= 0).any():
                issues.append("Found negative or zero amounts")
        
        # Check for duplicates
        if df['transaction_id'].duplicated().any():
            issues.append("Found duplicate transaction IDs")
        
        # Check timestamp format
        if 'timestamp' in df.columns:
            try:
                pd.to_datetime(df['timestamp'])
            except:
                issues.append("Invalid timestamp format")
        
        return len(issues) == 0, issues
    
    def clean_transaction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize transaction data"""
        logger.info("🧹 Cleaning transaction data...")
        
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        # Convert timestamp to datetime
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
        
        # Remove duplicates based on transaction_id
        df_clean = df_clean.drop_duplicates(subset=['transaction_id'])
        duplicates_removed = initial_rows - len(df_clean)
        
        # Clean amount field
        df_clean['amount'] = pd.to_numeric(df_clean['amount'], errors='coerce')
        df_clean = df_clean[df_clean['amount'] > 0]  # Remove negative/zero amounts
        
        # Standardize payment methods
        payment_method_mapping = {
            'card': 'credit_card',
            'cc': 'credit_card',
            'creditcard': 'credit_card',
            'debit': 'debit_card',
            'paypal': 'paypal',
            'apple pay': 'apple_pay',
            'google pay': 'google_pay',
            'googlepay': 'google_pay',
            'bank': 'bank_transfer',
            'wire': 'wire_transfer',
            'crypto': 'cryptocurrency',
            'bitcoin': 'cryptocurrency',
            'cash': 'cash'
        }
        
        df_clean['payment_method'] = df_clean['payment_method'].str.lower().str.strip()
        df_clean['payment_method'] = df_clean['payment_method'].replace(payment_method_mapping)
        
        # Standardize country codes
        if 'country' in df_clean.columns:
            df_clean['country'] = df_clean['country'].str.upper().str.strip()
        
        # Clean merchant IDs
        df_clean['merchant_id'] = df_clean['merchant_id'].str.strip()
        df_clean['user_id'] = df_clean['user_id'].str.strip()
        
        # Add processing metadata
        df_clean['processed_at'] = datetime.utcnow()
        df_clean['data_source'] = self.config.get('data_source', 'batch_processing')
        
        logger.info(f"   Removed {duplicates_removed} duplicate records")
        logger.info(f"   Final dataset: {len(df_clean):,} records")
        
        return df_clean
    
    def enrich_transaction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich transaction data with additional features"""
        logger.info("✨ Enriching transaction data...")
        
        df_enriched = df.copy()
        
        # Time-based features
        df_enriched['hour'] = df_enriched['timestamp'].dt.hour
        df_enriched['day_of_week'] = df_enriched['timestamp'].dt.dayofweek
        df_enriched['day_of_month'] = df_enriched['timestamp'].dt.day
        df_enriched['month'] = df_enriched['timestamp'].dt.month
        df_enriched['quarter'] = df_enriched['timestamp'].dt.quarter
        df_enriched['year'] = df_enriched['timestamp'].dt.year
        
        # Business hours indicators
        df_enriched['is_weekend'] = (df_enriched['day_of_week'] >= 5).astype(int)
        df_enriched['is_business_hours'] = (
            (df_enriched['hour'] >= 9) & (df_enriched['hour'] <= 17)
        ).astype(int)
        df_enriched['is_late_night'] = (
            (df_enriched['hour'] >= 22) | (df_enriched['hour'] <= 6)
        ).astype(int)
        
        # Amount-based features
        df_enriched['amount_log'] = np.log1p(df_enriched['amount'])
        df_enriched['is_round_amount'] = (df_enriched['amount'] % 1 == 0).astype(int)
        df_enriched['is_round_10'] = (df_enriched['amount'] % 10 == 0).astype(int)
        df_enriched['is_round_100'] = (df_enriched['amount'] % 100 == 0).astype(int)
        
        # Payment method risk scores (based on historical fraud rates)
        payment_risk_scores = {
            'credit_card': 0.3,
            'debit_card': 0.2,
            'paypal': 0.15,
            'apple_pay': 0.1,
            'google_pay': 0.1,
            'bank_transfer': 0.05,
            'wire_transfer': 0.08,
            'cryptocurrency': 0.8,
            'cash': 0.01
        }
        
        df_enriched['payment_method_risk'] = df_enriched['payment_method'].map(
            payment_risk_scores
        ).fillna(0.5)  # Default risk for unknown payment methods
        
        # Transaction sequence features (within user)
        df_sorted = df_enriched.sort_values(['user_id', 'timestamp'])
        df_sorted['transaction_sequence'] = df_sorted.groupby('user_id').cumcount() + 1
        
        # Time since last transaction (in hours)
        df_sorted['time_since_last_transaction'] = (
            df_sorted.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600
        ).fillna(0)
        
        # Velocity features (transactions per hour)
        df_sorted['velocity_1h'] = df_sorted.groupby('user_id').apply(
            lambda x: x.set_index('timestamp').rolling('1H')['transaction_id'].count()
        ).reset_index(level=0, drop=True)
        
        df_sorted['velocity_24h'] = df_sorted.groupby('user_id').apply(
            lambda x: x.set_index('timestamp').rolling('24H')['transaction_id'].count()
        ).reset_index(level=0, drop=True)
        
        # Amount percentile within merchant
        df_sorted['amount_percentile_merchant'] = df_sorted.groupby('merchant_id')['amount'].rank(pct=True)
        
        # Restore original order
        df_enriched = df_sorted.sort_index()
        
        logger.info(f"   Added {df_enriched.shape[1] - df.shape[1]} new features")
        
        return df_enriched
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect basic anomalies in transaction data"""
        logger.info("🚨 Detecting transaction anomalies...")
        
        df_with_flags = df.copy()
        anomaly_flags = []
        
        # High amount transactions (99th percentile)
        amount_threshold = df['amount'].quantile(0.99)
        high_amount_flag = df['amount'] > amount_threshold
        df_with_flags['anomaly_high_amount'] = high_amount_flag.astype(int)
        anomaly_flags.append('anomaly_high_amount')
        
        # Off-hours transactions
        df_with_flags['anomaly_off_hours'] = (
            (df['hour'] < 6) | (df['hour'] > 22)
        ).astype(int)
        anomaly_flags.append('anomaly_off_hours')
        
        # High velocity users (more than 10 transactions in 1 hour)
        high_velocity_flag = df_with_flags['velocity_1h'] > 10
        df_with_flags['anomaly_high_velocity'] = high_velocity_flag.astype(int)
        anomaly_flags.append('anomaly_high_velocity')
        
        # Rapid succession transactions (less than 1 minute apart)
        rapid_succession_flag = df_with_flags['time_since_last_transaction'] < (1/60)  # 1 minute
        df_with_flags['anomaly_rapid_succession'] = rapid_succession_flag.astype(int)
        anomaly_flags.append('anomaly_rapid_succession')
        
        # High-risk payment methods
        high_risk_payment_flag = df_with_flags['payment_method_risk'] > 0.5
        df_with_flags['anomaly_high_risk_payment'] = high_risk_payment_flag.astype(int)
        anomaly_flags.append('anomaly_high_risk_payment')
        
        # Overall anomaly score (sum of all flags)
        df_with_flags['anomaly_score'] = df_with_flags[anomaly_flags].sum(axis=1)
        df_with_flags['is_anomaly'] = (df_with_flags['anomaly_score'] >= 2).astype(int)
        
        anomalies_detected = df_with_flags['is_anomaly'].sum()
        logger.info(f"   Detected {anomalies_detected:,} anomalous transactions ({anomalies_detected/len(df)*100:.2f}%)")
        
        return df_with_flags
    
    def save_processed_data(self, df: pd.DataFrame, filename_suffix: str = "") -> str:
        """Save processed data to output path"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_transactions_{timestamp}{filename_suffix}.parquet"
        output_file = self.output_path / filename
        
        # Save as parquet for efficiency
        df.to_parquet(output_file, index=False, compression='snappy')
        
        logger.info(f"💾 Saved processed data: {output_file}")
        logger.info(f"   Records: {len(df):,}")
        logger.info(f"   Columns: {df.shape[1]}")
        logger.info(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Also save metadata
        metadata = {
            'filename': str(output_file),
            'records': len(df),
            'columns': df.shape[1],
            'processing_time': timestamp,
            'column_names': df.columns.tolist(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'anomalies_detected': int(df.get('is_anomaly', pd.Series([0])).sum()),
            'date_range': {
                'start': str(df['timestamp'].min()),
                'end': str(df['timestamp'].max())
            }
        }
        
        metadata_file = output_file.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return str(output_file)
    
    def process_batch(self, input_file: str) -> bool:
        """Process a single batch of transaction data"""
        try:
            logger.info(f"📁 Processing file: {input_file}")
            
            # Load data
            if input_file.endswith('.csv'):
                df = pd.read_csv(input_file)
            elif input_file.endswith('.parquet'):
                df = pd.read_parquet(input_file)
            elif input_file.endswith('.json'):
                df = pd.read_json(input_file)
            else:
                logger.error(f"Unsupported file format: {input_file}")
                return False
            
            logger.info(f"   Loaded {len(df):,} records with {df.shape[1]} columns")
            
            # Validate input data
            is_valid, issues = self.validate_input_data(df)
            if not is_valid:
                logger.error(f"Data validation failed: {issues}")
                return False
            
            # Process data pipeline
            df_cleaned = self.clean_transaction_data(df)
            df_enriched = self.enrich_transaction_data(df_cleaned)
            df_final = self.detect_anomalies(df_enriched)
            
            # Save processed data
            filename_suffix = f"_{Path(input_file).stem}"
            output_file = self.save_processed_data(df_final, filename_suffix)
            
            # Generate processing report
            self.generate_processing_report(df, df_final, input_file, output_file)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to process {input_file}: {str(e)}")
            return False
    
    def generate_processing_report(self, df_original: pd.DataFrame, 
                                 df_processed: pd.DataFrame,
                                 input_file: str, output_file: str):
        """Generate processing report"""
        report = {
            'input_file': input_file,
            'output_file': output_file,
            'processing_timestamp': datetime.utcnow().isoformat(),
            'original_records': len(df_original),
            'processed_records': len(df_processed),
            'records_removed': len(df_original) - len(df_processed),
            'removal_rate': (len(df_original) - len(df_processed)) / len(df_original) * 100,
            'features_added': df_processed.shape[1] - df_original.shape[1],
            'anomalies_detected': int(df_processed.get('is_anomaly', pd.Series([0])).sum()),
            'anomaly_rate': float(df_processed.get('is_anomaly', pd.Series([0])).mean() * 100),
            'data_quality': {
                'missing_values': df_processed.isnull().sum().sum(),
                'duplicate_transactions': df_processed['transaction_id'].duplicated().sum(),
                'date_range': {
                    'start': str(df_processed['timestamp'].min()),
                    'end': str(df_processed['timestamp'].max()),
                    'span_days': (df_processed['timestamp'].max() - df_processed['timestamp'].min()).days
                }
            },
            'feature_summary': {
                'amount_stats': {
                    'mean': float(df_processed['amount'].mean()),
                    'median': float(df_processed['amount'].median()),
                    'std': float(df_processed['amount'].std()),
                    'min': float(df_processed['amount'].min()),
                    'max': float(df_processed['amount'].max())
                },
                'payment_methods': df_processed['payment_method'].value_counts().to_dict(),
                'top_merchants': df_processed['merchant_id'].value_counts().head(10).to_dict()
            }
        }
        
        report_file = Path(output_file).with_suffix('.report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Processing report saved: {report_file}")
        
        # Log summary
        logger.info(f"📈 Processing Summary:")
        logger.info(f"   Records processed: {report['processed_records']:,}")
        logger.info(f"   Records removed: {report['records_removed']:,} ({report['removal_rate']:.2f}%)")
        logger.info(f"   Features added: {report['features_added']}")
        logger.info(f"   Anomalies detected: {report['anomalies_detected']:,} ({report['anomaly_rate']:.2f}%)")
    
    def process_all_files(self) -> bool:
        """Process all files in input directory"""
        logger.info(f"🔄 Starting batch processing from: {self.input_path}")
        
        # Find all supported files
        supported_extensions = ['.csv', '.parquet', '.json']
        input_files = []
        
        for ext in supported_extensions:
            input_files.extend(list(self.input_path.glob(f"*{ext}")))
        
        if not input_files:
            logger.warning("No supported files found in input directory")
            return False
        
        logger.info(f"Found {len(input_files)} files to process")
        
        # Process each file
        success_count = 0
        for input_file in input_files:
            if self.process_batch(str(input_file)):
                success_count += 1
            else:
                logger.error(f"Failed to process: {input_file}")
        
        logger.info(f"🎉 Batch processing completed!")
        logger.info(f"   Successfully processed: {success_count}/{len(input_files)} files")
        
        return success_count == len(input_files)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Process transaction data for fraud detection")
    parser.add_argument('--input', required=True, help='Input directory or file path')
    parser.add_argument('--output', required=True, help='Output directory path')
    parser.add_argument('--batch-size', type=int, default=10000, help='Batch size for processing')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--data-source', default='batch_processing', help='Data source identifier')
    
    args = parser.parse_args()
    
    config = {
        'input_path': args.input,
        'output_path': args.output,
        'batch_size': args.batch_size,
        'parallel_workers': args.workers,
        'data_source': args.data_source
    }
    
    processor = TransactionProcessor(config)
    success = processor.process_all_files()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
