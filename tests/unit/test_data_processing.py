"""
Unit Tests for Data Processing Modules
Test data validation, cleaning, and feature engineering
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.data_processing import DataProcessor, DataValidator, FeatureEngineer
from core.exceptions import DataValidationError, ProcessingError


class TestDataValidator:
    """Test data validation functionality"""
    
    @pytest.mark.unit
    def test_validate_required_columns(self, sample_transaction_data):
        """Test validation of required columns"""
        validator = DataValidator()
        
        # Valid data should pass
        is_valid, errors = validator.validate_schema(sample_transaction_data)
        assert is_valid
        assert len(errors) == 0
        
        # Missing required column should fail
        invalid_data = sample_transaction_data.drop(columns=['transaction_id'])
        is_valid, errors = validator.validate_schema(invalid_data)
        assert not is_valid
        assert any('transaction_id' in error for error in errors)
    
    @pytest.mark.unit
    def test_validate_data_types(self, sample_transaction_data):
        """Test data type validation"""
        validator = DataValidator()
        
        # Invalid amount type should fail
        invalid_data = sample_transaction_data.copy()
        invalid_data['amount'] = ['invalid', 'data', 'type'] * (len(invalid_data) // 3 + 1)
        invalid_data = invalid_data.iloc[:len(sample_transaction_data)]
        
        is_valid, errors = validator.validate_data_types(invalid_data)
        assert not is_valid
        assert any('amount' in error for error in errors)
    
    @pytest.mark.unit
    def test_validate_business_rules(self, sample_transaction_data):
        """Test business rule validation"""
        validator = DataValidator()
        
        # Negative amounts should fail
        invalid_data = sample_transaction_data.copy()
        invalid_data.loc[0, 'amount'] = -100
        
        is_valid, errors = validator.validate_business_rules(invalid_data)
        assert not is_valid
        assert any('negative amount' in error.lower() for error in errors)
    
    @pytest.mark.unit
    def test_validate_duplicates(self, sample_transaction_data):
        """Test duplicate detection"""
        validator = DataValidator()
        
        # Add duplicate transaction
        duplicate_data = pd.concat([sample_transaction_data, sample_transaction_data.iloc[[0]]])
        
        is_valid, errors = validator.validate_duplicates(duplicate_data)
        assert not is_valid
        assert any('duplicate' in error.lower() for error in errors)


class TestDataProcessor:
    """Test data processing functionality"""
    
    @pytest.mark.unit
    def test_clean_transaction_data(self, sample_transaction_data):
        """Test transaction data cleaning"""
        processor = DataProcessor()
        
        # Add some dirty data
        dirty_data = sample_transaction_data.copy()
        dirty_data.loc[0, 'amount'] = -50  # Negative amount
        dirty_data.loc[1, 'payment_method'] = 'CARD'  # Non-standard format
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[[0]]])  # Add duplicate
        
        cleaned_data = processor.clean_data(dirty_data)
        
        # Should remove negative amounts
        assert all(cleaned_data['amount'] > 0)
        
        # Should standardize payment methods
        assert 'CARD' not in cleaned_data['payment_method'].values
        
        # Should remove duplicates
        assert len(cleaned_data) < len(dirty_data)
        
        # Should not have any nulls in critical columns
        critical_columns = ['transaction_id', 'user_id', 'amount']
        for col in critical_columns:
            assert not cleaned_data[col].isnull().any()
    
    @pytest.mark.unit
    def test_standardize_payment_methods(self, sample_transaction_data):
        """Test payment method standardization"""
        processor = DataProcessor()
        
        # Add various payment method formats
        test_data = sample_transaction_data.copy()
        test_data['payment_method'] = ['card', 'CC', 'PayPal', 'apple pay', 'DEBIT'][:len(test_data)]
        
        standardized_data = processor.standardize_payment_methods(test_data)
        
        expected_methods = ['credit_card', 'credit_card', 'paypal', 'apple_pay', 'debit_card']
        for i, expected in enumerate(expected_methods[:len(test_data)]):
            if i < len(standardized_data):
                assert standardized_data.iloc[i]['payment_method'] == expected
    
    @pytest.mark.unit
    def test_handle_missing_values(self):
        """Test missing value handling"""
        processor = DataProcessor()
        
        # Create data with missing values
        data = pd.DataFrame({
            'transaction_id': ['T1', 'T2', 'T3'],
            'amount': [100, np.nan, 200],
            'country': ['US', None, 'UK'],
            'is_fraud': [0, 1, 0]
        })
        
        processed_data = processor.handle_missing_values(data)
        
        # Should not have any missing values
        assert not processed_data.isnull().any().any()
        
        # Amount should be filled with median
        assert processed_data.loc[1, 'amount'] == 150  # median of 100, 200
    
    @pytest.mark.unit
    def test_process_batch_size_limit(self, performance_test_data):
        """Test processing with batch size limits"""
        processor = DataProcessor(batch_size=1000)
        
        # Should handle large datasets by batching
        processed_data = processor.process_data(performance_test_data)
        
        assert len(processed_data) <= len(performance_test_data)  # May remove some invalid records
        assert processed_data is not None
        assert isinstance(processed_data, pd.DataFrame)


class TestFeatureEngineer:
    """Test feature engineering functionality"""
    
    @pytest.mark.unit
    def test_create_time_features(self, sample_transaction_data):
        """Test time-based feature creation"""
        engineer = FeatureEngineer()
        
        featured_data = engineer.create_time_features(sample_transaction_data)
        
        # Should have time-based columns
        expected_time_features = ['hour', 'day_of_week', 'month', 'quarter', 
                                'is_weekend', 'is_business_hours']
        
        for feature in expected_time_features:
            assert feature in featured_data.columns
        
        # Weekend flag should be correct
        weekend_mask = featured_data['day_of_week'] >= 5
        assert (featured_data.loc[weekend_mask, 'is_weekend'] == 1).all()
        
        # Business hours should be 9-17
        business_mask = (featured_data['hour'] >= 9) & (featured_data['hour'] <= 17)
        assert (featured_data.loc[business_mask, 'is_business_hours'] == 1).all()
    
    @pytest.mark.unit
    def test_create_amount_features(self, sample_transaction_data):
        """Test amount-based feature creation"""
        engineer = FeatureEngineer()
        
        featured_data = engineer.create_amount_features(sample_transaction_data)
        
        # Should have amount-based features
        expected_amount_features = ['amount_log', 'is_round_amount', 'amount_z_score']
        
        for feature in expected_amount_features:
            assert feature in featured_data.columns
        
        # Log transform should be correct
        assert np.allclose(featured_data['amount_log'], np.log1p(featured_data['amount']))
        
        # Round amount detection should work
        round_amounts = featured_data[featured_data['amount'] % 1 == 0]
        if len(round_amounts) > 0:
            assert (round_amounts['is_round_amount'] == 1).all()
    
    @pytest.mark.unit
    def test_create_velocity_features(self, sample_transaction_data):
        """Test velocity feature creation"""
        engineer = FeatureEngineer()
        
        # Sort by user and timestamp for realistic velocity calculation
        sorted_data = sample_transaction_data.sort_values(['user_id', 'timestamp'])
        
        featured_data = engineer.create_velocity_features(sorted_data)
        
        # Should have velocity features
        expected_velocity_features = ['velocity_1h', 'velocity_24h', 'transaction_count']
        
        for feature in expected_velocity_features:
            assert feature in featured_data.columns
        
        # Velocity should be non-negative integers
        assert (featured_data['velocity_1h'] >= 0).all()
        assert (featured_data['velocity_24h'] >= 0).all()
        assert featured_data['velocity_1h'].dtype in [int, 'int64']
    
    @pytest.mark.unit
    def test_create_user_features(self, sample_transaction_data):
        """Test user-based feature creation"""
        engineer = FeatureEngineer()
        
        featured_data = engineer.create_user_features(sample_transaction_data)
        
        # Should have user-based features
        expected_user_features = ['user_transaction_count', 'user_avg_amount', 
                                'days_since_first_transaction']
        
        for feature in expected_user_features:
            assert feature in featured_data.columns
        
        # User transaction count should be positive
        assert (featured_data['user_transaction_count'] > 0).all()
        
        # Average amount should be reasonable
        assert (featured_data['user_avg_amount'] > 0).all()
    
    @pytest.mark.unit
    def test_feature_selection(self, sample_transaction_data):
        """Test automatic feature selection"""
        engineer = FeatureEngineer()
        
        # Create features first
        featured_data = engineer.create_all_features(sample_transaction_data)
        
        # Select top features
        selected_features = engineer.select_top_features(featured_data, target_col='is_fraud', k=10)
        
        assert len(selected_features) <= 10
        assert 'is_fraud' not in selected_features  # Target should not be included
        assert all(col in featured_data.columns for col in selected_features)
    
    @pytest.mark.unit
    def test_handle_categorical_features(self, sample_transaction_data):
        """Test categorical feature handling"""
        engineer = FeatureEngineer()
        
        featured_data = engineer.encode_categorical_features(sample_transaction_data)
        
        # Payment method should be encoded
        if 'payment_method_encoded' in featured_data.columns:
            assert featured_data['payment_method_encoded'].dtype in ['int64', int]
        
        # Original categorical columns might be kept or removed
        # Just ensure the process completes without error


class TestDataProcessingIntegration:
    """Integration tests for complete data processing pipeline"""
    
    @pytest.mark.unit
    def test_complete_processing_pipeline(self, sample_transaction_data):
        """Test complete data processing pipeline"""
        validator = DataValidator()
        processor = DataProcessor()
        engineer = FeatureEngineer()
        
        # Validate
        is_valid, errors = validator.validate_complete(sample_transaction_data)
        if not is_valid:
            pytest.skip(f"Sample data not valid: {errors}")
        
        # Clean
        cleaned_data = processor.clean_data(sample_transaction_data)
        
        # Engineer features
        featured_data = engineer.create_all_features(cleaned_data)
        
        # Final validation
        assert len(featured_data) > 0
        assert featured_data.shape[1] > sample_transaction_data.shape[1]  # More columns
        assert not featured_data.isnull().any().any()  # No missing values
        assert 'is_fraud' in featured_data.columns  # Target preserved
    
    @pytest.mark.unit
    def test_error_handling(self):
        """Test error handling in data processing"""
        processor = DataProcessor()
        
        # Empty DataFrame
        with pytest.raises(ProcessingError):
            processor.process_data(pd.DataFrame())
        
        # Invalid data
        invalid_data = pd.DataFrame({'invalid': ['data']})
        with pytest.raises((ProcessingError, DataValidationError)):
            processor.process_data(invalid_data)
    
    @pytest.mark.unit
    def test_processing_metadata(self, sample_transaction_data):
        """Test processing metadata generation"""
        processor = DataProcessor()
        
        result = processor.process_data_with_metadata(sample_transaction_data)
        
        assert 'data' in result
        assert 'metadata' in result
        
        metadata = result['metadata']
        assert 'original_rows' in metadata
        assert 'processed_rows' in metadata
        assert 'features_added' in metadata
        assert 'processing_time' in metadata
        assert 'data_quality_score' in metadata


# Performance tests for data processing
class TestDataProcessingPerformance:
    """Performance tests for data processing"""
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_large_dataset_processing_time(self, performance_test_data):
        """Test processing time for large datasets"""
        import time
        
        processor = DataProcessor()
        
        start_time = time.time()
        processed_data = processor.process_data(performance_test_data)
        processing_time = time.time() - start_time
        
        # Should process 10k records in reasonable time (less than 30 seconds)
        assert processing_time < 30
        assert len(processed_data) > 0
        
        # Memory usage should be reasonable
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb < 500  # Less than 500MB
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_feature_engineering_performance(self, performance_test_data):
        """Test feature engineering performance"""
        import time
        
        engineer = FeatureEngineer()
        
        start_time = time.time()
        featured_data = engineer.create_all_features(performance_test_data)
        feature_time = time.time() - start_time
        
        # Should create features in reasonable time
        assert feature_time < 60  # Less than 1 minute
        assert featured_data.shape[1] > performance_test_data.shape[1]
