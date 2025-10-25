"""
Performance Tests for Fraud Detection Models
Test model training and prediction performance under load
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import Mock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ml.fraud_detector import FraudDetector
from ml.model_training import ModelTrainer
from core.data_processing import DataProcessor, FeatureEngineer


class TestModelTrainingPerformance:
    """Test model training performance"""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_training_time_with_large_dataset(self, performance_test_data):
        """Test training time with large dataset"""
        trainer = ModelTrainer()
        
        # Prepare data
        X, y = trainer.prepare_training_data(performance_test_data, target_col='is_fraud')
        
        # Test different model types
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        import xgboost as xgb
        
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss')
        }
        
        training_times = {}
        
        for model_name, model in models.items():
            start_time = time.time()
            trained_model, metrics = trainer.train_model(model, X, y)
            training_time = time.time() - start_time
            
            training_times[model_name] = training_time
            
            # Performance requirements
            assert training_time < 300  # Less than 5 minutes
            assert trained_model is not None
            assert metrics['accuracy'] > 0.5  # Basic sanity check
        
        # Log training times
        print(f"\nTraining times for {len(X)} samples:")
        for model_name, time_taken in training_times.items():
            print(f"  {model_name}: {time_taken:.2f} seconds")
    
    @pytest.mark.performance
    def test_memory_usage_during_training(self, performance_test_data):
        """Test memory usage during model training"""
        trainer = ModelTrainer()
        
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        X, y = trainer.prepare_training_data(performance_test_data, target_col='is_fraud')
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        
        # Train model and monitor memory
        start_time = time.time()
        trained_model, metrics = trainer.train_model(model, X, y)
        training_time = time.time() - start_time
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory usage:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")
        print(f"  Training time: {training_time:.2f} seconds")
        
        # Memory requirements
        assert memory_increase < 1000  # Less than 1GB increase
        assert final_memory < 2000  # Total memory less than 2GB
    
    @pytest.mark.performance
    def test_cross_validation_performance(self, sample_transaction_data):
        """Test cross-validation performance"""
        trainer = ModelTrainer()
        X, y = trainer.prepare_training_data(sample_transaction_data, target_col='is_fraud')
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        # Test different CV fold numbers
        cv_folds = [3, 5, 10]
        cv_times = {}
        
        for cv in cv_folds:
            start_time = time.time()
            cv_scores = trainer.cross_validate(model, X, y, cv=cv)
            cv_time = time.time() - start_time
            
            cv_times[cv] = cv_time
            
            assert cv_time < 120  # Less than 2 minutes
            assert cv_scores['mean_score'] > 0.5
        
        print(f"\nCross-validation times:")
        for cv, time_taken in cv_times.items():
            print(f"  {cv}-fold CV: {time_taken:.2f} seconds")


class TestPredictionPerformance:
    """Test model prediction performance"""
    
    @pytest.mark.performance
    def test_single_prediction_latency(self, mock_model):
        """Test single prediction latency"""
        detector = FraudDetector(model=mock_model)
        
        # Mock prediction
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        transaction = pd.DataFrame({
            'amount': [500],
            'hour': [23],
            'day_of_week': [5]
        })
        
        # Warm up
        detector.predict(transaction)
        
        # Measure latency
        latencies = []
        for _ in range(100):
            start_time = time.perf_counter()
            result = detector.predict(transaction)
            latency = time.perf_counter() - start_time
            latencies.append(latency * 1000)  # Convert to ms
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"\nSingle prediction latency:")
        print(f"  Average: {avg_latency:.2f} ms")
        print(f"  95th percentile: {p95_latency:.2f} ms") 
        print(f"  99th percentile: {p99_latency:.2f} ms")
        
        # Latency requirements
        assert avg_latency < 50  # Average under 50ms
        assert p95_latency < 100  # 95th percentile under 100ms
        assert p99_latency < 200  # 99th percentile under 200ms
    
    @pytest.mark.performance
    def test_batch_prediction_throughput(self, mock_model, performance_test_data):
        """Test batch prediction throughput"""
        detector = FraudDetector(model=mock_model)
        
        # Prepare batch data
        batch_size = 1000
        batch_data = performance_test_data.head(batch_size)
        
        # Mock batch predictions
        mock_model.predict_proba.return_value = np.random.rand(batch_size, 2)
        mock_model.predict_proba.return_value[:, 1] = np.random.rand(batch_size)  # Fraud probabilities
        
        # Measure throughput
        start_time = time.time()
        results = detector.predict_batch(batch_data)
        total_time = time.time() - start_time
        
        throughput = batch_size / total_time
        
        print(f"\nBatch prediction throughput:")
        print(f"  Batch size: {batch_size}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Throughput: {throughput:.1f} predictions/second")
        
        # Throughput requirements
        assert throughput > 100  # At least 100 predictions/second
        assert len(results) == batch_size
        assert all('fraud_probability' in result for result in results)
    
    @pytest.mark.performance
    def test_concurrent_predictions(self, mock_model):
        """Test concurrent prediction performance"""
        detector = FraudDetector(model=mock_model)
        
        # Mock prediction
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        def make_prediction():
            transaction = pd.DataFrame({
                'amount': [np.random.randint(10, 1000)],
                'hour': [np.random.randint(0, 24)],
                'day_of_week': [np.random.randint(0, 7)]
            })
            return detector.predict(transaction)
        
        # Test different numbers of concurrent workers
        worker_counts = [1, 5, 10, 20]
        results = {}
        
        for num_workers in worker_counts:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(make_prediction) for _ in range(100)]
                predictions = [future.result() for future in futures]
            
            total_time = time.time() - start_time
            throughput = 100 / total_time
            
            results[num_workers] = {
                'time': total_time,
                'throughput': throughput,
                'predictions': len(predictions)
            }
            
            assert len(predictions) == 100
            assert all('fraud_probability' in pred for pred in predictions)
        
        print(f"\nConcurrent prediction performance:")
        for workers, metrics in results.items():
            print(f"  {workers} workers: {metrics['throughput']:.1f} predictions/sec ({metrics['time']:.2f}s)")
    
    @pytest.mark.performance
    def test_memory_usage_during_prediction(self, mock_model, performance_test_data):
        """Test memory usage during batch predictions"""
        detector = FraudDetector(model=mock_model)
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Prepare large batch
        batch_size = 5000
        large_batch = performance_test_data.head(batch_size)
        
        # Mock predictions
        mock_model.predict_proba.return_value = np.random.rand(batch_size, 2)
        
        # Make predictions and monitor memory
        start_time = time.time()
        results = detector.predict_batch(large_batch)
        prediction_time = time.time() - start_time
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"\nBatch prediction memory usage:")
        print(f"  Batch size: {batch_size}")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Memory increase: {memory_increase:.1f} MB")
        print(f"  Prediction time: {prediction_time:.2f} seconds")
        
        # Memory requirements
        assert memory_increase < 500  # Less than 500MB increase
        assert len(results) == batch_size


class TestDataProcessingPerformance:
    """Test data processing performance"""
    
    @pytest.mark.performance
    def test_data_cleaning_performance(self, performance_test_data):
        """Test data cleaning performance with large dataset"""
        processor = DataProcessor()
        
        # Add some dirty data
        dirty_data = performance_test_data.copy()
        
        # Add duplicates
        dirty_data = pd.concat([dirty_data, dirty_data.sample(100)])
        
        # Add missing values
        dirty_data.loc[np.random.choice(dirty_data.index, 500), 'amount'] = np.nan
        
        print(f"Processing {len(dirty_data)} records...")
        
        start_time = time.time()
        cleaned_data = processor.clean_data(dirty_data)
        cleaning_time = time.time() - start_time
        
        processing_rate = len(dirty_data) / cleaning_time
        
        print(f"\nData cleaning performance:")
        print(f"  Input records: {len(dirty_data)}")
        print(f"  Output records: {len(cleaned_data)}")
        print(f"  Cleaning time: {cleaning_time:.2f} seconds")
        print(f"  Processing rate: {processing_rate:.1f} records/second")
        
        # Performance requirements
        assert cleaning_time < 60  # Less than 1 minute
        assert processing_rate > 100  # At least 100 records/second
        assert len(cleaned_data) <= len(dirty_data)  # Should remove some records
    
    @pytest.mark.performance
    def test_feature_engineering_performance(self, performance_test_data):
        """Test feature engineering performance"""
        engineer = FeatureEngineer()
        
        print(f"Engineering features for {len(performance_test_data)} records...")
        
        start_time = time.time()
        featured_data = engineer.create_all_features(performance_test_data)
        engineering_time = time.time() - start_time
        
        processing_rate = len(performance_test_data) / engineering_time
        features_added = featured_data.shape[1] - performance_test_data.shape[1]
        
        print(f"\nFeature engineering performance:")
        print(f"  Input records: {len(performance_test_data)}")
        print(f"  Features added: {features_added}")
        print(f"  Engineering time: {engineering_time:.2f} seconds")
        print(f"  Processing rate: {processing_rate:.1f} records/second")
        
        # Performance requirements
        assert engineering_time < 120  # Less than 2 minutes
        assert processing_rate > 50  # At least 50 records/second
        assert features_added > 0  # Should add features
        assert not featured_data.isnull().any().any()  # No missing values
    
    @pytest.mark.performance
    def test_parallel_data_processing(self, performance_test_data):
        """Test parallel data processing performance"""
        processor = DataProcessor()
        
        # Split data into chunks
        chunk_size = 2000
        chunks = [performance_test_data[i:i+chunk_size] 
                 for i in range(0, len(performance_test_data), chunk_size)]
        
        print(f"Processing {len(chunks)} chunks of size {chunk_size}...")
        
        # Serial processing
        start_time = time.time()
        serial_results = [processor.clean_data(chunk) for chunk in chunks]
        serial_time = time.time() - start_time
        
        # Parallel processing
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=4) as executor:
            parallel_results = list(executor.map(processor.clean_data, chunks))
        parallel_time = time.time() - start_time
        
        speedup = serial_time / parallel_time if parallel_time > 0 else 1
        
        print(f"\nParallel processing performance:")
        print(f"  Serial time: {serial_time:.2f} seconds")
        print(f"  Parallel time: {parallel_time:.2f} seconds")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Verify results are equivalent
        serial_total = sum(len(chunk) for chunk in serial_results)
        parallel_total = sum(len(chunk) for chunk in parallel_results)
        
        assert abs(serial_total - parallel_total) < 100  # Small difference acceptable
        assert speedup > 1.2  # At least 20% speedup
        assert parallel_time < serial_time


class TestSystemLoadPerformance:
    """Test system performance under various load conditions"""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_sustained_load_performance(self, mock_model):
        """Test performance under sustained load"""
        detector = FraudDetector(model=mock_model)
        
        # Mock prediction
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        def make_predictions(duration_seconds=60):
            """Make predictions for a specified duration"""
            end_time = time.time() + duration_seconds
            predictions_made = 0
            
            while time.time() < end_time:
                transaction = pd.DataFrame({
                    'amount': [np.random.randint(10, 1000)],
                    'hour': [np.random.randint(0, 24)]
                })
                
                result = detector.predict(transaction)
                predictions_made += 1
                
                if predictions_made % 100 == 0:
                    # Small delay to simulate realistic load
                    time.sleep(0.01)
            
            return predictions_made
        
        print("Running sustained load test for 30 seconds...")
        
        start_time = time.time()
        predictions_made = make_predictions(30)  # 30 second test
        actual_duration = time.time() - start_time
        
        avg_throughput = predictions_made / actual_duration
        
        print(f"\nSustained load performance:")
        print(f"  Duration: {actual_duration:.1f} seconds")
        print(f"  Predictions made: {predictions_made}")
        print(f"  Average throughput: {avg_throughput:.1f} predictions/second")
        
        # Performance requirements
        assert avg_throughput > 50  # At least 50 predictions/second sustained
        assert predictions_made > 1000  # Should make substantial number of predictions
    
    @pytest.mark.performance
    def test_memory_leak_detection(self, mock_model):
        """Test for memory leaks during extended operation"""
        detector = FraudDetector(model=mock_model)
        
        # Mock prediction
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        process = psutil.Process(os.getpid())
        
        # Take initial memory measurement
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make many predictions
        for i in range(1000):
            transaction = pd.DataFrame({
                'amount': [np.random.randint(10, 1000)],
                'hour': [np.random.randint(0, 24)]
            })
            
            result = detector.predict(transaction)
            
            # Measure memory periodically
            if i % 100 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                # Log memory usage
                if i % 500 == 0:
                    print(f"  After {i} predictions: {current_memory:.1f} MB ({memory_increase:+.1f} MB)")
        
        # Final memory measurement
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_increase = final_memory - initial_memory
        
        print(f"\nMemory leak detection:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Total increase: {total_increase:.1f} MB")
        
        # Memory leak detection (allowing for some reasonable growth)
        assert total_increase < 50  # Less than 50MB increase
    
    @pytest.mark.performance
    def test_cpu_usage_under_load(self, mock_model):
        """Test CPU usage during high load"""
        detector = FraudDetector(model=mock_model)
        
        # Mock prediction
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        # Monitor CPU usage
        process = psutil.Process(os.getpid())
        
        def cpu_monitoring():
            """Monitor CPU usage in background"""
            cpu_samples = []
            for _ in range(30):  # Sample for 30 seconds
                cpu_percent = process.cpu_percent(interval=1)
                cpu_samples.append(cpu_percent)
            return cpu_samples
        
        # Start CPU monitoring in background
        import threading
        cpu_samples = []
        
        def monitor_cpu():
            for _ in range(30):
                cpu_percent = process.cpu_percent(interval=1)
                cpu_samples.append(cpu_percent)
        
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Generate load
        predictions_made = 0
        start_time = time.time()
        
        while time.time() - start_time < 30:  # 30 seconds of load
            transaction = pd.DataFrame({
                'amount': [np.random.randint(10, 1000)],
                'hour': [np.random.randint(0, 24)]
            })
            
            result = detector.predict(transaction)
            predictions_made += 1
        
        # Wait for monitoring to complete
        monitor_thread.join()
        
        avg_cpu = np.mean(cpu_samples) if cpu_samples else 0
        max_cpu = np.max(cpu_samples) if cpu_samples else 0
        
        print(f"\nCPU usage under load:")
        print(f"  Predictions made: {predictions_made}")
        print(f"  Average CPU usage: {avg_cpu:.1f}%")
        print(f"  Maximum CPU usage: {max_cpu:.1f}%")
        
        # CPU usage requirements (allowing for test environment variations)
        assert max_cpu < 100  # Should not max out CPU completely
        assert avg_cpu > 0  # Should show some CPU usage
