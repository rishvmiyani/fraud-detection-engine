"""
Unit Tests for Fraud Detection Models
Test model training, prediction, and evaluation
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import joblib
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ml.fraud_detector import FraudDetector, ModelManager, PredictionService
from ml.model_training import ModelTrainer, HyperparameterTuner
from ml.evaluation import ModelEvaluator, PerformanceMetrics
from core.exceptions import ModelError, PredictionError


class TestFraudDetector:
    """Test fraud detection model functionality"""
    
    @pytest.mark.unit
    def test_fraud_detector_initialization(self, mock_model):
        """Test fraud detector initialization"""
        detector = FraudDetector(model=mock_model, threshold=0.5)
        
        assert detector.model is mock_model
        assert detector.threshold == 0.5
        assert detector.feature_columns is None
    
    @pytest.mark.unit
    def test_predict_single_transaction(self, mock_model, sample_transaction_data):
        """Test prediction for single transaction"""
        detector = FraudDetector(model=mock_model)
        
        # Single transaction
        transaction = sample_transaction_data.iloc[0:1]
        
        # Mock model predictions
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        result = detector.predict(transaction)
        
        assert 'fraud_probability' in result
        assert 'fraud_prediction' in result
        assert 'risk_level' in result
        assert result['fraud_probability'] == 0.7
        assert result['fraud_prediction'] == 1  # Above default threshold 0.5
    
    @pytest.mark.unit
    def test_predict_batch_transactions(self, mock_model, sample_transaction_data):
        """Test batch prediction"""
        detector = FraudDetector(model=mock_model)
        
        # Batch of transactions
        batch = sample_transaction_data.iloc[:5]
        
        # Mock model predictions
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.85, 0.15]
        ])
        
        results = detector.predict_batch(batch)
        
        assert len(results) == 5
        assert all('fraud_probability' in result for result in results)
        assert all('fraud_prediction' in result for result in results)
        assert all('risk_level' in result for result in results)
    
    @pytest.mark.unit
    def test_risk_level_classification(self, mock_model):
        """Test risk level classification"""
        detector = FraudDetector(model=mock_model)
        
        test_cases = [
            (0.1, 'low'),
            (0.3, 'medium'), 
            (0.7, 'high'),
            (0.9, 'critical')
        ]
        
        for probability, expected_risk in test_cases:
            risk_level = detector._classify_risk_level(probability)
            assert risk_level == expected_risk
    
    @pytest.mark.unit
    def test_feature_validation(self, mock_model, sample_transaction_data):
        """Test feature validation"""
        required_features = ['amount', 'hour', 'payment_method_risk']
        detector = FraudDetector(model=mock_model, feature_columns=required_features)
        
        # Valid features
        valid_data = pd.DataFrame({
            'amount': [100],
            'hour': [14],
            'payment_method_risk': [0.3]
        })
        
        assert detector._validate_features(valid_data) == True
        
        # Missing features
        invalid_data = pd.DataFrame({
            'amount': [100],
            'hour': [14]
            # missing payment_method_risk
        })
        
        with pytest.raises(PredictionError):
            detector.predict(invalid_data)
    
    @pytest.mark.unit
    def test_threshold_adjustment(self, mock_model):
        """Test dynamic threshold adjustment"""
        detector = FraudDetector(model=mock_model, threshold=0.5)
        
        # Mock predictions
        mock_model.predict_proba.return_value = np.array([[0.4, 0.6]])
        
        # Default threshold
        result = detector.predict(pd.DataFrame({'feature': [1]}))
        assert result['fraud_prediction'] == 1  # 0.6 > 0.5
        
        # Higher threshold
        detector.set_threshold(0.8)
        result = detector.predict(pd.DataFrame({'feature': [1]}))
        assert result['fraud_prediction'] == 0  # 0.6 < 0.8
    
    @pytest.mark.unit
    def test_prediction_explanation(self, mock_model):
        """Test prediction explanation generation"""
        detector = FraudDetector(model=mock_model, explain_predictions=True)
        
        # Mock feature importance
        mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        
        transaction = pd.DataFrame({
            'amount': [500],
            'hour': [23],
            'velocity': [5]
        })
        
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        
        result = detector.predict(transaction)
        
        assert 'explanation' in result
        assert 'risk_factors' in result['explanation']
        assert len(result['explanation']['risk_factors']) > 0


class TestModelManager:
    """Test model management functionality"""
    
    @pytest.mark.unit
    def test_load_model(self, temp_dir, mock_model):
        """Test model loading"""
        # Save a model first
        model_path = temp_dir / "test_model.pkl"
        joblib.dump(mock_model, model_path)
        
        manager = ModelManager()
        loaded_model = manager.load_model(str(model_path))
        
        assert loaded_model is not None
    
    @pytest.mark.unit
    def test_save_model(self, temp_dir, mock_model):
        """Test model saving"""
        manager = ModelManager()
        
        model_path = temp_dir / "saved_model.pkl"
        manager.save_model(mock_model, str(model_path))
        
        assert model_path.exists()
        
        # Verify we can load it back
        loaded_model = joblib.load(model_path)
        assert loaded_model is not None
    
    @pytest.mark.unit
    def test_model_versioning(self, temp_dir, mock_model):
        """Test model version management"""
        manager = ModelManager(models_directory=str(temp_dir))
        
        # Save model with version
        model_info = manager.save_model_with_version(
            model=mock_model,
            model_name="fraud_detector",
            version="1.0.0",
            metadata={"accuracy": 0.95}
        )
        
        assert "model_path" in model_info
        assert "version" in model_info
        assert model_info["version"] == "1.0.0"
        
        # Load latest version
        loaded_model, metadata = manager.load_latest_model("fraud_detector")
        assert loaded_model is not None
        assert metadata["accuracy"] == 0.95
    
    @pytest.mark.unit
    def test_model_registry(self, temp_dir):
        """Test model registry functionality"""
        manager = ModelManager(models_directory=str(temp_dir))
        
        # Register multiple models
        manager.register_model("model_v1", "1.0.0", {"accuracy": 0.90})
        manager.register_model("model_v2", "2.0.0", {"accuracy": 0.95})
        
        registry = manager.get_model_registry()
        
        assert len(registry) == 2
        assert "model_v1" in registry
        assert "model_v2" in registry
        assert registry["model_v2"]["accuracy"] > registry["model_v1"]["accuracy"]


class TestModelTraining:
    """Test model training functionality"""
    
    @pytest.mark.unit
    def test_model_trainer_initialization(self):
        """Test model trainer initialization"""
        trainer = ModelTrainer(random_state=42)
        
        assert trainer.random_state == 42
        assert trainer.models is not None
        assert len(trainer.models) > 0
    
    @pytest.mark.unit
    def test_data_preparation(self, sample_transaction_data):
        """Test training data preparation"""
        trainer = ModelTrainer()
        
        X, y = trainer.prepare_training_data(sample_transaction_data, target_col='is_fraud')
        
        assert len(X) == len(y)
        assert len(X) > 0
        assert 'is_fraud' not in X.columns
        assert y.name == 'is_fraud' or y.dtype in [int, bool]
    
    @pytest.mark.unit
    def test_train_single_model(self, sample_transaction_data):
        """Test training a single model"""
        from sklearn.ensemble import RandomForestClassifier
        
        trainer = ModelTrainer()
        X, y = trainer.prepare_training_data(sample_transaction_data, target_col='is_fraud')
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        trained_model, metrics = trainer.train_model(model, X, y)
        
        assert trained_model is not None
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert all(0 <= score <= 1 for score in metrics.values())
    
    @pytest.mark.unit
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    def test_mlflow_integration(self, mock_log_metrics, mock_log_params, mock_start_run, sample_transaction_data):
        """Test MLflow integration in training"""
        trainer = ModelTrainer(use_mlflow=True)
        X, y = trainer.prepare_training_data(sample_transaction_data, target_col='is_fraud')
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        trainer.train_model(model, X, y)
        
        # Verify MLflow calls
        mock_start_run.assert_called()
        mock_log_params.assert_called()
        mock_log_metrics.assert_called()
    
    @pytest.mark.unit
    def test_hyperparameter_tuning(self, sample_transaction_data):
        """Test hyperparameter tuning"""
        from sklearn.ensemble import RandomForestClassifier
        
        tuner = HyperparameterTuner()
        X, y = tuner.prepare_data(sample_transaction_data, target_col='is_fraud')
        
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
        
        best_model, best_params = tuner.tune_model(
            RandomForestClassifier(random_state=42), 
            param_grid, X, y, cv=2  # Small CV for testing
        )
        
        assert best_model is not None
        assert best_params is not None
        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params
    
    @pytest.mark.unit
    def test_cross_validation(self, sample_transaction_data):
        """Test cross-validation functionality"""
        trainer = ModelTrainer()
        X, y = trainer.prepare_training_data(sample_transaction_data, target_col='is_fraud')
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        cv_scores = trainer.cross_validate(model, X, y, cv=3)
        
        assert 'mean_score' in cv_scores
        assert 'std_score' in cv_scores
        assert 'cv_scores' in cv_scores
        assert len(cv_scores['cv_scores']) == 3


class TestModelEvaluation:
    """Test model evaluation functionality"""
    
    @pytest.mark.unit
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation"""
        # Mock predictions
        y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8, 0.7, 0.6])
        
        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_prob)
        
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 
            'roc_auc', 'precision_recall_auc'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1
    
    @pytest.mark.unit
    def test_confusion_matrix_metrics(self):
        """Test confusion matrix derived metrics"""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        
        evaluator = ModelEvaluator()
        cm_metrics = evaluator.calculate_confusion_matrix_metrics(y_true, y_pred)
        
        assert 'true_positives' in cm_metrics
        assert 'false_positives' in cm_metrics
        assert 'true_negatives' in cm_metrics
        assert 'false_negatives' in cm_metrics
        assert 'sensitivity' in cm_metrics
        assert 'specificity' in cm_metrics
    
    @pytest.mark.unit
    def test_business_metrics(self):
        """Test business impact metrics"""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        
        evaluator = ModelEvaluator()
        business_metrics = evaluator.calculate_business_metrics(
            y_true, y_pred, 
            avg_fraud_amount=1000, 
            false_positive_cost=10
        )
        
        assert 'fraud_prevented' in business_metrics
        assert 'fraud_missed' in business_metrics
        assert 'false_positive_cost' in business_metrics
        assert 'net_benefit' in business_metrics
        assert 'roi' in business_metrics
    
    @pytest.mark.unit
    def test_threshold_optimization(self):
        """Test threshold optimization"""
        y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.6, 0.4, 0.25, 0.85])
        
        evaluator = ModelEvaluator()
        optimal_threshold, threshold_metrics = evaluator.optimize_threshold(
            y_true, y_prob, metric='f1_score'
        )
        
        assert 0 <= optimal_threshold <= 1
        assert len(threshold_metrics) > 0
        assert all('threshold' in tm for tm in threshold_metrics)
        assert all('f1_score' in tm for tm in threshold_metrics)
    
    @pytest.mark.unit
    def test_model_comparison(self, mock_model):
        """Test model comparison functionality"""
        evaluator = ModelEvaluator()
        
        # Mock results for multiple models
        model_results = {
            'model_a': {
                'accuracy': 0.90,
                'precision': 0.85,
                'recall': 0.80,
                'f1_score': 0.82
            },
            'model_b': {
                'accuracy': 0.88,
                'precision': 0.90,
                'recall': 0.75,
                'f1_score': 0.82
            }
        }
        
        comparison = evaluator.compare_models(model_results)
        
        assert 'best_model' in comparison
        assert 'comparison_table' in comparison
        assert 'ranking' in comparison
        
        # Best model should be determined by primary metric
        assert comparison['best_model'] in ['model_a', 'model_b']


class TestPredictionService:
    """Test prediction service functionality"""
    
    @pytest.mark.unit
    def test_prediction_service_initialization(self, mock_model):
        """Test prediction service initialization"""
        service = PredictionService(model=mock_model)
        
        assert service.model is mock_model
        assert service.is_ready()
    
    @pytest.mark.unit
    def test_real_time_prediction(self, mock_model):
        """Test real-time prediction"""
        service = PredictionService(model=mock_model)
        
        # Mock model prediction
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        transaction_data = {
            'amount': 500,
            'hour': 23,
            'payment_method_risk': 0.8
        }
        
        result = service.predict_single(transaction_data)
        
        assert 'fraud_probability' in result
        assert 'prediction' in result
        assert 'timestamp' in result
        assert result['fraud_probability'] == 0.7
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_batch_prediction(self, mock_model):
        """Test async batch prediction"""
        service = PredictionService(model=mock_model)
        
        # Mock batch predictions
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.9, 0.1]
        ])
        
        batch_data = [
            {'amount': 100, 'hour': 14},
            {'amount': 500, 'hour': 23},
            {'amount': 50, 'hour': 9}
        ]
        
        results = await service.predict_batch_async(batch_data)
        
        assert len(results) == 3
        assert all('fraud_probability' in result for result in results)
    
    @pytest.mark.unit
    def test_prediction_caching(self, mock_model, mock_redis):
        """Test prediction result caching"""
        service = PredictionService(model=mock_model, cache=mock_redis, cache_ttl=300)
        
        transaction_data = {'amount': 100, 'transaction_id': 'TXN_001'}
        
        # First prediction - should call model
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_redis.get.return_value = None  # Not in cache
        
        result1 = service.predict_single(transaction_data)
        
        # Should have cached the result
        mock_redis.set.assert_called()
        
        # Second prediction - should use cache
        mock_redis.get.return_value = '{"fraud_probability": 0.7, "prediction": 1}'
        
        result2 = service.predict_single(transaction_data)
        
        assert result1['fraud_probability'] == result2['fraud_probability']
    
    @pytest.mark.unit
    def test_prediction_monitoring(self, mock_model):
        """Test prediction monitoring and metrics"""
        service = PredictionService(model=mock_model, enable_monitoring=True)
        
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        # Make several predictions
        for i in range(10):
            service.predict_single({'amount': 100 * i, 'hour': 14})
        
        metrics = service.get_monitoring_metrics()
        
        assert 'prediction_count' in metrics
        assert 'average_response_time' in metrics
        assert 'fraud_rate' in metrics
        assert metrics['prediction_count'] == 10
