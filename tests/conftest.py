"""
Test Configuration and Fixtures
Shared test configuration, fixtures, and utilities
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import asyncio
from typing import Dict, List, Any, Optional
import os
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Test configuration
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config():
    """Test configuration dictionary"""
    return {
        'database': {
            'url': 'sqlite:///test_fraud_detection.db',
            'echo': False,
            'pool_size': 5
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 1  # Use different DB for testing
        },
        'kafka': {
            'bootstrap_servers': 'localhost:9092',
            'test_topic': 'test-transactions'
        },
        'api': {
            'host': '127.0.0.1',
            'port': 8001,  # Different port for testing
            'workers': 1
        },
        'model': {
            'threshold': 0.5,
            'model_path': 'tests/fixtures/test_model.pkl',
            'feature_columns': ['amount', 'hour', 'day_of_week']
        },
        'logging': {
            'level': 'DEBUG',
            'file': 'tests/reports/test.log'
        }
    }


@pytest.fixture(scope="function")
def temp_dir():
    """Create temporary directory for test files"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def sample_transaction_data():
    """Generate sample transaction data for testing"""
    np.random.seed(42)
    
    n_samples = 1000
    data = {
        'transaction_id': [f'TXN_{i:06d}' for i in range(n_samples)],
        'user_id': [f'USER_{np.random.randint(1, 101):03d}' for _ in range(n_samples)],
        'merchant_id': [f'MERCHANT_{np.random.randint(1, 51):02d}' for _ in range(n_samples)],
        'amount': np.random.lognormal(mean=3, sigma=1, size=n_samples).round(2),
        'currency': ['USD'] * n_samples,
        'payment_method': np.random.choice(
            ['credit_card', 'debit_card', 'paypal', 'apple_pay'], 
            size=n_samples, p=[0.5, 0.3, 0.15, 0.05]
        ),
        'timestamp': [
            datetime.now() - timedelta(days=np.random.randint(0, 30), 
                                     hours=np.random.randint(0, 24),
                                     minutes=np.random.randint(0, 60))
            for _ in range(n_samples)
        ],
        'country': np.random.choice(['US', 'UK', 'CA', 'DE'], size=n_samples, p=[0.6, 0.2, 0.1, 0.1]),
        'ip_address': [f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(n_samples)],
        'is_fraud': np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])
    }
    
    df = pd.DataFrame(data)
    
    # Add some realistic patterns
    # High amounts more likely to be fraud
    high_amount_mask = df['amount'] > df['amount'].quantile(0.95)
    df.loc[high_amount_mask, 'is_fraud'] = np.random.choice([0, 1], size=high_amount_mask.sum(), p=[0.7, 0.3])
    
    # Late night transactions more likely to be fraud  
    df['hour'] = df['timestamp'].dt.hour
    late_night_mask = (df['hour'] < 6) | (df['hour'] > 22)
    df.loc[late_night_mask, 'is_fraud'] = np.random.choice([0, 1], size=late_night_mask.sum(), p=[0.9, 0.1])
    
    return df


@pytest.fixture(scope="function") 
def sample_user_data():
    """Generate sample user data"""
    np.random.seed(42)
    
    n_users = 100
    data = {
        'user_id': [f'USER_{i:03d}' for i in range(1, n_users + 1)],
        'username': [f'user_{i}' for i in range(1, n_users + 1)],
        'email': [f'user{i}@example.com' for i in range(1, n_users + 1)],
        'first_name': [f'FirstName{i}' for i in range(1, n_users + 1)],
        'last_name': [f'LastName{i}' for i in range(1, n_users + 1)],
        'status': np.random.choice(['active', 'inactive'], size=n_users, p=[0.9, 0.1]),
        'role': np.random.choice(['viewer', 'analyst', 'admin'], size=n_users, p=[0.7, 0.25, 0.05]),
        'created_at': [datetime.now() - timedelta(days=np.random.randint(1, 365)) for _ in range(n_users)],
        'country': np.random.choice(['US', 'UK', 'CA', 'DE'], size=n_users, p=[0.5, 0.2, 0.2, 0.1])
    }
    
    return pd.DataFrame(data)


@pytest.fixture(scope="function")
def sample_merchant_data():
    """Generate sample merchant data"""
    np.random.seed(42)
    
    n_merchants = 50
    categories = ['electronics', 'clothing', 'food_beverage', 'travel', 'entertainment']
    
    data = {
        'merchant_id': [f'MERCHANT_{i:02d}' for i in range(1, n_merchants + 1)],
        'name': [f'Merchant {i}' for i in range(1, n_merchants + 1)],
        'category': np.random.choice(categories, size=n_merchants),
        'risk_level': np.random.choice(['low', 'medium', 'high'], size=n_merchants, p=[0.6, 0.3, 0.1]),
        'country': np.random.choice(['US', 'UK', 'CA'], size=n_merchants, p=[0.7, 0.2, 0.1]),
        'created_at': [datetime.now() - timedelta(days=np.random.randint(30, 1000)) for _ in range(n_merchants)]
    }
    
    return pd.DataFrame(data)


@pytest.fixture(scope="function")
def mock_model():
    """Create a mock ML model for testing"""
    model = Mock()
    model.predict.return_value = np.array([0, 1, 0, 1, 0])
    model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.85, 0.15]])
    model.feature_importances_ = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.03, 0.02])
    return model


@pytest.fixture(scope="function") 
def mock_database():
    """Create mock database connection"""
    db = Mock()
    db.execute.return_value = Mock()
    db.fetchall.return_value = []
    db.fetchone.return_value = None
    db.commit.return_value = None
    db.rollback.return_value = None
    return db


@pytest.fixture(scope="function")
def mock_redis():
    """Create mock Redis client"""
    redis_client = Mock()
    redis_client.get.return_value = None
    redis_client.set.return_value = True
    redis_client.delete.return_value = 1
    redis_client.ping.return_value = True
    redis_client.keys.return_value = []
    return redis_client


@pytest.fixture(scope="function")
def mock_kafka_producer():
    """Create mock Kafka producer"""
    producer = Mock()
    producer.send.return_value = Mock()
    producer.flush.return_value = None
    producer.close.return_value = None
    return producer


@pytest.fixture(scope="function")
def api_client():
    """Create test API client"""
    from fastapi.testclient import TestClient
    from src.main import app
    
    return TestClient(app)


@pytest.fixture(scope="session")
def test_database_url():
    """Test database URL"""
    return "sqlite:///./test_fraud_detection.db"


@pytest.fixture(scope="function", autouse=True)
def setup_test_environment():
    """Setup test environment variables"""
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ["ENVIRONMENT"] = "test"
    os.environ["DATABASE_URL"] = "sqlite:///./test_fraud_detection.db"
    os.environ["REDIS_URL"] = "redis://localhost:6379/1"
    os.environ["SECRET_KEY"] = "test-secret-key"
    os.environ["DEBUG"] = "true"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="function")
def fraud_features():
    """Standard fraud detection features for testing"""
    return [
        'amount', 'amount_log', 'hour', 'day_of_week', 'is_weekend',
        'is_business_hours', 'payment_method_risk', 'velocity_1h',
        'velocity_24h', 'user_transaction_count', 'merchant_risk_score'
    ]


@pytest.fixture(scope="function")
def sample_alert_data():
    """Generate sample alert data"""
    np.random.seed(42)
    
    n_alerts = 50
    data = {
        'alert_id': [f'ALERT_{i:04d}' for i in range(1, n_alerts + 1)],
        'alert_type': np.random.choice(['high_amount', 'velocity', 'anomaly'], size=n_alerts),
        'severity': np.random.choice(['low', 'medium', 'high', 'critical'], size=n_alerts, p=[0.3, 0.4, 0.2, 0.1]),
        'status': np.random.choice(['open', 'in_progress', 'resolved'], size=n_alerts, p=[0.4, 0.3, 0.3]),
        'transaction_id': [f'TXN_{np.random.randint(1, 1000):06d}' for _ in range(n_alerts)],
        'user_id': [f'USER_{np.random.randint(1, 101):03d}' for _ in range(n_alerts)],
        'risk_score': np.random.uniform(0.5, 1.0, size=n_alerts),
        'created_at': [datetime.now() - timedelta(hours=np.random.randint(1, 72)) for _ in range(n_alerts)]
    }
    
    return pd.DataFrame(data)


@pytest.fixture(scope="function")
def performance_test_data():
    """Large dataset for performance testing"""
    np.random.seed(42)
    
    n_samples = 10000  # Larger dataset for performance tests
    data = {
        'transaction_id': [f'TXN_{i:08d}' for i in range(n_samples)],
        'user_id': [f'USER_{np.random.randint(1, 1001):04d}' for _ in range(n_samples)],
        'merchant_id': [f'MERCHANT_{np.random.randint(1, 501):03d}' for _ in range(n_samples)],
        'amount': np.random.lognormal(mean=3, sigma=1.5, size=n_samples).round(2),
        'timestamp': [
            datetime.now() - timedelta(days=np.random.randint(0, 90),
                                     hours=np.random.randint(0, 24),
                                     minutes=np.random.randint(0, 60))
            for _ in range(n_samples)
        ],
        'is_fraud': np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])
    }
    
    return pd.DataFrame(data)


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_db: mark test as requiring database connection"
    )
    config.addinivalue_line(
        "markers", "requires_redis: mark test as requiring Redis connection"
    )
    config.addinivalue_line(
        "markers", "requires_kafka: mark test as requiring Kafka connection"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location"""
    for item in items:
        # Add markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session", autouse=True)
def setup_test_reports_dir():
    """Ensure test reports directory exists"""
    reports_dir = Path("tests/reports")
    reports_dir.mkdir(exist_ok=True)
    yield reports_dir


# Test utilities
class TestDataGenerator:
    """Utility class for generating test data"""
    
    @staticmethod
    def create_transaction_batch(n: int = 100, fraud_rate: float = 0.02) -> pd.DataFrame:
        """Create a batch of test transactions"""
        np.random.seed(42)
        
        data = {
            'transaction_id': [f'BATCH_TXN_{i:06d}' for i in range(n)],
            'user_id': [f'USER_{np.random.randint(1, 51):03d}' for _ in range(n)],
            'merchant_id': [f'MERCHANT_{np.random.randint(1, 21):02d}' for _ in range(n)],
            'amount': np.random.lognormal(mean=3, sigma=1, size=n).round(2),
            'timestamp': [datetime.now() - timedelta(minutes=np.random.randint(0, 1440)) for _ in range(n)],
            'is_fraud': np.random.choice([0, 1], size=n, p=[1-fraud_rate, fraud_rate])
        }
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_user_profile(user_id: str) -> Dict[str, Any]:
        """Create a test user profile"""
        return {
            'user_id': user_id,
            'username': f'testuser_{user_id}',
            'email': f'test_{user_id}@example.com',
            'status': 'active',
            'role': 'viewer',
            'created_at': datetime.now()
        }


# Helper functions for tests
def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype: bool = False):
    """Custom assertion for DataFrame equality"""
    pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)


def assert_model_predictions_valid(predictions: np.ndarray, probabilities: np.ndarray):
    """Assert model predictions are valid"""
    assert len(predictions) == len(probabilities)
    assert all(pred in [0, 1] for pred in predictions)
    assert all(0 <= prob <= 1 for prob_pair in probabilities for prob in prob_pair)
    assert all(abs(sum(prob_pair) - 1.0) < 1e-6 for prob_pair in probabilities)


def create_test_model_artifact(temp_dir: Path, model_name: str = "test_model"):
    """Create a test model artifact"""
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    # Create and train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X_dummy = np.random.rand(100, 5)
    y_dummy = np.random.randint(0, 2, 100)
    model.fit(X_dummy, y_dummy)
    
    # Save model
    model_path = temp_dir / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    
    return model_path
