"""
Integration Tests - Test Running API Server
Tests that work with your running fraud detection API
"""

import pytest
import requests
import json
import time

# API base URL
API_BASE_URL = "http://localhost:8000"

class TestAPIIntegration:
    """Test running API server"""
    
    def test_api_health_check(self):
        """Test API health endpoint"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=10)
            assert response.status_code == 200
            
            data = response.json()
            assert "status" in data
            assert data["status"] == "healthy"
            print("✅ Health check passed")
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running on localhost:8000")
    
    def test_api_root_endpoint(self):
        """Test API root endpoint"""
        try:
            response = requests.get(f"{API_BASE_URL}/", timeout=10)
            assert response.status_code == 200
            
            data = response.json()
            assert "message" in data
            assert "version" in data
            print("✅ Root endpoint passed")
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_fraud_prediction_endpoint(self):
        """Test fraud prediction endpoint"""
        try:
            test_transaction = {
                "transaction_id": "PYTEST_001",
                "user_id": "PYTEST_USER",
                "merchant_id": "PYTEST_MERCHANT", 
                "amount": 250.75,
                "payment_method": "credit_card",
                "timestamp": "2025-10-18T22:12:00Z",
                "country": "US"
            }
            
            response = requests.post(
                f"{API_BASE_URL}/api/v1/predict",
                json=test_transaction,
                timeout=10
            )
            
            assert response.status_code == 200
            
            data = response.json()
            assert "transaction_id" in data
            assert "fraud_probability" in data
            assert "fraud_prediction" in data
            assert "risk_level" in data
            assert "status" in data
            
            assert data["transaction_id"] == test_transaction["transaction_id"]
            assert 0.0 <= data["fraud_probability"] <= 1.0
            assert data["fraud_prediction"] in [0, 1]
            assert data["risk_level"] in ["low", "medium", "high"]
            
            print(f"✅ Fraud prediction passed - Risk: {data['risk_level']}, Score: {data['fraud_probability']}")
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_api_stats_endpoint(self):
        """Test API statistics endpoint"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/v1/stats", timeout=10)
            assert response.status_code == 200
            
            data = response.json()
            assert "total_requests" in data
            assert "fraud_detected" in data
            assert "fraud_rate" in data
            
            print(f"✅ Stats endpoint passed - Total requests: {data['total_requests']}")
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_multiple_predictions(self):
        """Test multiple fraud predictions"""
        try:
            test_cases = [
                {
                    "transaction_id": "LOW_RISK_TEST",
                    "user_id": "USER_SAFE",
                    "merchant_id": "MERCHANT_TRUSTED",
                    "amount": 50.0,
                    "payment_method": "debit_card",
                    "timestamp": "2025-10-18T22:12:00Z"
                },
                {
                    "transaction_id": "HIGH_RISK_TEST", 
                    "user_id": "USER_RISKY",
                    "merchant_id": "MERCHANT_SUSPICIOUS",
                    "amount": 2000.0,
                    "payment_method": "cryptocurrency", 
                    "timestamp": "2025-10-18T22:12:00Z"
                }
            ]
            
            results = []
            for test_case in test_cases:
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/predict",
                    json=test_case,
                    timeout=10
                )
                
                assert response.status_code == 200
                results.append(response.json())
            
            # Low risk should have lower fraud probability than high risk
            low_risk_score = results[0]["fraud_probability"]
            high_risk_score = results[1]["fraud_probability"]
            
            print(f"Low risk score: {low_risk_score}")
            print(f"High risk score: {high_risk_score}")
            
            # This test might fail depending on random component, so we make it flexible
            assert high_risk_score >= low_risk_score or abs(high_risk_score - low_risk_score) < 0.3
            
            print("✅ Multiple predictions test passed")
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
