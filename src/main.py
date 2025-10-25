"""
Enhanced Fraud Detection API with ML Models
Clean version without syntax errors
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import joblib
import json
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Enhanced Fraud Detection Engine",
    description="Real-time Fraud Detection with Advanced ML Models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TransactionRequest(BaseModel):
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float
    payment_method: str
    merchant_category: Optional[str] = "other"
    country: Optional[str] = "US"
    device_type: Optional[str] = "mobile"
    timestamp: str

class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    fraud_prediction: int
    risk_level: str
    status: str
    model_used: str
    confidence: float
    timestamp: str

class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionRequest]

# Enhanced ML-based Fraud Detector
class EnhancedMLFraudDetector:
    def __init__(self):
        self.models = {}
        self.encoders = None
        self.scaler = None
        self.feature_columns = []
        self.model_info = {}
        self.prediction_count = 0
        self.fraud_count = 0
        
        self.load_models()
    
    def load_models(self):
        """Load trained ML models"""
        try:
            models_path = '../models/production/'
            
            # Load model comparison info if exists
            comparison_path = os.path.join(models_path, 'model_comparison.json')
            if os.path.exists(comparison_path):
                with open(comparison_path, 'r') as f:
                    self.model_info = json.load(f)
                
                # Load best model
                if self.model_info:
                    best_model_name = max(self.model_info.items(), key=lambda x: x[1].get('roc_auc', 0))[0]
                    model_path = os.path.join(models_path, f'{best_model_name}_model.pkl')
                    
                    if os.path.exists(model_path):
                        self.models['primary'] = joblib.load(model_path)
                        logger.info(f"Loaded primary model: {best_model_name}")
            
            # Load encoders if exist
            encoders_path = os.path.join(models_path, 'encoders.pkl')
            if os.path.exists(encoders_path):
                self.encoders = joblib.load(encoders_path)
                logger.info("Loaded feature encoders")
            
            # Load feature info if exists
            feature_info_path = os.path.join(models_path, 'feature_info.json')
            if os.path.exists(feature_info_path):
                with open(feature_info_path, 'r') as f:
                    feature_info = json.load(f)
                    self.feature_columns = feature_info.get('feature_columns', [])
                    logger.info(f"Loaded {len(self.feature_columns)} features")
                    
        except Exception as e:
            logger.warning(f"Could not load ML models: {e}")
            logger.info("Falling back to rule-based detection")
    
    def prepare_features(self, transaction: TransactionRequest) -> pd.DataFrame:
        """Prepare features for ML model"""
        try:
            # Parse timestamp
            ts = pd.to_datetime(transaction.timestamp)
            
            # Create base features
            data = {
                'amount': transaction.amount,
                'hour': ts.hour,
                'day_of_week': ts.dayofweek,
                'month': ts.month,
                'is_weekend': 1 if ts.dayofweek >= 5 else 0,
                'is_night': 1 if (ts.hour >= 22 or ts.hour <= 6) else 0,
                'is_business_hours': 1 if (9 <= ts.hour <= 17) else 0,
                'is_round_amount': 1 if (transaction.amount % 100 == 0) else 0,
            }
            
            # Add log amount
            data['amount_log'] = np.log1p(transaction.amount)
            data['amount_vs_user_avg'] = 1.0  # Placeholder
            
            # Encode categorical features if encoders available
            if self.encoders:
                categorical_mapping = {
                    'payment_method': transaction.payment_method,
                    'merchant_category': transaction.merchant_category,
                    'country': transaction.country,
                    'device_type': transaction.device_type
                }
                
                for col, value in categorical_mapping.items():
                    encoder = self.encoders.get(col)
                    if encoder:
                        try:
                            encoded = encoder.transform([str(value)])[0]
                            data[f'{col}_encoded'] = encoded
                        except:
                            # Handle unknown categories
                            data[f'{col}_encoded'] = 0
            
            # Create DataFrame
            df = pd.DataFrame([data])
            
            # Select only available features
            if self.feature_columns:
                available_features = [col for col in self.feature_columns if col in df.columns]
                if available_features:
                    df = df[available_features]
            
            return df
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            # Return simple fallback features
            return pd.DataFrame([{
                'amount': transaction.amount,
                'hour': 12,
                'is_weekend': 0
            }])
    
    def predict(self, transaction: TransactionRequest) -> Dict[str, Any]:
        """Make fraud prediction"""
        try:
            self.prediction_count += 1
            
            # Use ML model if available
            if 'primary' in self.models and self.feature_columns:
                return self.ml_predict(transaction)
            else:
                return self.rule_based_predict(transaction)
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self.rule_based_predict(transaction)
    
    def ml_predict(self, transaction: TransactionRequest) -> Dict[str, Any]:
        """ML-based prediction"""
        try:
            # Prepare features
            features_df = self.prepare_features(transaction)
            
            # Make prediction
            model = self.models['primary']
            fraud_prob = model.predict_proba(features_df)[0][1]
            fraud_pred = 1 if fraud_prob > 0.5 else 0
            
            # Calculate confidence
            confidence = max(fraud_prob, 1 - fraud_prob)
            
            # Determine risk level and status
            if fraud_prob < 0.2:
                risk_level = "low"
                status = "approved"
            elif fraud_prob < 0.5:
                risk_level = "medium"
                status = "review"
            elif fraud_prob < 0.8:
                risk_level = "high" 
                status = "blocked"
            else:
                risk_level = "critical"
                status = "blocked"
            
            if fraud_pred == 1:
                self.fraud_count += 1
            
            return {
                'transaction_id': transaction.transaction_id,
                'fraud_probability': round(fraud_prob, 4),
                'fraud_prediction': fraud_pred,
                'risk_level': risk_level,
                'status': status,
                'model_used': 'ml_model',
                'confidence': round(confidence, 4),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self.rule_based_predict(transaction)
    
    def rule_based_predict(self, transaction: TransactionRequest) -> Dict[str, Any]:
        """Fallback rule-based prediction"""
        fraud_score = 0.0
        
        # Rule-based scoring
        if transaction.amount > 1000:
            fraud_score += 0.3
        elif transaction.amount > 500:
            fraud_score += 0.1
            
        if transaction.payment_method in ['cryptocurrency', 'wire_transfer']:
            fraud_score += 0.4
        elif transaction.payment_method == 'credit_card':
            fraud_score += 0.05
            
        if transaction.amount % 100 == 0 and transaction.amount >= 1000:
            fraud_score += 0.2
            
        # Add timestamp-based risk
        try:
            ts = pd.to_datetime(transaction.timestamp)
            hour = ts.hour
            if hour >= 22 or hour <= 6:  # Night time
                fraud_score += 0.15
        except:
            pass
            
        # Add some randomness
        fraud_score += np.random.uniform(-0.05, 0.05)
        fraud_score = max(0.0, min(1.0, fraud_score))
        
        fraud_pred = 1 if fraud_score > 0.5 else 0
        
        if fraud_score < 0.2:
            risk_level = 'low'
            status = 'approved'
        elif fraud_score < 0.5:
            risk_level = 'medium'
            status = 'review'
        else:
            risk_level = 'high'
            status = 'blocked'
        
        if fraud_pred == 1:
            self.fraud_count += 1
        
        return {
            'transaction_id': transaction.transaction_id,
            'fraud_probability': round(fraud_score, 4),
            'fraud_prediction': fraud_pred,
            'risk_level': risk_level,
            'status': status,
            'model_used': 'rule_based',
            'confidence': round(abs(fraud_score - 0.5) * 2, 4),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_stats(self):
        """Get prediction statistics"""
        fraud_rate = (self.fraud_count / self.prediction_count * 100) if self.prediction_count > 0 else 0
        return {
            'total_predictions': self.prediction_count,
            'fraud_detected': self.fraud_count,
            'fraud_rate': f'{fraud_rate:.1f}%',
            'models_loaded': len(self.models),
            'features_available': len(self.feature_columns)
        }

# Initialize enhanced detector
fraud_detector = EnhancedMLFraudDetector()

# API Routes
@app.get("/")
async def root():
    return {
        "message": "Enhanced Fraud Detection Engine with ML",
        "version": "2.0.0",
        "models_loaded": len(fraud_detector.models),
        "features": len(fraud_detector.feature_columns),
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "ml_models": len(fraud_detector.models)
    }

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionRequest):
    """Enhanced fraud prediction with ML models"""
    try:
        result = fraud_detector.predict(transaction)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/v1/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Batch fraud prediction"""
    try:
        results = []
        for transaction in request.transactions:
            result = fraud_detector.predict(transaction)
            results.append(result)
        
        return {"predictions": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/api/v1/stats")
async def get_enhanced_stats():
    """Enhanced API statistics"""
    stats = fraud_detector.get_stats()
    stats.update({
        "model_info": fraud_detector.model_info if fraud_detector.model_info else {},
        "uptime": "running",
        "version": "2.0.0"
    })
    return stats

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics"""
    stats = fraud_detector.get_stats()
    return {
        "fraud_detection_predictions_total": stats["total_predictions"],
        "fraud_detection_fraud_total": stats["fraud_detected"],
        "fraud_detection_models_loaded": stats["models_loaded"],
        "fraud_detection_features_count": stats["features_available"]
    }

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Enhanced Fraud Detection Engine v2.0 started!")
    logger.info(f"📊 Models loaded: {len(fraud_detector.models)}")
    logger.info(f"🔧 Features available: {len(fraud_detector.feature_columns)}")
    logger.info("📖 Enhanced API Documentation: http://localhost:8000/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
