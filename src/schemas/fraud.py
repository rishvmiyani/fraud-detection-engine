"""
Fraud Detection Pydantic Schemas
Data models for API request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Decision(str, Enum):
    """Decision enumeration"""
    APPROVE = "approve"
    REVIEW = "review"
    BLOCK = "block"
    CHALLENGE = "challenge"


class PaymentMethod(str, Enum):
    """Payment method enumeration"""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    PAYPAL = "paypal"
    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"
    BANK_TRANSFER = "bank_transfer"
    WIRE_TRANSFER = "wire_transfer"
    CRYPTOCURRENCY = "cryptocurrency"


class FraudDetectionRequest(BaseModel):
    """Request schema for fraud detection"""
    
    # Required fields
    transaction_id: str = Field(..., min_length=1, max_length=100)
    user_id: str = Field(..., min_length=1, max_length=100)
    merchant_id: str = Field(..., min_length=1, max_length=100)
    amount: float = Field(..., gt=0, le=1000000)
    currency: str = Field(..., min_length=3, max_length=3)
    timestamp: datetime
    
    # Optional transaction details
    payment_method: Optional[PaymentMethod] = None
    country: Optional[str] = Field(None, min_length=2, max_length=2)
    ip_address: Optional[str] = None
    device_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Additional context
    merchant_category: Optional[str] = None
    billing_address: Optional[Dict[str, Any]] = None
    shipping_address: Optional[Dict[str, Any]] = None
    card_info: Optional[Dict[str, Any]] = None
    
    # Custom fields
    custom_fields: Optional[Dict[str, Any]] = None
    
    @validator('currency')
    def validate_currency(cls, v):
        if v and len(v) != 3:
            raise ValueError('Currency code must be 3 characters')
        return v.upper() if v else v
    
    @validator('country')
    def validate_country(cls, v):
        if v and len(v) != 2:
            raise ValueError('Country code must be 2 characters')
        return v.upper() if v else v
    
    @validator('ip_address')
    def validate_ip_address(cls, v):
        if v:
            import re
            ip_pattern = re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')
            if not ip_pattern.match(v):
                raise ValueError('Invalid IP address format')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "TXN123456789",
                "user_id": "USR789456123",
                "merchant_id": "MRC654321987",
                "amount": 1500.00,
                "currency": "USD",
                "timestamp": "2025-10-17T10:30:00Z",
                "payment_method": "credit_card",
                "country": "US",
                "ip_address": "192.168.1.100",
                "device_id": "DEV123456",
                "merchant_category": "electronics"
            }
        }


class FraudExplanation(BaseModel):
    """Fraud detection explanation"""
    top_risk_factors: List[str]
    feature_importance: Dict[str, float]
    model_version: str
    confidence_intervals: Optional[Dict[str, Any]] = None


class FraudDetectionData(BaseModel):
    """Fraud detection response data"""
    fraud_score: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    decision: Decision
    confidence: float = Field(..., ge=0, le=1)
    explanation: FraudExplanation
    recommended_actions: List[str]
    
    # Additional insights
    velocity_check: Optional[Dict[str, Any]] = None
    behavioral_analysis: Optional[Dict[str, Any]] = None
    external_checks: Optional[Dict[str, Any]] = None


class ResponseMetadata(BaseModel):
    """Response metadata"""
    request_id: str
    processing_time_ms: int
    timestamp: int
    model_version: str
    from_cache: Optional[bool] = False
    
    # Performance metrics
    latency_breakdown: Optional[Dict[str, int]] = None


class FraudDetectionResponse(BaseModel):
    """Response schema for fraud detection"""
    status: str = "success"
    data: FraudDetectionData
    metadata: ResponseMetadata


class BatchTransactionRequest(BaseModel):
    """Single transaction in batch request"""
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float = Field(..., gt=0, le=1000000)
    currency: str = Field(..., min_length=3, max_length=3)
    timestamp: datetime
    payment_method: Optional[PaymentMethod] = None
    country: Optional[str] = None
    ip_address: Optional[str] = None
    device_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "TXN123",
                "user_id": "USR456",
                "merchant_id": "MRC789",
                "amount": 150.00,
                "currency": "USD",
                "timestamp": "2025-10-17T10:30:00Z",
                "payment_method": "credit_card",
                "country": "US"
            }
        }


class BatchOptions(BaseModel):
    """Options for batch processing"""
    return_explanations: bool = True
    async_processing: bool = False
    priority: str = "normal"  # low, normal, high
    callback_url: Optional[str] = None
    
    # Processing options
    parallel_processing: bool = True
    batch_size: int = Field(default=100, ge=1, le=1000)


class BatchFraudRequest(BaseModel):
    """Request schema for batch fraud detection"""
    transactions: List[BatchTransactionRequest] = Field(..., min_items=1, max_items=1000)
    options: Optional[BatchOptions] = None
    
    @validator('transactions')
    def validate_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError('Batch size cannot exceed 1000 transactions')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "transactions": [
                    {
                        "transaction_id": "TXN001",
                        "user_id": "USR123",
                        "merchant_id": "MRC456",
                        "amount": 150.00,
                        "currency": "USD",
                        "timestamp": "2025-10-17T10:30:00Z"
                    }
                ],
                "options": {
                    "return_explanations": True,
                    "async_processing": False
                }
            }
        }


class BatchFraudResult(BaseModel):
    """Single result in batch fraud detection"""
    transaction_id: str
    fraud_score: float
    risk_level: RiskLevel
    decision: Decision
    confidence: float
    processing_time_ms: int
    explanation: Optional[FraudExplanation] = None


class BatchSummary(BaseModel):
    """Summary of batch processing"""
    total_processed: int
    fraud_detected: int
    average_score: float
    processing_time_ms: int
    
    # Distribution
    risk_distribution: Optional[Dict[str, int]] = None
    decision_distribution: Optional[Dict[str, int]] = None


class BatchFraudData(BaseModel):
    """Batch fraud detection response data"""
    results: List[BatchFraudResult]
    summary: BatchSummary
    
    # Processing info
    batch_id: Optional[str] = None
    status: str = "completed"  # completed, processing, failed


class BatchFraudResponse(BaseModel):
    """Response schema for batch fraud detection"""
    status: str = "success"
    data: BatchFraudData
    metadata: ResponseMetadata


class EntityType(str, Enum):
    """Entity type for risk scoring"""
    USER = "user"
    MERCHANT = "merchant"
    IP_ADDRESS = "ip_address"
    DEVICE = "device"


class RiskScoreRequest(BaseModel):
    """Request schema for risk score calculation"""
    entity_type: EntityType
    entity_id: str = Field(..., min_length=1, max_length=100)
    lookback_days: int = Field(default=30, ge=1, le=365)
    include_features: bool = True
    
    # Optional filters
    transaction_types: Optional[List[str]] = None
    amount_range: Optional[Dict[str, float]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "entity_type": "user",
                "entity_id": "USR123456",
                "lookback_days": 30,
                "include_features": True
            }
        }


class RiskFactors(BaseModel):
    """Risk factors breakdown"""
    velocity_risk: float
    behavioral_risk: float
    geographic_risk: float
    amount_risk: float
    payment_method_risk: float
    
    # Detailed factors
    specific_factors: List[Dict[str, Any]]


class HistoricalSummary(BaseModel):
    """Historical data summary"""
    total_transactions: int
    fraud_transactions: int
    fraud_rate: float
    average_amount: float
    unique_merchants: int
    unique_countries: int
    
    # Trends
    trend_analysis: Optional[Dict[str, Any]] = None


class RiskScoreData(BaseModel):
    """Risk score response data"""
    entity_type: EntityType
    entity_id: str
    risk_score: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    confidence: float = Field(..., ge=0, le=1)
    factors: RiskFactors
    historical_data: HistoricalSummary
    recommendations: List[str]
    
    # Additional insights
    peer_comparison: Optional[Dict[str, Any]] = None
    time_series_data: Optional[List[Dict[str, Any]]] = None


class RiskScoreResponse(BaseModel):
    """Response schema for risk score"""
    status: str = "success"
    data: RiskScoreData
    metadata: ResponseMetadata


# Error response schemas
class ErrorDetail(BaseModel):
    """Error detail schema"""
    code: str
    message: str
    details: Dict[str, Any] = {}


class ErrorResponse(BaseModel):
    """Error response schema"""
    status: str = "error"
    error: ErrorDetail
    metadata: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "status": "error",
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid transaction data",
                    "details": {
                        "field": "amount",
                        "issue": "Amount must be greater than 0"
                    }
                },
                "metadata": {
                    "request_id": "req_123456",
                    "timestamp": 1697543400
                }
            }
        }
