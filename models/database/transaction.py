"""
Transaction Database Model
Core transaction data with fraud detection results
"""

from sqlalchemy import Column, String, Numeric, DateTime, Boolean, Text, Integer, Float, ForeignKey, Enum as SQLEnum, Index
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
import json
from decimal import Decimal

from ..enums.common import TransactionStatus, PaymentMethod, RiskLevel, Decision
from ..mixins.base import BaseModelMixin

Base = declarative_base()


class Transaction(Base, BaseModelMixin):
    """Transaction model with fraud detection data"""
    
    __tablename__ = "transactions"
    __table_args__ = (
        Index('idx_transaction_user_date', 'user_id', 'transaction_date'),
        Index('idx_transaction_merchant_date', 'merchant_id', 'transaction_date'),
        Index('idx_transaction_fraud_score', 'fraud_score'),
        Index('idx_transaction_amount', 'amount'),
        Index('idx_transaction_status', 'status'),
        Index('idx_transaction_risk_level', 'risk_level'),
        {'comment': 'Transaction records with fraud detection results'}
    )
    
    # Core Transaction Data
    transaction_id = Column(String(100), unique=True, nullable=False, index=True)
    external_transaction_id = Column(String(100), nullable=True, index=True)
    
    # Parties
    user_id = Column(String(100), nullable=False, index=True)
    merchant_id = Column(String(100), nullable=False, index=True)
    
    # Financial Information
    amount = Column(Numeric(15, 2), nullable=False)
    currency = Column(String(3), nullable=False, default='USD')
    original_amount = Column(Numeric(15, 2), nullable=True)
    original_currency = Column(String(3), nullable=True)
    exchange_rate = Column(Float, nullable=True)
    
    # Payment Details
    payment_method = Column(SQLEnum(PaymentMethod), nullable=False)
    payment_processor = Column(String(100), nullable=True)
    payment_reference = Column(String(255), nullable=True)
    
    # Card Information (if applicable)
    card_last_four = Column(String(4), nullable=True)
    card_brand = Column(String(50), nullable=True)
    card_type = Column(String(50), nullable=True)  # credit, debit, prepaid
    card_country = Column(String(2), nullable=True)
    card_bin = Column(String(8), nullable=True)
    
    # Transaction Context
    transaction_date = Column(DateTime(timezone=True), nullable=False, index=True)
    processing_date = Column(DateTime(timezone=True), nullable=True)
    settlement_date = Column(DateTime(timezone=True), nullable=True)
    
    # Status
    status = Column(SQLEnum(TransactionStatus), default=TransactionStatus.PENDING, nullable=False)
    status_reason = Column(String(255), nullable=True)
    
    # Location Information
    country = Column(String(2), nullable=True)
    region = Column(String(100), nullable=True)
    city = Column(String(100), nullable=True)
    timezone = Column(String(50), nullable=True)
    
    # Technical Context
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(Text, nullable=True)
    device_id = Column(String(255), nullable=True)
    session_id = Column(String(255), nullable=True)
    
    # Fraud Detection Results
    fraud_score = Column(Float, nullable=True, index=True)
    risk_level = Column(SQLEnum(RiskLevel), nullable=True, index=True)
    decision = Column(SQLEnum(Decision), nullable=True)
    model_version = Column(String(50), nullable=True)
    model_confidence = Column(Float, nullable=True)
    
    # Fraud Analysis Details
    risk_factors = Column(JSONB, nullable=True)
    feature_values = Column(JSONB, nullable=True)
    rule_results = Column(JSONB, nullable=True)
    
    # Velocity Metrics
    velocity_1h = Column(Integer, default=0, nullable=False)
    velocity_24h = Column(Integer, default=0, nullable=False)
    velocity_7d = Column(Integer, default=0, nullable=False)
    
    # Behavioral Metrics
    user_transaction_count = Column(Integer, default=0, nullable=False)
    merchant_transaction_count = Column(Integer, default=0, nullable=False)
    amount_percentile = Column(Float, nullable=True)
    
    # External Checks
    blacklist_check = Column(Boolean, default=False, nullable=False)
    whitelist_check = Column(Boolean, default=False, nullable=False)
    watchlist_match = Column(Boolean, default=False, nullable=False)
    
    # Verification Status
    is_verified = Column(Boolean, default=False, nullable=False)
    verification_method = Column(String(100), nullable=True)
    verification_date = Column(DateTime(timezone=True), nullable=True)
    
    # Business Information
    merchant_category = Column(String(100), nullable=True)
    product_category = Column(String(100), nullable=True)
    order_id = Column(String(100), nullable=True)
    
    # Fees and Costs
    processing_fee = Column(Numeric(10, 4), nullable=True)
    fraud_cost = Column(Numeric(10, 2), nullable=True)
    chargeback_probability = Column(Float, nullable=True)
    
    # Review Information
    requires_review = Column(Boolean, default=False, nullable=False)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    reviewed_by = Column(String(36), ForeignKey('users.id'), nullable=True)
    review_notes = Column(Text, nullable=True)
    
    # False Positive/Negative Tracking
    is_false_positive = Column(Boolean, nullable=True)
    is_false_negative = Column(Boolean, nullable=True)
    actual_fraud_status = Column(Boolean, nullable=True)
    ground_truth_source = Column(String(100), nullable=True)
    
    # Additional Data
    custom_fields = Column(JSONB, nullable=True)
    external_scores = Column(JSONB, nullable=True)
    
    # Relationships
    created_by_user = relationship("User", back_populates="transactions", foreign_keys=[reviewed_by])
    alerts = relationship("Alert", back_populates="transaction", lazy="dynamic")
    
    def __repr__(self):
        return f"<Transaction(id={self.transaction_id}, amount={self.amount}, fraud_score={self.fraud_score})>"
    
    def get_risk_factors(self) -> list:
        """Get risk factors as list"""
        if not self.risk_factors:
            return []
        try:
            return self.risk_factors if isinstance(self.risk_factors, list) else []
        except (TypeError, AttributeError):
            return []
    
    def set_risk_factors(self, factors: list) -> None:
        """Set risk factors from list"""
        self.risk_factors = factors if factors else None
    
    def get_feature_values(self) -> dict:
        """Get feature values as dict"""
        if not self.feature_values:
            return {}
        try:
            return self.feature_values if isinstance(self.feature_values, dict) else {}
        except (TypeError, AttributeError):
            return {}
    
    def set_feature_values(self, features: dict) -> None:
        """Set feature values from dict"""
        self.feature_values = features if features else None
    
    def get_custom_fields(self) -> dict:
        """Get custom fields as dict"""
        if not self.custom_fields:
            return {}
        try:
            return self.custom_fields if isinstance(self.custom_fields, dict) else {}
        except (TypeError, AttributeError):
            return {}
    
    def set_custom_fields(self, fields: dict) -> None:
        """Set custom fields from dict"""
        self.custom_fields = fields if fields else None
    
    def is_high_risk(self) -> bool:
        """Check if transaction is high risk"""
        return self.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    
    def is_fraud_suspected(self) -> bool:
        """Check if fraud is suspected"""
        return self.fraud_score and self.fraud_score > 0.5
    
    def should_block(self) -> bool:
        """Check if transaction should be blocked"""
        return self.decision == Decision.BLOCK
    
    def needs_manual_review(self) -> bool:
        """Check if transaction needs manual review"""
        return self.requires_review or self.decision == Decision.REVIEW
    
    def calculate_amount_usd(self) -> Decimal:
        """Calculate amount in USD"""
        if self.currency == 'USD':
            return self.amount
        elif self.exchange_rate:
            return self.amount * Decimal(str(self.exchange_rate))
        else:
            return self.amount  # Fallback to original amount
    
    def get_processing_time_seconds(self) -> float:
        """Get processing time in seconds"""
        if self.processing_date and self.transaction_date:
            delta = self.processing_date - self.transaction_date
            return delta.total_seconds()
        return 0.0
    
    def mark_as_fraud(self, source: str = "manual") -> None:
        """Mark transaction as confirmed fraud"""
        self.actual_fraud_status = True
        self.ground_truth_source = source
        if self.fraud_score and self.fraud_score < 0.5:
            self.is_false_negative = True
    
    def mark_as_legitimate(self, source: str = "manual") -> None:
        """Mark transaction as confirmed legitimate"""
        self.actual_fraud_status = False
        self.ground_truth_source = source
        if self.fraud_score and self.fraud_score > 0.5:
            self.is_false_positive = True
    
    def to_dict(self, include_sensitive: bool = False) -> dict:
        """Convert transaction to dictionary"""
        data = {
            'id': self.id,
            'transaction_id': self.transaction_id,
            'user_id': self.user_id,
            'merchant_id': self.merchant_id,
            'amount': float(self.amount) if self.amount else None,
            'currency': self.currency,
            'payment_method': self.payment_method.value if self.payment_method else None,
            'transaction_date': self.transaction_date.isoformat() if self.transaction_date else None,
            'status': self.status.value if self.status else None,
            'fraud_score': self.fraud_score,
            'risk_level': self.risk_level.value if self.risk_level else None,
            'decision': self.decision.value if self.decision else None,
            'country': self.country,
            'merchant_category': self.merchant_category,
            'requires_review': self.requires_review,
        }
        
        if include_sensitive:
            data.update({
                'ip_address': self.ip_address,
                'device_id': self.device_id,
                'card_last_four': self.card_last_four,
                'risk_factors': self.get_risk_factors(),
                'feature_values': self.get_feature_values(),
                'custom_fields': self.get_custom_fields(),
            })
        
        return data
