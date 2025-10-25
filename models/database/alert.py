"""
Alert Database Model
Fraud alerts and notifications
"""

from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, Float, ForeignKey, Enum as SQLEnum, Index
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime

from ..enums.common import AlertStatus, AlertSeverity, RiskLevel
from ..mixins.base import BaseModelMixin

Base = declarative_base()


class Alert(Base, BaseModelMixin):
    """Alert model for fraud detection notifications"""
    
    __tablename__ = "alerts"
    __table_args__ = (
        Index('idx_alert_status_severity', 'status', 'severity'),
        Index('idx_alert_transaction', 'transaction_id'),
        Index('idx_alert_created_date', 'created_at'),
        Index('idx_alert_assigned_user', 'assigned_to'),
        {'comment': 'Fraud detection alerts and notifications'}
    )
    
    # Alert Identification
    alert_id = Column(String(100), unique=True, nullable=False, index=True)
    alert_type = Column(String(100), nullable=False)
    alert_subtype = Column(String(100), nullable=True)
    
    # Alert Content
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    message = Column(Text, nullable=False)
    
    # Severity and Status
    severity = Column(SQLEnum(AlertSeverity), nullable=False, index=True)
    status = Column(SQLEnum(AlertStatus), default=AlertStatus.OPEN, nullable=False, index=True)
    priority = Column(Integer, default=5, nullable=False)  # 1-10 scale
    
    # Related Entities
    transaction_id = Column(String(36), ForeignKey('transactions.id'), nullable=True, index=True)
    user_id = Column(String(100), nullable=True, index=True)
    merchant_id = Column(String(100), nullable=True, index=True)
    
    # Risk Information
    risk_score = Column(Float, nullable=True)
    risk_level = Column(SQLEnum(RiskLevel), nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Alert Triggers
    triggered_by_rule = Column(String(200), nullable=True)
    rule_threshold = Column(Float, nullable=True)
    actual_value = Column(Float, nullable=True)
    
    # Geographic Information
    country = Column(String(2), nullable=True)
    region = Column(String(100), nullable=True)
    ip_address = Column(String(45), nullable=True)
    
    # Timing Information
    event_timestamp = Column(DateTime(timezone=True), nullable=True)
    alert_generated_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    first_seen_at = Column(DateTime(timezone=True), nullable=True)
    last_seen_at = Column(DateTime(timezone=True), nullable=True)
    
    # Assignment and Review
    assigned_to = Column(String(36), ForeignKey('users.id'), nullable=True, index=True)
    assigned_at = Column(DateTime(timezone=True), nullable=True)
    
    # Resolution Information
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolved_by = Column(String(36), ForeignKey('users.id'), nullable=True)
    resolution_status = Column(String(100), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    resolution_action = Column(String(200), nullable=True)
    
    # Investigation
    investigation_started_at = Column(DateTime(timezone=True), nullable=True)
    investigation_completed_at = Column(DateTime(timezone=True), nullable=True)
    investigation_duration_seconds = Column(Integer, nullable=True)
    
    # False Positive Tracking
    is_false_positive = Column(Boolean, nullable=True)
    false_positive_reason = Column(String(500), nullable=True)
    false_positive_feedback = Column(Text, nullable=True)
    
    # Escalation
    escalation_level = Column(Integer, default=0, nullable=False)
    escalated_at = Column(DateTime(timezone=True), nullable=True)
    escalated_to = Column(String(36), ForeignKey('users.id'), nullable=True)
    escalation_reason = Column(String(500), nullable=True)
    
    # Notification Tracking
    notifications_sent = Column(Integer, default=0, nullable=False)
    last_notification_at = Column(DateTime(timezone=True), nullable=True)
    notification_channels = Column(String(500), nullable=True)  # email,sms,slack
    
    # Analytics and Metrics
    similar_alerts_count = Column(Integer, default=0, nullable=False)
    recurrence_count = Column(Integer, default=1, nullable=False)
    impact_score = Column(Float, nullable=True)
    
    # Additional Data
    context_data = Column(JSONB, nullable=True)
    risk_factors = Column(JSONB, nullable=True)
    related_entities = Column(JSONB, nullable=True)
    
    # External References
    external_alert_id = Column(String(200), nullable=True)
    external_system = Column(String(100), nullable=True)
    
    # Relationships
    transaction = relationship("Transaction", back_populates="alerts")
    assigned_to_user = relationship("User", back_populates="alerts", foreign_keys=[assigned_to])
    resolved_by_user = relationship("User", foreign_keys=[resolved_by])
    escalated_to_user = relationship("User", foreign_keys=[escalated_to])
    
    def __repr__(self):
        return f"<Alert(id={self.alert_id}, type={self.alert_type}, severity={self.severity})>"
    
    def is_open(self) -> bool:
        """Check if alert is open"""
        return self.status == AlertStatus.OPEN
    
    def is_resolved(self) -> bool:
        """Check if alert is resolved"""
        return self.status in [AlertStatus.RESOLVED, AlertStatus.CLOSED]
    
    def is_high_priority(self) -> bool:
        """Check if alert is high priority"""
        return self.priority >= 8 or self.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
    
    def is_overdue(self, sla_hours: int = 24) -> bool:
        """Check if alert is overdue based on SLA"""
        if self.is_resolved():
            return False
        
        time_diff = datetime.utcnow() - self.alert_generated_at
        return time_diff.total_seconds() > (sla_hours * 3600)
    
    def assign_to(self, user_id: str) -> None:
        """Assign alert to a user"""
        self.assigned_to = user_id
        self.assigned_at = datetime.utcnow()
        if self.status == AlertStatus.OPEN:
            self.status = AlertStatus.IN_PROGRESS
    
    def start_investigation(self) -> None:
        """Start investigation"""
        self.investigation_started_at = datetime.utcnow()
        if self.status == AlertStatus.OPEN:
            self.status = AlertStatus.IN_PROGRESS
    
    def resolve_alert(self, resolved_by: str, resolution_status: str, 
                     resolution_notes: str = None, resolution_action: str = None) -> None:
        """Resolve the alert"""
        self.resolved_at = datetime.utcnow()
        self.resolved_by = resolved_by
        self.resolution_status = resolution_status
        self.resolution_notes = resolution_notes
        self.resolution_action = resolution_action
        self.status = AlertStatus.RESOLVED
        
        # Calculate investigation duration
        if self.investigation_started_at:
            duration = self.resolved_at - self.investigation_started_at
            self.investigation_duration_seconds = int(duration.total_seconds())
        
        self.investigation_completed_at = datetime.utcnow()
    
    def mark_false_positive(self, reason: str, feedback: str = None) -> None:
        """Mark alert as false positive"""
        self.is_false_positive = True
        self.false_positive_reason = reason
        self.false_positive_feedback = feedback
        self.status = AlertStatus.FALSE_POSITIVE
    
    def escalate(self, escalated_to: str, reason: str = None) -> None:
        """Escalate alert"""
        self.escalation_level += 1
        self.escalated_at = datetime.utcnow()
        self.escalated_to = escalated_to
        self.escalation_reason = reason
    
    def record_notification(self, channels: list = None) -> None:
        """Record that notification was sent"""
        self.notifications_sent += 1
        self.last_notification_at = datetime.utcnow()
        if channels:
            self.notification_channels = ','.join(channels)
    
    def get_context_data(self) -> dict:
        """Get context data as dict"""
        if not self.context_data:
            return {}
        try:
            return self.context_data if isinstance(self.context_data, dict) else {}
        except (TypeError, AttributeError):
            return {}
    
    def set_context_data(self, data: dict) -> None:
        """Set context data from dict"""
        self.context_data = data if data else None
    
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
    
    def get_age_hours(self) -> float:
        """Get alert age in hours"""
        time_diff = datetime.utcnow() - self.alert_generated_at
        return time_diff.total_seconds() / 3600
    
    def get_resolution_time_hours(self) -> float:
        """Get resolution time in hours"""
        if not self.resolved_at:
            return 0.0
        
        time_diff = self.resolved_at - self.alert_generated_at
        return time_diff.total_seconds() / 3600
    
    def to_dict(self, include_sensitive: bool = False) -> dict:
        """Convert alert to dictionary"""
        data = {
            'id': self.id,
            'alert_id': self.alert_id,
            'alert_type': self.alert_type,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value if self.severity else None,
            'status': self.status.value if self.status else None,
            'priority': self.priority,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level.value if self.risk_level else None,
            'transaction_id': self.transaction_id,
            'user_id': self.user_id,
            'merchant_id': self.merchant_id,
            'assigned_to': self.assigned_to,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'age_hours': self.get_age_hours(),
            'is_overdue': self.is_overdue(),
        }
        
        if include_sensitive:
            data.update({
                'ip_address': self.ip_address,
                'context_data': self.get_context_data(),
                'risk_factors': self.get_risk_factors(),
                'resolution_notes': self.resolution_notes,
            })
        
        return data
