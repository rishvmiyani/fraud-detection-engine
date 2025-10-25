from sqlalchemy import Column, String, DateTime, Text, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from ..enums.common import AuditAction
from ..mixins.base import TimestampMixin, UUIDMixin

Base = declarative_base()

class AuditLog(Base, TimestampMixin, UUIDMixin):
    __tablename__ = "audit_logs"
    
    user_id = Column(String(36), ForeignKey('users.id'), nullable=True)
    action = Column(SQLEnum(AuditAction), nullable=False)
    entity_type = Column(String(100), nullable=False)
    entity_id = Column(String(100), nullable=False)
    details = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    
    user = relationship("User", back_populates="audit_logs")
    
    def __repr__(self):
        return f"<AuditLog(action={self.action}, entity={self.entity_type})>"
