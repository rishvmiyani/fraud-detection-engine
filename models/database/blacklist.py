from sqlalchemy import Column, String, DateTime, Boolean, Text, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from ..enums.common import BlacklistType, BlacklistReason
from ..mixins.base import BaseModelMixin

Base = declarative_base()

class Blacklist(Base, BaseModelMixin):
    __tablename__ = "blacklists"
    
    entry_type = Column(SQLEnum(BlacklistType), nullable=False)
    value = Column(String(500), nullable=False, index=True)
    reason = Column(SQLEnum(BlacklistReason), nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<Blacklist(type={self.entry_type}, value={self.value})>"
