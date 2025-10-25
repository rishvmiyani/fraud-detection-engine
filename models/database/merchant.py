from sqlalchemy import Column, String, Float, DateTime, Boolean, Integer, Enum as SQLEnum, Text
from sqlalchemy.ext.declarative import declarative_base
from ..enums.common import MerchantCategory, MerchantRisk
from ..mixins.base import BaseModelMixin

Base = declarative_base()

class Merchant(Base, BaseModelMixin):
    __tablename__ = "merchants"
    
    merchant_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(500), nullable=False)
    category = Column(SQLEnum(MerchantCategory), nullable=False)
    risk_level = Column(SQLEnum(MerchantRisk), default=MerchantRisk.MEDIUM)
    country = Column(String(2), nullable=True)
    website = Column(String(500), nullable=True)
    
    def __repr__(self):
        return f"<Merchant(id={self.merchant_id}, name={self.name})>"
