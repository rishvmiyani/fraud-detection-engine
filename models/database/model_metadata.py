from sqlalchemy import Column, String, Float, DateTime, Boolean, Text, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from ..enums.common import ModelStatus, ModelType
from ..mixins.base import BaseModelMixin

Base = declarative_base()

class ModelMetadata(Base, BaseModelMixin):
    __tablename__ = "model_metadata"
    
    model_name = Column(String(200), unique=True, nullable=False)
    model_version = Column(String(100), nullable=False)
    model_type = Column(SQLEnum(ModelType), nullable=False)
    status = Column(SQLEnum(ModelStatus), default=ModelStatus.TRAINING)
    accuracy = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<ModelMetadata(name={self.model_name}, version={self.model_version})>"
