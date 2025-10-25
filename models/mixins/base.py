"""
Base Mixins for Database Models
Common functionality and fields for all models
"""

from sqlalchemy import Column, DateTime, String, Boolean, Text
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declared_attr
import uuid


class TimestampMixin:
    """Mixin to add timestamp fields to models"""
    
    @declared_attr
    def created_at(cls):
        return Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    @declared_attr  
    def updated_at(cls):
        return Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class UUIDMixin:
    """Mixin to add UUID primary key"""
    
    @declared_attr
    def id(cls):
        return Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))


class AuditMixin:
    """Mixin to add audit fields"""
    
    @declared_attr
    def created_by(cls):
        return Column(String(100), nullable=True)
    
    @declared_attr
    def updated_by(cls):
        return Column(String(100), nullable=True)
    
    @declared_attr
    def version(cls):
        return Column(String(50), nullable=True, default="1.0")


class SoftDeleteMixin:
    """Mixin to add soft delete functionality"""
    
    @declared_attr
    def is_deleted(cls):
        return Column(Boolean, default=False, nullable=False)
    
    @declared_attr
    def deleted_at(cls):
        return Column(DateTime(timezone=True), nullable=True)
    
    @declared_attr
    def deleted_by(cls):
        return Column(String(100), nullable=True)


class MetadataMixin:
    """Mixin to add metadata fields"""
    
    @declared_attr
    def metadata_json(cls):
        return Column(Text, nullable=True, comment="JSON metadata for additional fields")
    
    @declared_attr
    def tags(cls):
        return Column(Text, nullable=True, comment="Comma-separated tags")
    
    @declared_attr
    def notes(cls):
        return Column(Text, nullable=True, comment="Additional notes or comments")


class BaseMixin(UUIDMixin, TimestampMixin, AuditMixin):
    """Base mixin combining common functionality"""
    pass


class BaseModelMixin(BaseMixin, SoftDeleteMixin, MetadataMixin):
    """Complete base mixin with all common functionality"""
    pass
