"""
Models Package - Database Models, Schemas, and ML Models
Comprehensive data models for the fraud detection engine
"""

from .database.user import User
from .database.transaction import Transaction
from .database.alert import Alert
from .database.model_metadata import ModelMetadata
from .database.audit_log import AuditLog
from .database.merchant import Merchant
from .database.blacklist import Blacklist

# Import all database models for SQLAlchemy Base registration
__all__ = [
    # Database Models
    'User',
    'Transaction', 
    'Alert',
    'ModelMetadata',
    'AuditLog',
    'Merchant',
    'Blacklist',
]

# Version
__version__ = "2.0.0"
