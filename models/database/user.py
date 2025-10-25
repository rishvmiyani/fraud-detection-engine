"""
User Database Model
Represents system users with authentication and authorization
"""

from sqlalchemy import Column, String, Text, Boolean, DateTime, Enum as SQLEnum, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from passlib.context import CryptContext
from datetime import datetime, timedelta
import json

from ..enums.common import UserRole, UserStatus
from ..mixins.base import BaseModelMixin

Base = declarative_base()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class User(Base, BaseModelMixin):
    """User model for authentication and authorization"""
    
    __tablename__ = "users"
    __table_args__ = (
        {'comment': 'System users with roles and permissions'}
    )
    
    # Basic Information
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    
    # Authentication
    password_hash = Column(String(255), nullable=False)
    salt = Column(String(100), nullable=True)
    
    # Status and Role
    status = Column(SQLEnum(UserStatus), default=UserStatus.ACTIVE, nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.VIEWER, nullable=False)
    
    # Security
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    phone_verified = Column(Boolean, default=False, nullable=False)
    two_factor_enabled = Column(Boolean, default=False, nullable=False)
    
    # Contact Information
    phone = Column(String(20), nullable=True)
    department = Column(String(100), nullable=True)
    job_title = Column(String(100), nullable=True)
    
    # Login Tracking
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    last_login_ip = Column(String(45), nullable=True)  # IPv6 compatible
    login_count = Column(Integer, default=0, nullable=False)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    
    # Security Settings
    password_expires_at = Column(DateTime(timezone=True), nullable=True)
    must_change_password = Column(Boolean, default=False, nullable=False)
    
    # API Access
    api_key = Column(String(255), unique=True, nullable=True, index=True)
    api_key_expires_at = Column(DateTime(timezone=True), nullable=True)
    api_calls_count = Column(Integer, default=0, nullable=False)
    api_rate_limit = Column(Integer, default=1000, nullable=False)  # per minute
    
    # Preferences
    timezone = Column(String(50), default="UTC", nullable=False)
    language = Column(String(10), default="en", nullable=False)
    theme = Column(String(20), default="light", nullable=False)
    
    # Notifications
    email_notifications = Column(Boolean, default=True, nullable=False)
    sms_notifications = Column(Boolean, default=False, nullable=False)
    push_notifications = Column(Boolean, default=True, nullable=False)
    
    # Additional Data
    permissions = Column(Text, nullable=True, comment="JSON array of permissions")
    settings = Column(Text, nullable=True, comment="JSON user settings")
    
    # Relationships
    transactions = relationship("Transaction", back_populates="created_by_user", lazy="dynamic")
    alerts = relationship("Alert", back_populates="assigned_to_user", lazy="dynamic")
    audit_logs = relationship("AuditLog", back_populates="user", lazy="dynamic")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"
    
    def set_password(self, password: str) -> None:
        """Hash and set password"""
        self.password_hash = pwd_context.hash(password)
    
    def verify_password(self, password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(password, self.password_hash)
    
    def check_password_expiry(self) -> bool:
        """Check if password has expired"""
        if not self.password_expires_at:
            return False
        return datetime.utcnow() > self.password_expires_at
    
    def is_locked(self) -> bool:
        """Check if user account is locked"""
        if not self.locked_until:
            return False
        return datetime.utcnow() < self.locked_until
    
    def lock_account(self, duration_minutes: int = 30) -> None:
        """Lock user account for specified duration"""
        self.locked_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
        self.status = UserStatus.LOCKED
    
    def unlock_account(self) -> None:
        """Unlock user account"""
        self.locked_until = None
        self.failed_login_attempts = 0
        if self.status == UserStatus.LOCKED:
            self.status = UserStatus.ACTIVE
    
    def increment_failed_login(self, max_attempts: int = 5) -> None:
        """Increment failed login attempts and lock if necessary"""
        self.failed_login_attempts += 1
        if self.failed_login_attempts >= max_attempts:
            self.lock_account()
    
    def record_successful_login(self, ip_address: str = None) -> None:
        """Record successful login"""
        self.last_login_at = datetime.utcnow()
        self.last_login_ip = ip_address
        self.login_count += 1
        self.failed_login_attempts = 0
        if self.locked_until:
            self.unlock_account()
    
    def get_permissions(self) -> list:
        """Get user permissions as list"""
        if not self.permissions:
            return []
        try:
            return json.loads(self.permissions)
        except (json.JSONDecodeError, TypeError):
            return []
    
    def set_permissions(self, permissions: list) -> None:
        """Set user permissions from list"""
        self.permissions = json.dumps(permissions) if permissions else None
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        permissions = self.get_permissions()
        return permission in permissions
    
    def get_settings(self) -> dict:
        """Get user settings as dict"""
        if not self.settings:
            return {}
        try:
            return json.loads(self.settings)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def set_settings(self, settings: dict) -> None:
        """Set user settings from dict"""
        self.settings = json.dumps(settings) if settings else None
    
    def get_full_name(self) -> str:
        """Get user's full name"""
        return f"{self.first_name} {self.last_name}".strip()
    
    def is_admin(self) -> bool:
        """Check if user is admin"""
        return self.role == UserRole.ADMIN
    
    def can_access_api(self) -> bool:
        """Check if user can access API"""
        return (
            self.is_active and 
            self.status == UserStatus.ACTIVE and 
            not self.is_locked() and
            self.api_key is not None
        )
    
    def to_dict(self, include_sensitive: bool = False) -> dict:
        """Convert user to dictionary"""
        data = {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'full_name': self.get_full_name(),
            'status': self.status.value,
            'role': self.role.value,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'department': self.department,
            'job_title': self.job_title,
            'last_login_at': self.last_login_at.isoformat() if self.last_login_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        
        if include_sensitive:
            data.update({
                'phone': self.phone,
                'last_login_ip': self.last_login_ip,
                'login_count': self.login_count,
                'failed_login_attempts': self.failed_login_attempts,
                'permissions': self.get_permissions(),
                'settings': self.get_settings(),
            })
        
        return data
