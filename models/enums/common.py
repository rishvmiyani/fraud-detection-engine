"""
Common Enumerations
Shared enums across the fraud detection system
"""

from enum import Enum


class RiskLevel(str, Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Decision(str, Enum):
    """Fraud detection decision"""
    APPROVE = "approve"
    REVIEW = "review"
    BLOCK = "block"
    CHALLENGE = "challenge"


class TransactionStatus(str, Enum):
    """Transaction processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class PaymentMethod(str, Enum):
    """Payment method types"""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    PAYPAL = "paypal"
    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"
    BANK_TRANSFER = "bank_transfer"
    WIRE_TRANSFER = "wire_transfer"
    CRYPTOCURRENCY = "cryptocurrency"
    CASH = "cash"
    CHECK = "check"


class UserRole(str, Enum):
    """User role types"""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    SYSTEM = "system"
    API_USER = "api_user"


class UserStatus(str, Enum):
    """User account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    LOCKED = "locked"
    PENDING_VERIFICATION = "pending_verification"


class AlertStatus(str, Enum):
    """Alert processing status"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    FALSE_POSITIVE = "false_positive"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelStatus(str, Enum):
    """ML model status"""
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class ModelType(str, Enum):
    """ML model types"""
    FRAUD_DETECTION = "fraud_detection"
    RISK_SCORING = "risk_scoring"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"
    FEATURE_EXTRACTION = "feature_extraction"


class AuditAction(str, Enum):
    """Audit log actions"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    APPROVE = "approve"
    REJECT = "reject"
    DEPLOY = "deploy"
    TRAIN = "train"


class MerchantCategory(str, Enum):
    """Merchant business categories"""
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    FOOD_BEVERAGE = "food_beverage"
    TRAVEL = "travel"
    ENTERTAINMENT = "entertainment"
    HEALTHCARE = "healthcare"
    AUTOMOTIVE = "automotive"
    HOME_GARDEN = "home_garden"
    SPORTS_OUTDOORS = "sports_outdoors"
    BOOKS_MEDIA = "books_media"
    JEWELRY = "jewelry"
    FINANCIAL_SERVICES = "financial_services"
    GAMBLING = "gambling"
    ADULT_CONTENT = "adult_content"
    CRYPTOCURRENCY = "cryptocurrency"
    OTHER = "other"


class MerchantRisk(str, Enum):
    """Merchant risk classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PROHIBITED = "prohibited"


class BlacklistType(str, Enum):
    """Blacklist entry types"""
    IP_ADDRESS = "ip_address"
    EMAIL = "email"
    PHONE = "phone"
    DEVICE_ID = "device_id"
    USER_ID = "user_id"
    CREDIT_CARD = "credit_card"
    MERCHANT_ID = "merchant_id"


class BlacklistReason(str, Enum):
    """Reasons for blacklisting"""
    FRAUD = "fraud"
    ABUSE = "abuse"
    SPAM = "spam"
    POLICY_VIOLATION = "policy_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    CHARGEBACKS = "chargebacks"
    REGULATORY = "regulatory"
    MANUAL = "manual"


class DataSource(str, Enum):
    """Data source types"""
    API = "api"
    BATCH_UPLOAD = "batch_upload"
    STREAMING = "streaming"
    MANUAL_ENTRY = "manual_entry"
    EXTERNAL_FEED = "external_feed"
    WEBHOOK = "webhook"


class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"
