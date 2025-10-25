"""
Application Configuration
Handles environment variables, settings, and configuration management
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import BaseSettings, validator
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    APP_NAME: str = "fraud-detection-engine"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_PREFIX: str = "/api/v2"
    
    # Security
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    JWT_SECRET_KEY: str = "your-jwt-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    DATABASE_URL: str = "postgresql://fraud_user:fraud_password@localhost:5432/fraud_detection"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_TIMEOUT: int = 30
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: Optional[str] = None
    REDIS_MAX_CONNECTIONS: int = 100
    CACHE_TTL_SECONDS: int = 3600
    
    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_CONSUMER_GROUP: str = "fraud-detection-consumer"
    KAFKA_TOPICS_TRANSACTIONS: str = "raw-transactions"
    KAFKA_TOPICS_ALERTS: str = "fraud-alerts"
    
    # ML Models
    MODEL_REGISTRY_URL: str = "http://localhost:5000"
    MODEL_ARTIFACT_STORE: str = "s3://fraud-models"
    MODEL_CACHE_DIR: str = "/tmp/model_cache"
    MODEL_INFERENCE_TIMEOUT: float = 5.0
    MODEL_BATCH_SIZE: int = 1000
    
    # Feature Store
    FEATURE_STORE_URL: str = "localhost:6566"
    FEATURE_CACHE_TTL: int = 1800  # 30 minutes
    
    # Monitoring
    PROMETHEUS_PORT: int = 9090
    METRICS_ENABLED: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT: str = "1000/minute"
    RATE_LIMIT_FRAUD_DETECT: str = "500/minute"
    RATE_LIMIT_BURST: str = "100/second"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://fraud-dashboard.company.com"
    ]
    
    # WebSocket
    WS_MAX_CONNECTIONS: int = 1000
    WS_PING_INTERVAL: int = 30
    WS_PING_TIMEOUT: int = 10
    
    # External Services
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    ELASTICSEARCH_INDEX: str = "fraud-detection-logs"
    
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    
    # Business Rules
    FRAUD_THRESHOLD_HIGH: float = 0.8
    FRAUD_THRESHOLD_MEDIUM: float = 0.5
    FRAUD_THRESHOLD_LOW: float = 0.2
    
    # Performance
    ASYNC_DB_POOL_SIZE: int = 20
    ASYNC_DB_MAX_OVERFLOW: int = 0
    HTTP_TIMEOUT: int = 30
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v, values):
        if isinstance(v, str):
            return v
        # Build from components if individual parts are provided
        scheme = values.get("DB_SCHEME", "postgresql")
        user = values.get("DB_USER", "fraud_user")
        password = values.get("DB_PASSWORD", "fraud_password")
        host = values.get("DB_HOST", "localhost")
        port = values.get("DB_PORT", "5432")
        db = values.get("DB_NAME", "fraud_detection")
        return f"{scheme}://{user}:{password}@{host}:{port}/{db}"
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def is_testing(self) -> bool:
        return self.ENVIRONMENT.lower() == "testing"


class DatabaseSettings:
    """Database-specific settings"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    @property
    def connection_args(self) -> dict:
        return {
            "pool_size": self.settings.DATABASE_POOL_SIZE,
            "max_overflow": self.settings.DATABASE_MAX_OVERFLOW,
            "pool_timeout": self.settings.DATABASE_POOL_TIMEOUT,
            "pool_pre_ping": True,
            "echo": self.settings.DEBUG,
        }
    
    @property
    def async_connection_args(self) -> dict:
        return {
            "pool_size": self.settings.ASYNC_DB_POOL_SIZE,
            "max_overflow": self.settings.ASYNC_DB_MAX_OVERFLOW,
            "pool_pre_ping": True,
            "echo": self.settings.DEBUG,
        }


class RedisSettings:
    """Redis-specific settings"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    @property
    def connection_args(self) -> dict:
        args = {
            "max_connections": self.settings.REDIS_MAX_CONNECTIONS,
            "retry_on_timeout": True,
            "socket_keepalive": True,
            "socket_keepalive_options": {},
            "decode_responses": True,
        }
        
        if self.settings.REDIS_PASSWORD:
            args["password"] = self.settings.REDIS_PASSWORD
        
        return args


class SecuritySettings:
    """Security-related settings"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    @property
    def jwt_config(self) -> dict:
        return {
            "secret_key": self.settings.JWT_SECRET_KEY,
            "algorithm": self.settings.JWT_ALGORITHM,
            "access_token_expire_minutes": self.settings.ACCESS_TOKEN_EXPIRE_MINUTES,
            "refresh_token_expire_days": self.settings.REFRESH_TOKEN_EXPIRE_DAYS,
        }
    
    @property
    def password_config(self) -> dict:
        return {
            "schemes": ["bcrypt"],
            "deprecated": "auto",
            "bcrypt__rounds": 12,
        }


# Cache settings instance
@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Convenience functions
@lru_cache()
def get_database_settings() -> DatabaseSettings:
    return DatabaseSettings(get_settings())


@lru_cache()
def get_redis_settings() -> RedisSettings:
    return RedisSettings(get_settings())


@lru_cache()
def get_security_settings() -> SecuritySettings:
    return SecuritySettings(get_settings())
