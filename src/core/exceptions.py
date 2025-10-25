"""
Custom Exception Classes
Defines application-specific exceptions for better error handling
"""

from typing import Optional, Dict, Any


class FraudDetectionException(Exception):
    """Base exception for fraud detection system"""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class ValidationException(FraudDetectionException):
    """Exception for validation errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details=details
        )


class AuthenticationException(FraudDetectionException):
    """Exception for authentication errors"""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_ERROR"
        )


class AuthorizationException(FraudDetectionException):
    """Exception for authorization errors"""
    
    def __init__(self, message: str = "Access denied"):
        super().__init__(
            message=message,
            status_code=403,
            error_code="AUTHORIZATION_ERROR"
        )


class ResourceNotFoundException(FraudDetectionException):
    """Exception for resource not found errors"""
    
    def __init__(self, resource_type: str, resource_id: str):
        message = f"{resource_type} with ID '{resource_id}' not found"
        super().__init__(
            message=message,
            status_code=404,
            error_code="RESOURCE_NOT_FOUND",
            details={"resource_type": resource_type, "resource_id": resource_id}
        )


class ModelException(FraudDetectionException):
    """Exception for ML model errors"""
    
    def __init__(self, message: str, model_name: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="MODEL_ERROR",
            details={"model_name": model_name} if model_name else {}
        )


class ModelNotFoundException(ModelException):
    """Exception for model not found errors"""
    
    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' not found",
            model_name=model_name
        )
        self.status_code = 404
        self.error_code = "MODEL_NOT_FOUND"


class ModelLoadException(ModelException):
    """Exception for model loading errors"""
    
    def __init__(self, model_name: str, error_details: str):
        super().__init__(
            message=f"Failed to load model '{model_name}': {error_details}",
            model_name=model_name
        )
        self.error_code = "MODEL_LOAD_ERROR"


class ModelInferenceException(ModelException):
    """Exception for model inference errors"""
    
    def __init__(self, model_name: str, error_details: str):
        super().__init__(
            message=f"Model inference failed for '{model_name}': {error_details}",
            model_name=model_name
        )
        self.error_code = "MODEL_INFERENCE_ERROR"


class DatabaseException(FraudDetectionException):
    """Exception for database errors"""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="DATABASE_ERROR",
            details={"operation": operation} if operation else {}
        )


class CacheException(FraudDetectionException):
    """Exception for cache errors"""
    
    def __init__(self, message: str, cache_key: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="CACHE_ERROR",
            details={"cache_key": cache_key} if cache_key else {}
        )


class ExternalServiceException(FraudDetectionException):
    """Exception for external service errors"""
    
    def __init__(self, service_name: str, message: str, status_code: int = 502):
        super().__init__(
            message=f"External service '{service_name}' error: {message}",
            status_code=status_code,
            error_code="EXTERNAL_SERVICE_ERROR",
            details={"service_name": service_name}
        )


class RateLimitException(FraudDetectionException):
    """Exception for rate limit errors"""
    
    def __init__(self, limit: str, window: str):
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window}",
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            details={"limit": limit, "window": window}
        )


class ConfigurationException(FraudDetectionException):
    """Exception for configuration errors"""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key} if config_key else {}
        )


class FeatureEngineeringException(FraudDetectionException):
    """Exception for feature engineering errors"""
    
    def __init__(self, message: str, feature_name: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="FEATURE_ENGINEERING_ERROR",
            details={"feature_name": feature_name} if feature_name else {}
        )


class StreamingException(FraudDetectionException):
    """Exception for streaming/messaging errors"""
    
    def __init__(self, message: str, topic: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="STREAMING_ERROR",
            details={"topic": topic} if topic else {}
        )


class WebSocketException(FraudDetectionException):
    """Exception for WebSocket errors"""
    
    def __init__(self, message: str, connection_id: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="WEBSOCKET_ERROR",
            details={"connection_id": connection_id} if connection_id else {}
        )


# Exception mapping for HTTP status codes
EXCEPTION_STATUS_MAP = {
    ValidationException: 400,
    AuthenticationException: 401,
    AuthorizationException: 403,
    ResourceNotFoundException: 404,
    ModelNotFoundException: 404,
    RateLimitException: 429,
    ModelException: 500,
    DatabaseException: 500,
    CacheException: 500,
    ExternalServiceException: 502,
    ConfigurationException: 500,
    FeatureEngineeringException: 500,
    StreamingException: 500,
    WebSocketException: 500,
    FraudDetectionException: 500,
}


def get_exception_status_code(exception: Exception) -> int:
    """Get HTTP status code for exception"""
    return EXCEPTION_STATUS_MAP.get(type(exception), 500)


def format_exception_response(exception: FraudDetectionException, request_id: str) -> Dict[str, Any]:
    """Format exception for API response"""
    return {
        "status": "error",
        "error": {
            "code": exception.error_code,
            "message": exception.message,
            "details": exception.details
        },
        "metadata": {
            "request_id": request_id,
            "timestamp": int(__import__('time').time())
        }
    }
