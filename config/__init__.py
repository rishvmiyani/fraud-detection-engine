"""
Configuration module for Real-Time Fraud Detection Engine
Handles all environment configurations, model parameters, and system settings
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseSettings, validator

class Settings(BaseSettings):
    """Base settings class with environment variable support"""
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Application
    APP_NAME: str = "fraud-detection-engine"
    APP_VERSION: str = "2.0.0"
    API_PREFIX: str = "/api/v2"
    
    # Security
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@dataclass
class ConfigManager:
    """Configuration manager for loading and validating configs"""
    
    config_dir: Path = Path(__file__).parent
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Replace environment variables
        return self._replace_env_vars(config)
    
    def _replace_env_vars(self, obj: Any) -> Any:
        """Recursively replace environment variables in config"""
        if isinstance(obj, dict):
            return {k: self._replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_env_vars(v) for v in obj]
        elif isinstance(obj, str) and obj.startswith('EOF{') and obj.endswith('}'):
            env_var = obj[3:-1]
            return os.getenv(env_var, obj)
        return obj

# Global configuration manager instance
config_manager = ConfigManager()
settings = Settings()
