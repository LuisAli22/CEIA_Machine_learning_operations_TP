"""
Configuration module for FastAPI application.

This module handles all configuration settings using Pydantic Settings,
following the Single Responsibility Principle.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application settings
    app_name: str = "ML Model Service"
    app_version: str = "1.0.0"
    debug: bool = False

    # MLflow settings
    mlflow_tracking_uri: str = "http://mlflow:5000"
    model_name: str = "cern_xgboost"
    model_version: str = "1"
    model_stage: str = "Production"  # Production, Staging, None

    # AWS/MinIO settings
    aws_access_key_id: str = "minio"
    aws_secret_access_key: str = "minio123"
    aws_endpoint_url_s3: str = "http://s3:9000"
    mlflow_s3_endpoint_url: str = "http://s3:9000"

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8800
    workers: int = 1

    # Model settings
    model_cache_enabled: bool = True
    prediction_timeout: int = 30

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings instance.
    """
    return Settings()
