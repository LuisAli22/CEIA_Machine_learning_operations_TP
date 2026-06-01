"""
Data schemas for API requests and responses.

This module defines Pydantic models for data validation and serialization,
following the Interface Segregation Principle.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_info: Optional[Dict[str, Any]] = Field(
        None, description="Model metadata"
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00",
            "model_loaded": True,
            "model_info": {
                "name": "my_model",
                "version": "1"
            }
        }
    })


class PredictionRequest(BaseModel):
    """Base prediction request schema."""

    data: Union[List[Dict[str, Any]], Dict[str, Any]] = Field(
        ..., description="Input data for prediction"
    )
    return_probabilities: bool = Field(
        False, description="Return prediction probabilities"
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "data": [
                {"feature1": 1.0, "feature2": 2.0},
                {"feature1": 3.0, "feature2": 4.0}
            ],
            "return_probabilities": False
        }
    })


class PredictionResponse(BaseModel):
    """Prediction response schema."""

    predictions: List[Any] = Field(..., description="Model predictions")
    probabilities: Optional[List[List[float]]] = Field(
        None, description="Prediction probabilities"
    )
    model_name: str = Field(..., description="Model name used")
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "predictions": [0, 1],
            "probabilities": [[0.8, 0.2], [0.3, 0.7]],
            "model_name": "my_model",
            "model_version": "1",
            "timestamp": "2024-01-01T00:00:00"
        }
    })


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "error": "Model not loaded",
            "detail": "Please check model configuration",
            "timestamp": "2024-01-01T00:00:00"
        }
    })


class ModelInfo(BaseModel):
    """Model information schema."""

    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    stage: Optional[str] = Field(None, description="Model stage")
    run_id: Optional[str] = Field(None, description="MLflow run ID")
    loaded_at: Optional[datetime] = Field(None, description="Model load timestamp")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "name": "my_model",
            "version": "1",
            "stage": "Production",
            "run_id": "abc123",
            "loaded_at": "2024-01-01T00:00:00"
        }
    })
