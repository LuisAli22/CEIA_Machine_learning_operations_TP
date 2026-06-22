"""
Data schemas for API requests and responses.

This module defines Pydantic models for data validation and serialization,
following the Interface Segregation Principle.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import datetime
import math


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
            "timestamp": "2026-06-21T23:57:15.357224",
            "model_loaded": True,
            "model_info": {
                "name": "cern_xgboost",
                "version": "2",
                "stage": "Production",
                "uri": "models:/cern_xgboost/Production",
                "loaded_at": "2026-06-21T23:55:27.207977"
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
            "name": "cern_xgboost",
            "version": "2",
            "stage": "Production",
            "run_id": None,
            "loaded_at": "2026-06-21T23:55:27.207977"
        }
    })


class CERNElectronPair(BaseModel):
    """
    CERN detector data for a pair of electrons from collision events.
    
    This schema accepts raw detector measurements and automatically calculates
    derived features needed for mass prediction.
    """
    
    # Raw detector measurements for electron 1
    pt1: float = Field(
        gt=0, 
        description="Transverse momentum of electron 1 (GeV/c)"
    )
    eta1: float = Field(
        ge=-5, 
        le=5, 
        description="Pseudorapidity of electron 1"
    )
    phi1: float = Field(
        ge=-math.pi, 
        le=math.pi, 
        description="Azimuthal angle of electron 1 (radians)"
    )
    
    # Raw detector measurements for electron 2
    pt2: float = Field(
        gt=0, 
        description="Transverse momentum of electron 2 (GeV/c)"
    )
    eta2: float = Field(
        ge=-5, 
        le=5, 
        description="Pseudorapidity of electron 2"
    )
    phi2: float = Field(
        ge=-math.pi, 
        le=math.pi, 
        description="Azimuthal angle of electron 2 (radians)"
    )
    
    # Charge information
    charge1: int = Field(description="Charge of electron 1 (1 or -1)")
    charge2: int = Field(description="Charge of electron 2 (1 or -1)")
    
    @field_validator('charge1', 'charge2')
    @classmethod
    def validate_charge(cls, v: int) -> int:
        """Validate that charge is either +1 or -1."""
        if v not in [-1, 1]:
            raise ValueError('Charge must be -1 or 1')
        return v
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "pt1": 11.4625,
            "eta1": 1.5,
            "phi1": 0.8,
            "pt2": 2.01051,
            "eta2": 0.223163,
            "phi2": 2.064423,
            "charge1": 1,
            "charge2": -1
        }
    })


class CERNPredictionResponse(BaseModel):
    """Response schema for CERN electron pair mass prediction."""
    
    predicted_mass: float = Field(
        description="Predicted invariant mass (GeV/c²)"
    )
    input_features: Dict[str, float] = Field(
        description="Calculated features used for prediction"
    )
    model_name: str = Field(description="Model name used")
    model_version: str = Field(description="Model version used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "predicted_mass": 14.408,
            "input_features": {
                "pt1": 11.4625,
                "pt2": 2.01051,
                "E_total": 29.025277,
                "delta_eta": 1.276837,
                "delta_phi": 1.264423,
                "delta_R": 1.796964,
                "pt_product": 23.045471,
                "pt_ratio": 5.701290,
                "is_os": 1.0
            },
            "model_name": "cern_xgboost",
            "model_version": "2",
            "timestamp": "2026-06-21T23:57:15.357224"
        }
    })
