"""
FastAPI application for ML model serving.

This module implements the main FastAPI application with proper error handling,
logging, and dependency injection, following SOLID principles and PEP 8.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html

from config import Settings, get_settings
from schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    ErrorResponse,
    ModelInfo,
    CERNElectronPair,
    CERNPredictionResponse
)
from model_loader import ModelLoaderInterface, get_model_loader
from predictor import Predictor
from cern_features import calculate_cern_features
from exceptions import (
    ModelServiceException,
    ModelNotLoadedException,
    ModelLoadException,
    PredictionException,
    InvalidInputException
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global model loader instance
model_loader: ModelLoaderInterface = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    global model_loader
    settings = get_settings()

    logger.info("Starting application...")
    logger.info(f"Application: {settings.app_name} v{settings.app_version}")

    try:
        # Initialize model loader
        model_loader = get_model_loader(settings)

        # Load model
        logger.info("Loading model...")
        model_loader.load_model()
        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model during startup: {str(e)}")
        logger.warning("Application will start without a loaded model")

    yield

    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI application
app = FastAPI(
    title="CERN Particle Physics ML Service",
    description="""
Servicio de Machine Learning para predicción de masa invariante de pares de electrones 
a partir de datos de colisiones de partículas del experimento CMS (CERN).

**Características principales:**
- Predicción de masa invariante con modelo XGBoost entrenado en datos reales de CERN
- Validación automática de rangos físicos
- Cálculo automático de características derivadas
- Integración con MLflow para versionado de modelos
- Monitoreo del estado del servicio

**Uso recomendado:** POST /predict
    """,
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "CEIA MLOps Team",
        "url": "https://github.com/yourusername/CEIA_Machine_learning_operations_TP",
    },
    license_info={
        "name": "MIT",
    },
    openapi_tags=[
        {
            "name": "Prediction",
            "description": "Predicción de masa invariante para datos de CERN"
        },
        {
            "name": "Model",
            "description": "Gestión y consulta de modelos"
        },
        {
            "name": "Health",
            "description": "Monitoreo del servicio"
        },
        {
            "name": "General",
            "description": "Información general"
        }
    ],
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,  # Hide models section by default
        "docExpansion": "list",  # Show only tags by default
        "filter": True,  # Enable search filter
        "syntaxHighlight.theme": "monokai"  # Code highlighting theme
    }
)

# Mount static files for custom CSS
import os
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency injection
def get_predictor() -> Predictor:
    """
    Get predictor instance with dependency injection.

    Returns:
        Predictor: Predictor instance

    Raises:
        HTTPException: If model is not loaded
    """
    if model_loader is None or not model_loader.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return Predictor(model_loader)


# Exception handlers
@app.exception_handler(ModelNotLoadedException)
async def model_not_loaded_handler(request, exc: ModelNotLoadedException):
    """Handle model not loaded exceptions."""
    logger.error(f"Model not loaded: {exc.detail}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error=exc.message,
            detail=exc.detail
        ).model_dump()
    )


@app.exception_handler(ModelLoadException)
async def model_load_handler(request, exc: ModelLoadException):
    """Handle model load exceptions."""
    logger.error(f"Model load error: {exc.detail}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=exc.message,
            detail=exc.detail
        ).model_dump()
    )


@app.exception_handler(PredictionException)
async def prediction_handler(request, exc: PredictionException):
    """Handle prediction exceptions."""
    logger.error(f"Prediction error: {exc.detail}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=exc.message,
            detail=exc.detail
        ).model_dump()
    )


@app.exception_handler(InvalidInputException)
async def invalid_input_handler(request, exc: InvalidInputException):
    """Handle invalid input exceptions."""
    logger.error(f"Invalid input: {exc.detail}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error=exc.message,
            detail=exc.detail
        ).model_dump()
    )


@app.exception_handler(ModelServiceException)
async def model_service_handler(request, exc: ModelServiceException):
    """Handle generic model service exceptions."""
    logger.error(f"Model service error: {exc.detail}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=exc.message,
            detail=exc.detail
        ).model_dump()
    )


# API Routes
@app.get(
    "/",
    response_model=Dict[str, str],
    tags=["General"],
    summary="Información de la API"
)
async def read_root() -> Dict[str, str]:
    """
    Información básica sobre el servicio de predicción de masa invariante.
    """
    settings = get_settings()
    return {
        "service": "CERN Particle Physics ML Service",
        "description": "Predicción de masa invariante de pares de electrones",
        "version": settings.app_version,
        "model": "XGBoost trained on CMS data",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "predict": "/predict"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        HealthResponse: Service health status
    """
    model_loaded = model_loader is not None and model_loader.is_loaded()
    model_info = model_loader.get_model_info() if model_loaded else None

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_info=model_info
    )


@app.get(
    "/model/info",
    response_model=ModelInfo,
    tags=["Model"],
    summary="Get model information"
)
async def get_model_info() -> ModelInfo:
    """
    Get information about the loaded model.

    Returns:
        ModelInfo: Model metadata

    Raises:
        HTTPException: If model is not loaded
    """
    if model_loader is None or not model_loader.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    info = model_loader.get_model_info()
    return ModelInfo(**info)


@app.post(
    "/model/reload",
    response_model=Dict[str, str],
    tags=["Model"],
    summary="Reload the model"
)
async def reload_model(settings: Settings = Depends(get_settings)) -> Dict[str, str]:
    """
    Reload the model from MLflow.

    Args:
        settings: Application settings

    Returns:
        dict: Reload status message

    Raises:
        HTTPException: If reload fails
    """
    global model_loader

    try:
        if model_loader is None:
            model_loader = get_model_loader(settings)

        logger.info("Reloading model...")
        model_loader.reload_model()
        logger.info("Model reloaded successfully")

        return {
            "message": "Model reloaded successfully",
            "model_info": str(model_loader.get_model_info())
        }

    except Exception as e:
        logger.error(f"Failed to reload model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}"
        )


@app.post(
    "/predict",
    response_model=CERNPredictionResponse,
    tags=["Prediction"],
    summary="Predicción de masa invariante",
    responses={
        200: {"description": "Successful prediction"},
        422: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    }
)
async def predict(
    electron_pair: CERNElectronPair,
    predictor: Predictor = Depends(get_predictor),
    settings: Settings = Depends(get_settings)
) -> CERNPredictionResponse:
    """
    Predecir masa invariante de un par de electrones del experimento CMS.
    
    Entrada:
    - pt1, pt2: Momento transversal de cada electrón (GeV/c), debe ser > 0
    - eta1, eta2: Pseudorapidez, rango [-5, 5]
    - phi1, phi2: Ángulo azimutal en radianes, rango [-π, π]
    - charge1, charge2: Carga eléctrica, valores {-1, 1}
    
    El backend calcula automáticamente las características derivadas:
    E_total, delta_eta, delta_phi, delta_R, pt_product, pt_ratio, is_os
    
    Retorna:
    - predicted_mass: Masa invariante estimada (GeV/c²)
    - input_features: Todas las características calculadas
    - model_name, model_version: Información del modelo
    """
    logger.info(f"Received CERN prediction request: pt1={electron_pair.pt1}, pt2={electron_pair.pt2}")
    
    try:
        # Calculate features from raw measurements
        features = calculate_cern_features(electron_pair)
        logger.info(f"Calculated features: {features}")
        
        # Make prediction
        result = predictor.predict(
            data=[features],
            return_probabilities=False
        )
        
        # Get model info
        model_info = model_loader.get_model_info()
        
        # Build response
        response = CERNPredictionResponse(
            predicted_mass=float(result["predictions"][0]),
            input_features=features,
            model_name=model_info.get("name", settings.model_name),
            model_version=model_info.get("version", settings.model_version)
        )
        
        logger.info(f"Prediction completed: mass={response.predicted_mass:.4f} GeV/c²")
        return response
        
    except Exception as e:
        logger.error(f"CERN prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.workers,
        reload=settings.debug
    )
