"""
FastAPI application for ML model serving.

This module implements the main FastAPI application with proper error handling,
logging, and dependency injection, following SOLID principles and PEP 8.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import Settings, get_settings
from schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    ErrorResponse,
    ModelInfo
)
from model_loader import ModelLoaderInterface, get_model_loader
from predictor import Predictor
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
    title="ML Model Service",
    description="Production-ready ML model serving API with MLflow integration",
    version="1.0.0",
    lifespan=lifespan
)

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
    summary="Root endpoint"
)
async def read_root() -> Dict[str, str]:
    """
    Root endpoint.

    Returns:
        dict: Welcome message
    """
    settings = get_settings()
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs"
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
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Make predictions",
    responses={
        200: {"description": "Successful prediction"},
        422: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    }
)
async def predict(
    request: PredictionRequest,
    predictor: Predictor = Depends(get_predictor),
    settings: Settings = Depends(get_settings)
) -> PredictionResponse:
    """
    Make predictions on input data.

    Args:
        request: Prediction request with input data
        predictor: Predictor instance (injected)
        settings: Application settings (injected)

    Returns:
        PredictionResponse: Prediction results

    Raises:
        HTTPException: If prediction fails
    """
    logger.info("Received prediction request")

    # Make prediction
    result = predictor.predict(
        data=request.data,
        return_probabilities=request.return_probabilities
    )

    # Get model info
    model_info = model_loader.get_model_info()

    # Build response
    response = PredictionResponse(
        predictions=result["predictions"],
        probabilities=result.get("probabilities"),
        model_name=model_info.get("name", settings.model_name),
        model_version=model_info.get("version", settings.model_version)
    )

    logger.info("Prediction request completed successfully")
    return response


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
