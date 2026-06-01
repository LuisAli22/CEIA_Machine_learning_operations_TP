"""
Model loader module for MLflow models.

This module handles model loading and management from MLflow,
following the Single Responsibility Principle and Dependency Inversion Principle.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from datetime import datetime

import mlflow
from mlflow.pyfunc import PyFuncModel

from config import Settings
from exceptions import ModelLoadException, ModelNotLoadedException

logger = logging.getLogger(__name__)


class ModelLoaderInterface(ABC):

    @abstractmethod
    def load_model(self) -> Any:
        """Load the model."""
        pass

    @abstractmethod
    def get_model(self) -> Any:
        """Get the loaded model."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """Get model information."""
        pass


class MLflowModelLoader(ModelLoaderInterface):
    """
    MLflow model loader implementation.

    This class handles loading and caching of MLflow models.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the model loader.

        Args:
            settings: Application settings
        """
        self._settings = settings
        self._model: Optional[PyFuncModel] = None
        self._model_info: dict = {}
        self._loaded_at: Optional[datetime] = None
        self._configure_mlflow()

    def _configure_mlflow(self) -> None:
        """Configure MLflow client."""
        try:
            mlflow.set_tracking_uri(self._settings.mlflow_tracking_uri)
            logger.info(
                f"MLflow tracking URI set to: {self._settings.mlflow_tracking_uri}"
            )
        except Exception as e:
            logger.error(f"Failed to configure MLflow: {str(e)}")
            raise ModelLoadException(detail=str(e))

    def _get_model_uri(self) -> str:
        """
        Get the model URI based on configuration.

        Returns:
            str: Model URI for MLflow

        Raises:
            ModelLoadException: If model URI cannot be determined
        """
        try:
            if self._settings.model_stage and self._settings.model_stage != "None":
                # Load by stage (Production, Staging, etc.)
                model_uri = (
                    f"models:/{self._settings.model_name}/{self._settings.model_stage}"
                )
            else:
                # Load by version
                model_uri = (
                    f"models:/{self._settings.model_name}/{self._settings.model_version}"
                )

            logger.info(f"Model URI: {model_uri}")
            return model_uri

        except Exception as e:
            logger.error(f"Failed to determine model URI: {str(e)}")
            raise ModelLoadException(detail=f"Invalid model configuration: {str(e)}")

    def load_model(self) -> PyFuncModel:
        """
        Load the model from MLflow.

        Returns:
            PyFuncModel: Loaded MLflow model

        Raises:
            ModelLoadException: If model loading fails
        """
        try:
            model_uri = self._get_model_uri()
            logger.info(f"Loading model from: {model_uri}")

            self._model = mlflow.pyfunc.load_model(model_uri)
            self._loaded_at = datetime.utcnow()

            # Store model metadata
            self._model_info = {
                "name": self._settings.model_name,
                "version": self._settings.model_version,
                "stage": self._settings.model_stage,
                "uri": model_uri,
                "loaded_at": self._loaded_at.isoformat()
            }

            logger.info(f"Model loaded successfully: {self._model_info}")
            return self._model

        except mlflow.exceptions.MlflowException as e:
            error_msg = f"MLflow error while loading model: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadException(detail=error_msg)

        except Exception as e:
            error_msg = f"Unexpected error while loading model: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadException(detail=error_msg)

    def get_model(self) -> PyFuncModel:
        """
        Get the loaded model.

        Returns:
            PyFuncModel: The loaded model

        Raises:
            ModelNotLoadedException: If model is not loaded
        """
        if self._model is None:
            raise ModelNotLoadedException(
                detail="Model must be loaded before use. Call load_model() first."
            )
        return self._model

    def is_loaded(self) -> bool:
        """
        Check if model is loaded.

        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self._model is not None

    def get_model_info(self) -> dict:
        """
        Get model information.

        Returns:
            dict: Model metadata
        """
        return self._model_info.copy()

    def reload_model(self) -> PyFuncModel:
        """
        Reload the model from MLflow.

        Returns:
            PyFuncModel: Reloaded model
        """
        logger.info("Reloading model...")
        self._model = None
        return self.load_model()


def get_model_loader(settings: Settings) -> ModelLoaderInterface:
    """
    Factory function to get model loader instance.

    Args:
        settings: Application settings

    Returns:
        ModelLoaderInterface: Model loader instance
    """
    return MLflowModelLoader(settings)
