"""
Predictor module for making predictions with loaded models.

This module handles prediction logic, following the Single Responsibility Principle.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

from model_loader import ModelLoaderInterface
from exceptions import PredictionException, InvalidInputException

logger = logging.getLogger(__name__)


class Predictor:
    """
    Predictor class for making predictions.

    This class handles data preprocessing and prediction logic.
    """

    def __init__(self, model_loader: ModelLoaderInterface):
        """
        Initialize the predictor.

        Args:
            model_loader: Model loader instance
        """
        self._model_loader = model_loader

    def _prepare_input(
        self, data: Union[List[Dict[str, Any]], Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Prepare input data for prediction.

        Args:
            data: Input data (dict or list of dicts)

        Returns:
            pd.DataFrame: Prepared input data

        Raises:
            InvalidInputException: If input data is invalid
        """
        try:
            # Convert to list if single dict
            if isinstance(data, dict):
                data = [data]

            # Convert to DataFrame
            df = pd.DataFrame(data)

            if df.empty:
                raise InvalidInputException(detail="Input data is empty")

            logger.info(f"Prepared input data with shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Failed to prepare input data: {str(e)}")
            raise InvalidInputException(detail=str(e))

    def _extract_predictions(
        self, predictions: Any, return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Extract and format predictions.

        Args:
            predictions: Raw predictions from model
            return_probabilities: Whether to return probabilities

        Returns:
            dict: Formatted predictions
        """
        result = {}

        # Handle different prediction types
        if isinstance(predictions, np.ndarray):
            result["predictions"] = predictions.tolist()
        elif isinstance(predictions, pd.DataFrame):
            result["predictions"] = predictions.values.tolist()
        elif isinstance(predictions, pd.Series):
            result["predictions"] = predictions.tolist()
        else:
            result["predictions"] = predictions

        # Add probabilities if available and requested
        if return_probabilities:
            try:
                model = self._model_loader.get_model()
                # Try to get predict_proba if available
                if hasattr(model, "predict_proba"):
                    probas = model.predict_proba(predictions)
                    result["probabilities"] = probas.tolist()
            except Exception as e:
                logger.warning(f"Could not extract probabilities: {str(e)}")
                result["probabilities"] = None

        return result

    def predict(
        self,
        data: Union[List[Dict[str, Any]], Dict[str, Any]],
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Make predictions on input data.

        Args:
            data: Input data for prediction
            return_probabilities: Whether to return prediction probabilities

        Returns:
            dict: Prediction results

        Raises:
            PredictionException: If prediction fails
        """
        try:
            # Get the model
            model = self._model_loader.get_model()

            # Prepare input data
            input_df = self._prepare_input(data)

            # Make prediction
            logger.info("Making prediction...")
            predictions = model.predict(input_df)

            # Extract and format predictions
            result = self._extract_predictions(predictions, return_probabilities)

            logger.info(f"Prediction completed successfully")
            return result

        except InvalidInputException:
            raise

        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.error(error_msg)
            raise PredictionException(detail=error_msg)

    def predict_batch(
        self,
        data_list: List[Union[List[Dict[str, Any]], Dict[str, Any]]],
        return_probabilities: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Make predictions on multiple batches of data.

        Args:
            data_list: List of input data batches
            return_probabilities: Whether to return prediction probabilities

        Returns:
            list: List of prediction results

        Raises:
            PredictionException: If prediction fails
        """
        results = []

        for i, data in enumerate(data_list):
            try:
                result = self.predict(data, return_probabilities)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch {i} prediction failed: {str(e)}")
                results.append({
                    "error": str(e),
                    "batch_index": i
                })

        return results
