"""
Custom exceptions for the application.

This module defines custom exception classes for better error handling,
following the Open/Closed Principle.
"""


class ModelServiceException(Exception):
    """Base exception for model service errors."""

    def __init__(self, message: str, detail: str = None):
        """
        Initialize exception.

        Args:
            message: Error message
            detail: Detailed error information
        """
        self.message = message
        self.detail = detail
        super().__init__(self.message)


class ModelNotLoadedException(ModelServiceException):
    """Exception raised when model is not loaded."""

    def __init__(self, detail: str = None):
        """Initialize exception."""
        super().__init__(
            message="Model not loaded",
            detail=detail or "The model has not been loaded yet"
        )


class ModelLoadException(ModelServiceException):
    """Exception raised when model loading fails."""

    def __init__(self, detail: str = None):
        """Initialize exception."""
        super().__init__(
            message="Failed to load model",
            detail=detail or "An error occurred while loading the model"
        )


class PredictionException(ModelServiceException):
    """Exception raised when prediction fails."""

    def __init__(self, detail: str = None):
        """Initialize exception."""
        super().__init__(
            message="Prediction failed",
            detail=detail or "An error occurred during prediction"
        )


class InvalidInputException(ModelServiceException):
    """Exception raised when input data is invalid."""

    def __init__(self, detail: str = None):
        """Initialize exception."""
        super().__init__(
            message="Invalid input data",
            detail=detail or "The provided input data is invalid"
        )
