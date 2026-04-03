"""
Module: utils.exceptions

Description:
    Custom exception classes for the RAG system

Inputs:
    - None (exception definitions)

Outputs:
    - Custom exception classes

Dependencies:
    - builtins.Exception

Usage:
    >>> from src.utils.exceptions import PDFProcessingError
    >>> raise PDFProcessingError("Failed to process PDF file")
"""

from typing import Optional, Dict, Any


class RAGSystemError(Exception):
    """Base exception class for RAG system errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize RAG system error.

        Args:
            message: Error message
            details: Optional dictionary with additional error details
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """String representation of the error."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class PDFProcessingError(RAGSystemError):
    """Exception raised when PDF processing fails."""

    pass


class VideoProcessingError(RAGSystemError):
    """Exception raised when video processing fails."""

    pass


class TranscriptionError(RAGSystemError):
    """Exception raised when transcription fails."""

    pass


class EmbeddingError(RAGSystemError):
    """Exception raised when embedding generation fails."""

    pass


class VectorStoreError(RAGSystemError):
    """Exception raised when vector store operations fail."""

    pass


class RetrievalError(RAGSystemError):
    """Exception raised when retrieval operations fail."""

    pass


class LLMError(RAGSystemError):
    """Exception raised when LLM operations fail."""

    pass


class ConfigurationError(RAGSystemError):
    """Exception raised when configuration is invalid or missing."""

    pass


class ValidationError(RAGSystemError):
    """Exception raised when input validation fails."""

    pass

