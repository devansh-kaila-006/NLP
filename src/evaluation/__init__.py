"""
Evaluation Module for Multi-Modal RAG System

Provides comprehensive evaluation framework for validating system performance
across multiple dimensions: retrieval quality, temporal coherence, cross-modal
prediction, and system performance.
"""

from src.evaluation.base_evaluator import BaseEvaluator, EvaluationError, BenchmarkValidationError

__all__ = [
    'BaseEvaluator',
    'EvaluationError',
    'BenchmarkValidationError'
]