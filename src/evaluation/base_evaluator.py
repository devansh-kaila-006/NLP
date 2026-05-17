"""
Base Evaluator Interface for Multi-Modal RAG System

Provides abstract interface and common functionality for all evaluation components.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np
from datetime import datetime

from src.utils.logger import LoggerMixin
from src.utils.helpers import Timer


class BaseEvaluator(LoggerMixin, ABC):
    """
    Abstract base class for all evaluators in the Multi-Modal RAG system.

    Provides standard evaluation lifecycle:
    1. Setup: Initialize evaluator with pipeline and benchmarks
    2. Execute: Run evaluation and collect metrics
    3. Analyze: Calculate statistics and validate results
    4. Report: Generate insights and recommendations
    """

    def __init__(self, pipeline: Any, config: Optional[Dict] = None):
        """
        Initialize the base evaluator.

        Args:
            pipeline: UnifiedMultiModalRAGPipeline instance
            config: Optional configuration dictionary
        """
        self.pipeline = pipeline
        self.config = config or {}
        self.results = {}
        self.metadata = {
            'evaluator_type': self.__class__.__name__,
            'timestamp': datetime.now().isoformat(),
            'pipeline_config': self._get_pipeline_config()
        }

    @abstractmethod
    def evaluate(self, benchmark_data: Dict) -> Dict:
        """
        Run evaluation and return metrics.

        Args:
            benchmark_data: Dictionary containing test queries and ground truth

        Returns:
            Dictionary containing evaluation metrics and results
        """
        pass

    @abstractmethod
    def get_required_benchmark_fields(self) -> List[str]:
        """
        Return list of required fields in benchmark data.

        Returns:
            List of required field names
        """
        pass

    def setup_evaluation(self, benchmark_data: Dict) -> None:
        """
        Validate benchmark data and setup evaluation environment.

        Args:
            benchmark_data: Dictionary containing test queries and ground truth

        Raises:
            ValueError: If benchmark data is missing required fields
        """
        required_fields = self.get_required_benchmark_fields()
        missing_fields = [field for field in required_fields if field not in benchmark_data]

        if missing_fields:
            raise ValueError(f"Missing required benchmark fields: {missing_fields}")

        self.logger.info(f"Evaluation setup complete for {len(benchmark_data.get('queries', []))} queries")

    def save_results(self, output_path: Path) -> None:
        """
        Save evaluation results to disk.

        Args:
            output_path: Path to save results (JSON format)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_with_metadata = {
            'metadata': self.metadata,
            'results': self.results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_with_metadata, f, indent=2)

        self.logger.info(f"Results saved to {output_path}")

    def load_results(self, input_path: Path) -> Dict:
        """
        Load evaluation results from disk.

        Args:
            input_path: Path to load results from

        Returns:
            Dictionary containing loaded results
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        self.results = loaded_data.get('results', {})
        self.metadata = loaded_data.get('metadata', {})

        self.logger.info(f"Results loaded from {input_path}")
        return self.results

    def calculate_statistics(self, metrics: List[float]) -> Dict[str, float]:
        """
        Calculate statistical summary for a list of metrics.

        Args:
            metrics: List of metric values

        Returns:
            Dictionary containing mean, std, min, max, median, percentiles
        """
        if not metrics:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'p25': 0.0,
                'p75': 0.0,
                'count': 0
            }

        metrics_array = np.array(metrics)

        return {
            'mean': float(np.mean(metrics_array)),
            'std': float(np.std(metrics_array)),
            'min': float(np.min(metrics_array)),
            'max': float(np.max(metrics_array)),
            'median': float(np.median(metrics_array)),
            'p25': float(np.percentile(metrics_array, 25)),
            'p75': float(np.percentile(metrics_array, 75)),
            'count': len(metrics)
        }

    def calculate_confidence_interval(self, metrics: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for a list of metrics.

        Args:
            metrics: List of metric values
            confidence: Confidence level (default: 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(metrics) < 2:
            return (0.0, 0.0)

        metrics_array = np.array(metrics)
        mean = np.mean(metrics_array)
        std_err = np.std(metrics_array) / np.sqrt(len(metrics_array))

        # Calculate critical value for confidence interval
        # Using t-distribution for small samples
        from scipy import stats
        degrees_of_freedom = len(metrics) - 1
        t_critical = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)

        margin_of_error = t_critical * std_err

        return (float(mean - margin_of_error), float(mean + margin_of_error))

    def format_metrics_for_display(self, metrics: Dict) -> str:
        """
        Format metrics dictionary for human-readable display.

        Args:
            metrics: Dictionary of metric names to values

        Returns:
            Formatted string representation
        """
        formatted_lines = []
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                formatted_lines.append(f"{metric_name}: {value:.4f}")
            elif isinstance(value, dict):
                formatted_lines.append(f"{metric_name}:")
                for sub_name, sub_value in value.items():
                    if isinstance(sub_value, float):
                        formatted_lines.append(f"  {sub_name}: {sub_value:.4f}")
                    else:
                        formatted_lines.append(f"  {sub_name}: {sub_value}")
            else:
                formatted_lines.append(f"{metric_name}: {value}")

        return "\n".join(formatted_lines)

    def _get_pipeline_config(self) -> Dict:
        """
        Extract relevant configuration from pipeline.

        Returns:
            Dictionary containing pipeline configuration
        """
        try:
            return {
                'use_reranker': getattr(self.pipeline, 'use_reranker', False),
                'include_aman': getattr(self.pipeline, 'include_aman', False),
                'pdf_chunks': len(getattr(self.pipeline, 'pdf_chunks', [])),
                'video_chunks': len(getattr(self.pipeline, 'video_chunks', [])),
                'aman_chunks': len(getattr(self.pipeline, 'aman_chunks', []))
            }
        except Exception as e:
            self.logger.warning(f"Could not extract pipeline config: {e}")
            return {}

    def validate_claim(self, metric_value: float, claim_value: float, threshold: float = 0.05) -> Dict:
        """
        Validate if a metric meets a claimed performance threshold.

        Args:
            metric_value: Actual measured value
            claim_value: Claimed/expected value
            threshold: Acceptable degradation threshold (default: 5%)

        Returns:
            Dictionary with validation results
        """
        difference = claim_value - metric_value
        relative_difference = difference / claim_value if claim_value != 0 else 0

        passed = metric_value >= (claim_value * (1 - threshold))

        return {
            'passed': passed,
            'metric_value': metric_value,
            'claim_value': claim_value,
            'difference': difference,
            'relative_difference': relative_difference,
            'threshold': threshold
        }


class EvaluationError(Exception):
    """Custom exception for evaluation errors."""

    pass


class BenchmarkValidationError(EvaluationError):
    """Exception raised when benchmark data validation fails."""

    pass