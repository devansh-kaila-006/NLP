"""
Modality Prediction Evaluator for Multi-Modal RAG System

Evaluates the cross-modal prediction system that automatically selects
the best content modality (video/PDF/Aman.ai) based on query characteristics.
"""

from typing import Dict, List, Any
import numpy as np

from src.evaluation.base_evaluator import BaseEvaluator, BenchmarkValidationError
from src.evaluation.metrics.accuracy_metrics import AccuracyMetrics


class ModalityPredictionEvaluator(BaseEvaluator):
    """
    Evaluator for cross-modal prediction accuracy.

    Validates the system's ability to predict which content modality will
    provide the best answers for a given query.
    """

    def __init__(self, pipeline: Any, config: Dict = None):
        """
        Initialize modality prediction evaluator.

        Args:
            pipeline: UnifiedMultiModalRAGPipeline instance
            config: Optional configuration dictionary
        """
        super().__init__(pipeline, config)
        self.target_accuracy = 0.85  # Claim: 85%+ accuracy

    def get_required_benchmark_fields(self) -> List[str]:
        """
        Return required fields for modality prediction benchmark.

        Returns:
            List of required field names
        """
        return ['queries', 'expected_modality']

    def evaluate(self, benchmark_data: Dict) -> Dict:
        """
        Run modality prediction evaluation.

        Args:
            benchmark_data: Dictionary containing:
                - queries: List of test queries
                - expected_modality: List of expected modality labels
                - query_types: Optional list of query type labels
                - confidence_scores: Optional list of confidence thresholds

        Returns:
            Dictionary containing evaluation metrics
        """
        self.setup_evaluation(benchmark_data)

        queries = benchmark_data['queries']
        expected_modalities = benchmark_data['expected_modality']
        query_types = benchmark_data.get('query_types', None)

        if len(queries) != len(expected_modalities):
            raise BenchmarkValidationError(
                f"Number of queries ({len(queries)}) must match "
                f"number of expected modalities ({len(expected_modalities)})"
            )

        self.logger.info(f"Evaluating modality prediction for {len(queries)} queries")

        predictions = []
        confidences = []
        modality_scores_list = []

        with Timer('modality_prediction_evaluation') as timer:
            for i, query in enumerate(queries):
                try:
                    # Get modality prediction from pipeline
                    modality_scores = self.pipeline._predict_modality(query)
                    predicted_modality = max(modality_scores, key=modality_scores.get)
                    confidence = modality_scores[predicted_modality]

                    predictions.append(predicted_modality)
                    confidences.append(confidence)
                    modality_scores_list.append(modality_scores)

                except Exception as e:
                    self.logger.error(f"Error predicting modality for query '{query}': {e}")
                    # Use default fallback
                    predictions.append('video')  # Default modality
                    confidences.append(0.0)
                    modality_scores_list.append({'video': 0.0, 'pdf': 0.0, 'aman': 0.0})

        # Calculate accuracy metrics
        labels = ['video', 'pdf', 'aman']
        accuracy_results = AccuracyMetrics.calculate_all_metrics(
            predictions=predictions,
            ground_truth=expected_modalities,
            confidences=confidences,
            labels=labels,
            query_types=query_types
        )

        # Validate against claimed accuracy
        overall_accuracy = accuracy_results['overall_accuracy']
        validation_result = self.validate_claim(
            metric_value=overall_accuracy,
            claim_value=self.target_accuracy,
            threshold=self.config.get('accuracy_threshold', 0.05)
        )

        # Compile results
        self.results = {
            'num_queries': len(queries),
            'overall_accuracy': overall_accuracy,
            'validation': validation_result,
            'accuracy_metrics': accuracy_results,
            'evaluation_time_seconds': timer.elapsed_time,
            'predictions_summary': {
                'video_predictions': sum(1 for p in predictions if p == 'video'),
                'pdf_predictions': sum(1 for p in predictions if p == 'pdf'),
                'aman_predictions': sum(1 for p in predictions if p == 'aman')
            },
            'ground_truth_summary': {
                'video_expected': sum(1 for g in expected_modalities if g == 'video'),
                'pdf_expected': sum(1 for g in expected_modalities if g == 'pdf'),
                'aman_expected': sum(1 for g in expected_modalities if g == 'aman')
            }
        }

        # Add detailed analysis if query types provided
        if query_types:
            self.results['query_type_analysis'] = accuracy_results.get('pattern_analysis', {})

        self.logger.info(f"Modality prediction evaluation complete: {overall_accuracy:.2%} accuracy")

        return self.results

    def analyze_prediction_errors(self, benchmark_data: Dict) -> Dict:
        """
        Analyze prediction errors to identify patterns.

        Args:
            benchmark_data: Benchmark data with queries and expected modalities

        Returns:
            Dictionary containing error analysis
        """
        queries = benchmark_data['queries']
        expected_modalities = benchmark_data['expected_modality']

        errors = []
        for i, (query, expected) in enumerate(zip(queries, expected_modalities)):
            try:
                modality_scores = self.pipeline._predict_modality(query)
                predicted = max(modality_scores, key=modality_scores.get)

                if predicted != expected:
                    errors.append({
                        'query': query,
                        'expected': expected,
                        'predicted': predicted,
                        'confidence': modality_scores[predicted],
                        'all_scores': modality_scores
                    })
            except Exception as e:
                self.logger.error(f"Error analyzing query '{query}': {e}")

        # Analyze error patterns
        error_patterns = {}
        for modality in ['video', 'pdf', 'aman']:
            modality_errors = [e for e in errors if e['expected'] == modality]
            error_patterns[modality] = {
                'count': len(modality_errors),
                'common_mispredictions': {}
            }

            # Count common mispredictions
            mispred_counts = {}
            for error in modality_errors:
                predicted = error['predicted']
                mispred_counts[predicted] = mispred_counts.get(predicted, 0) + 1

            error_patterns[modality]['common_mispredictions'] = mispred_counts

        return {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(queries) if queries else 0.0,
            'error_examples': errors[:5],  # First 5 errors
            'error_patterns': error_patterns
        }