"""
Accuracy Metrics for Multi-Modal RAG System

Implements classification and prediction accuracy metrics for evaluating
cross-modal prediction and other classification tasks.
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)


class AccuracyMetrics:
    """
    Calculate classification and prediction accuracy metrics.

    Focuses on evaluating modality prediction accuracy with detailed
    analysis of prediction patterns and calibration.
    """

    @staticmethod
    def calculate_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
        """
        Calculate overall classification accuracy.

        Args:
            predictions: List of predicted labels
            ground_truth: List of ground truth labels

        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        if not predictions:
            return 0.0

        return accuracy_score(ground_truth, predictions)

    @staticmethod
    def calculate_confusion_matrix(predictions: List[str], ground_truth: List[str],
                                   labels: List[str]) -> Dict:
        """
        Calculate confusion matrix for classification analysis.

        Args:
            predictions: List of predicted labels
            ground_truth: List of ground truth labels
            labels: List of all possible labels

        Returns:
            Dictionary containing confusion matrix and analysis
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        cm = confusion_matrix(ground_truth, predictions, labels=labels)

        # Calculate per-class accuracy
        per_class_accuracy = {}
        for i, label in enumerate(labels):
            true_positives = cm[i, i]
            total_actual = cm[i, :].sum()

            if total_actual > 0:
                per_class_accuracy[label] = true_positives / total_actual
            else:
                per_class_accuracy[label] = 0.0

        return {
            'confusion_matrix': cm.tolist(),
            'per_class_accuracy': per_class_accuracy,
            'labels': labels
        }

    @staticmethod
    def calculate_per_class_metrics(predictions: List[str], ground_truth: List[str],
                                   labels: List[str]) -> Dict:
        """
        Calculate precision, recall, and F1 score for each class.

        Args:
            predictions: List of predicted labels
            ground_truth: List of ground truth labels
            labels: List of all possible labels

        Returns:
            Dictionary containing per-class metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth, predictions, labels=labels, average=None, zero_division=0
        )

        per_class_metrics = {}
        for i, label in enumerate(labels):
            per_class_metrics[label] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }

        # Calculate macro averages
        per_class_metrics['macro_avg'] = {
            'precision': float(np.mean(precision)),
            'recall': float(np.mean(recall)),
            'f1_score': float(np.mean(f1)),
            'support': int(np.sum(support))
        }

        # Calculate weighted averages
        per_class_metrics['weighted_avg'] = {
            'precision': float(np.average(precision, weights=support)),
            'recall': float(np.average(recall, weights=support)),
            'f1_score': float(np.average(f1, weights=support)),
            'support': int(np.sum(support))
        }

        return per_class_metrics

    @staticmethod
    def calculate_calibration(predictions: List[str], confidences: List[float],
                            ground_truth: List[str], n_bins: int = 10) -> Dict:
        """
        Calculate calibration metrics to assess confidence score reliability.

        Args:
            predictions: List of predicted labels
            confidences: List of confidence scores for predictions
            ground_truth: List of ground truth labels
            n_bins: Number of bins for calibration analysis

        Returns:
            Dictionary containing calibration metrics
        """
        if len(predictions) != len(confidences) or len(predictions) != len(ground_truth):
            raise ValueError("Predictions, confidences, and ground truth must have same length")

        # Calculate accuracy per confidence bin
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(n_bins):
            lower_bound = bin_boundaries[i]
            upper_bound = bin_boundaries[i + 1]

            # Find predictions in this bin
            in_bin_mask = (confidences >= lower_bound) & (confidences < upper_bound)
            if i == n_bins - 1:  # Include upper bound for last bin
                in_bin_mask = (confidences >= lower_bound) & (confidences <= upper_bound)

            bin_predictions = np.array(predictions)[in_bin_mask]
            bin_ground_truth = np.array(ground_truth)[in_bin_mask]
            bin_confidence_values = np.array(confidences)[in_bin_mask]

            if len(bin_predictions) > 0:
                bin_accuracy = accuracy_score(bin_ground_truth, bin_predictions)
                bin_mean_confidence = np.mean(bin_confidence_values)

                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_mean_confidence)
                bin_counts.append(len(bin_predictions))
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(0.0)
                bin_counts.append(0)

        # Calculate Expected Calibration Error (ECE)
        ece = 0.0
        total_samples = len(predictions)

        for i in range(n_bins):
            if bin_counts[i] > 0:
                bin_weight = bin_counts[i] / total_samples
                ece += bin_weight * abs(bin_accuracies[i] - bin_confidences[i])

        return {
            'expected_calibration_error': ece,
            'bin_boundaries': bin_boundaries.tolist(),
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts
        }

    @staticmethod
    def analyze_prediction_patterns(predictions: List[str], ground_truth: List[str],
                                   confidences: List[float], query_types: List[str]) -> Dict:
        """
        Analyze prediction patterns across different query types.

        Args:
            predictions: List of predicted labels
            ground_truth: List of ground truth labels
            confidences: List of confidence scores
            query_types: List of query type labels (e.g., 'conceptual', 'mathematical')

        Returns:
            Dictionary containing pattern analysis
        """
        if len(predictions) != len(ground_truth) or len(predictions) != len(query_types):
            raise ValueError("All input lists must have same length")

        pattern_analysis = defaultdict(lambda: {
            'total': 0,
            'correct': 0,
            'confidences': [],
            'predicted_modalities': defaultdict(int)
        })

        for pred, truth, conf, qtype in zip(predictions, ground_truth, confidences, query_types):
            pattern_analysis[qtype]['total'] += 1
            pattern_analysis[qtype]['confidences'].append(conf)
            pattern_analysis[qtype]['predicted_modalities'][pred] += 1

            if pred == truth:
                pattern_analysis[qtype]['correct'] += 1

        # Calculate summary statistics
        summary = {}
        for qtype, stats in pattern_analysis.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            mean_confidence = np.mean(stats['confidences']) if stats['confidences'] else 0.0

            summary[qtype] = {
                'accuracy': accuracy,
                'count': stats['total'],
                'mean_confidence': mean_confidence,
                'prediction_distribution': dict(stats['predicted_modalities'])
            }

        return summary

    @staticmethod
    def calculate_all_metrics(predictions: List[str], ground_truth: List[str],
                            confidences: List[float], labels: List[str],
                            query_types: List[str] = None) -> Dict:
        """
        Calculate comprehensive accuracy metrics.

        Args:
            predictions: List of predicted labels
            ground_truth: List of ground truth labels
            confidences: List of confidence scores
            labels: List of all possible labels
            query_types: Optional list of query type labels

        Returns:
            Dictionary containing all accuracy metrics
        """
        metrics = {
            'overall_accuracy': AccuracyMetrics.calculate_accuracy(predictions, ground_truth),
            'confusion_matrix': AccuracyMetrics.calculate_confusion_matrix(predictions, ground_truth, labels),
            'per_class_metrics': AccuracyMetrics.calculate_per_class_metrics(predictions, ground_truth, labels),
            'calibration': AccuracyMetrics.calculate_calibration(predictions, confidences, ground_truth)
        }

        if query_types:
            metrics['pattern_analysis'] = AccuracyMetrics.analyze_prediction_patterns(
                predictions, ground_truth, confidences, query_types
            )

        return metrics