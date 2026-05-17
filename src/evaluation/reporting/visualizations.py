"""
Visualization Generator for Multi-Modal RAG System

Creates charts and graphs for evaluation results visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

from src.utils.logger import LoggerMixin


class VisualizationGenerator(LoggerMixin):
    """
    Generate visualizations for evaluation results.

    Creates charts and graphs for different evaluation metrics
    including confusion matrices, precision-recall curves, and performance distributions.
    """

    def __init__(self, output_dir: Path = None):
        """
        Initialize visualization generator.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir) if output_dir else Path("data/evaluation/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)

    def create_confusion_matrix(self, confusion_matrix: List[List[int]],
                              labels: List[str],
                              title: str = "Confusion Matrix") -> Path:
        """
        Create confusion matrix heatmap.

        Args:
            confusion_matrix: Confusion matrix as list of lists
            labels: List of label names
            title: Chart title

        Returns:
            Path to saved visualization
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax)

        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        plt.tight_layout()

        output_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Confusion matrix saved to {output_path}")

        return output_path

    def create_precision_recall_curve(self, precision_values: Dict[str, float],
                                     recall_values: Dict[str, float],
                                     title: str = "Precision-Recall Curve") -> Path:
        """
        Create precision-recall curve.

        Args:
            precision_values: Dictionary of k to precision@k values
            recall_values: Dictionary of k to recall@k values
            title: Chart title

        Returns:
            Path to saved visualization
        """
        fig, ax = plt.subplots()

        k_values = sorted(precision_values.keys())
        precision = [precision_values[k] for k in k_values]
        recall = [recall_values.get(k.replace('P', 'R'), 0.0) for k in k_values]

        ax.plot(range(len(k_values)), precision, marker='o', label='Precision')
        ax.plot(range(len(k_values)), recall, marker='s', label='Recall')
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels(k_values)
        ax.set_xlabel('K')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        plt.tight_layout()

        output_path = self.output_dir / 'precision_recall_curve.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Precision-recall curve saved to {output_path}")

        return output_path

    def create_performance_distribution(self, query_times: List[float],
                                       title: str = "Query Time Distribution") -> Path:
        """
        Create query time distribution histogram.

        Args:
            query_times: List of query execution times
            title: Chart title

        Returns:
            Path to saved visualization
        """
        fig, ax = plt.subplots()

        ax.hist(query_times, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(query_times), color='red', linestyle='--',
                  label=f'Mean: {np.mean(query_times):.2f}s')
        ax.axvline(np.median(query_times), color='green', linestyle='--',
                  label=f'Median: {np.median(query_times):.2f}s')

        ax.set_xlabel('Query Time (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / 'performance_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Performance distribution saved to {output_path}")

        return output_path

    def create_modality_prediction_chart(self, predictions: Dict[str, int],
                                       ground_truth: Dict[str, int],
                                       title: str = "Modality Prediction Analysis") -> Path:
        """
        Create modality prediction comparison chart.

        Args:
            predictions: Dictionary of predicted modality counts
            ground_truth: Dictionary of ground truth modality counts
            title: Chart title

        Returns:
            Path to saved visualization
        """
        fig, ax = plt.subplots()

        modalities = list(predictions.keys())
        x = np.arange(len(modalities))
        width = 0.35

        pred_counts = [predictions[m] for m in modalities]
        truth_counts = [ground_truth.get(m, 0) for m in modalities]

        bars1 = ax.bar(x - width/2, pred_counts, width, label='Predicted', alpha=0.8)
        bars2 = ax.bar(x + width/2, truth_counts, width, label='Ground Truth', alpha=0.8)

        ax.set_xlabel('Modality')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(modalities)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / 'modality_prediction_chart.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Modality prediction chart saved to {output_path}")

        return output_path

    def create_coherence_analysis(self, coherence_metrics: Dict,
                                 title: str = "Temporal Coherence Analysis") -> Path:
        """
        Create temporal coherence analysis chart.

        Args:
            coherence_metrics: Dictionary of coherence metrics
            title: Chart title

        Returns:
            Path to saved visualization
        """
        fig, ax = plt.subplots()

        metric_names = list(coherence_metrics.keys())
        metric_values = [coherence_metrics[name].get('mean', 0.0) for name in metric_names]
        metric_errors = [coherence_metrics[name].get('std', 0.0) for name in metric_names]

        x_pos = np.arange(len(metric_names))
        ax.bar(x_pos, metric_values, yerr=metric_errors, align='center',
              alpha=0.8, capsize=5, color='skyblue', edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([name.replace('_', ' ').title() for name in metric_names],
                          rotation=45, ha='right')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        output_path = self.output_dir / 'coherence_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Coherence analysis chart saved to {output_path}")

        return output_path

    def create_all_visualizations(self, evaluation_results: Dict) -> Dict[str, Path]:
        """
        Create all relevant visualizations from evaluation results.

        Args:
            evaluation_results: Dictionary containing all evaluation results

        Returns:
            Dictionary mapping visualization names to file paths
        """
        visualizations = {}

        # Modality prediction visualizations
        if 'modality_prediction' in evaluation_results:
            modality_results = evaluation_results['modality_prediction']
            accuracy_metrics = modality_results.get('accuracy_metrics', {})

            # Confusion matrix
            if 'confusion_matrix' in accuracy_metrics:
                cm_data = accuracy_metrics['confusion_matrix']
                confusion_matrix = cm_data.get('confusion_matrix', [])
                labels = cm_data.get('labels', [])

                if confusion_matrix and labels:
                    viz_path = self.create_confusion_matrix(
                        confusion_matrix, labels,
                        "Modality Prediction Confusion Matrix"
                    )
                    visualizations['modality_confusion_matrix'] = viz_path

            # Modality prediction comparison
            predictions_summary = modality_results.get('predictions_summary', {})
            ground_truth_summary = modality_results.get('ground_truth_summary', {})

            if predictions_summary and ground_truth_summary:
                viz_path = self.create_modality_prediction_chart(
                    predictions_summary, ground_truth_summary,
                    "Modality Prediction Comparison"
                )
                visualizations['modality_prediction_comparison'] = viz_path

        # Retrieval quality visualizations
        if 'retrieval_quality' in evaluation_results:
            retrieval_results = evaluation_results['retrieval_quality']
            retrieval_metrics = retrieval_results.get('retrieval_metrics', {})

            # Precision-Recall curve
            precision_at_k = retrieval_metrics.get('precision_at_k', {})
            recall_at_k = retrieval_metrics.get('recall_at_k', {})

            if precision_at_k and recall_at_k:
                viz_path = self.create_precision_recall_curve(
                    precision_at_k, recall_at_k,
                    "Retrieval Precision-Recall Curve"
                )
                visualizations['retrieval_precision_recall'] = viz_path

        # Performance visualizations
        if 'performance' in evaluation_results:
            perf_results = evaluation_results['performance']
            # Query times would need to be extracted from individual results
            # This is a placeholder for performance distribution

        # Temporal coherence visualizations
        if 'temporal_coherence' in evaluation_results:
            coherence_results = evaluation_results['temporal_coherence']
            coherence_metrics = coherence_results.get('coherence_metrics', {}).get('summary', {})

            if coherence_metrics:
                viz_path = self.create_coherence_analysis(
                    coherence_metrics,
                    "Temporal Coherence Metrics"
                )
                visualizations['temporal_coherence_analysis'] = viz_path

        self.logger.info(f"Generated {len(visualizations)} visualizations")

        return visualizations