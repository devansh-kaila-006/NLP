"""
Comprehensive Multi-Modal RAG Quality Evaluator
Combines RAGAS framework metrics with custom multi-modal metrics for complete RAG evaluation
"""

import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import time

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.evaluation.frameworks.ragas_integration import RAGASIntegration, RAGASEvaluationResult
    from src.evaluation.metrics.multimodal_metrics import (
        MultiModalRAGEvaluator,
        MultiModalMetricsResult
    )
except ImportError:
    # Handle import errors gracefully
    RAGASIntegration = None
    MultiModalRAGEvaluator = None
    logging.warning("Some evaluation components not available")

logger = logging.getLogger(__name__)


@dataclass
class ComprehensiveRAGResult:
    """Container for comprehensive RAG evaluation results"""
    # RAGAS metrics
    ragas_result: Optional[RAGASEvaluationResult] = None

    # Custom multi-modal metrics
    multimodal_result: Optional[MultiModalMetricsResult] = None

    # Overall assessment
    overall_rag_quality: float = 0.0
    evaluation_timestamp: str = None
    total_evaluation_time: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.evaluation_timestamp is None:
            self.evaluation_timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "ragas_metrics": self.ragas_result.to_dict() if self.ragas_result else None,
            "multimodal_metrics": self.multimodal_result.to_dict() if self.multimodal_result else None,
            "overall_rag_quality": self.overall_rag_quality,
            "evaluation_timestamp": self.evaluation_timestamp,
            "total_evaluation_time": self.total_evaluation_time,
            "metadata": self.metadata
        }

    def get_summary(self) -> str:
        """Get human-readable summary"""
        summary = []
        summary.append("=" * 60)
        summary.append("COMPREHENSIVE RAG EVALUATION RESULTS")
        summary.append("=" * 60)
        summary.append("")

        # RAGAS results
        if self.ragas_result:
            summary.append("RAGAS Framework Metrics:")
            summary.append(f"  Faithfulness: {self.ragas_result.faithfulness:.3f}")
            summary.append(f"  Answer Relevance: {self.ragas_result.answer_relevancy:.3f}")
            summary.append(f"  Context Relevance: {self.ragas_result.context_relevancy:.3f}")
            summary.append(f"  Context Precision: {self.ragas_result.context_precision:.3f}")
            if self.ragas_result.context_recall:
                summary.append(f"  Context Recall: {self.ragas_result.context_recall:.3f}")
            summary.append("")

        # Multi-modal results
        if self.multimodal_result:
            summary.append("Multi-Modal RAG Metrics:")
            summary.append(f"  Cross-Modal Consistency: {self.multimodal_result.cross_modal_consistency:.3f}")
            summary.append(f"  Temporal Coherence Quality: {self.multimodal_result.temporal_coherence_quality:.3f}")
            summary.append(f"  Multi-Modal Context Utilization: {self.multimodal_result.multimodal_context_utilization:.3f}")
            summary.append(f"  Source Diversity: {self.multimodal_result.source_diversity:.3f}")
            summary.append(f"  Citation Accuracy: {self.multimodal_result.citation_accuracy:.3f}")
            summary.append("")

        # Overall assessment
        summary.append(f"Overall RAG Quality Score: {self.overall_rag_quality:.3f}")
        summary.append(f"Evaluation Time: {self.total_evaluation_time:.2f}s")
        summary.append(f"Timestamp: {self.evaluation_timestamp}")
        summary.append("=" * 60)

        return "\n".join(summary)


class ComprehensiveRAGEvaluator:
    """
    Comprehensive RAG evaluator combining RAGAS and custom multi-modal metrics
    """

    def __init__(self, enable_ragas: bool = True, enable_multimodal: bool = True):
        """
        Initialize comprehensive RAG evaluator

        Args:
            enable_ragas: Enable RAGAS framework evaluation
            enable_multimodal: Enable custom multi-modal metrics
        """
        self.enable_ragas = enable_ragas and RAGASIntegration is not None
        self.enable_multimodal = enable_multimodal and MultiModalRAGEvaluator is not None

        # Initialize evaluators
        if self.enable_ragas:
            try:
                self.ragas_evaluator = RAGASIntegration()
                logger.info("RAGAS evaluator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize RAGAS: {e}")
                self.enable_ragas = False
                self.ragas_evaluator = None

        if self.enable_multimodal:
            try:
                self.multimodal_evaluator = MultiModalRAGEvaluator()
                logger.info("Multi-modal evaluator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize multi-modal evaluator: {e}")
                self.enable_multimodal = False
                self.multimodal_evaluator = None

        if not self.enable_ragas and not self.enable_multimodal:
            raise ValueError("At least one evaluator must be enabled")

        logger.info("Comprehensive RAG Evaluator initialized")

    def evaluate_single_query(
        self,
        query: str,
        retrieved_contexts: List[str],
        generated_answer: str,
        sources: List[Dict[str, Any]],
        ground_truth_answer: Optional[str] = None
    ) -> ComprehensiveRAGResult:
        """
        Evaluate a single query with comprehensive RAG metrics

        Args:
            query: User query
            retrieved_contexts: Retrieved contexts for the query
            generated_answer: Generated answer from RAG system
            sources: Source dictionaries with metadata
            ground_truth_answer: Optional ground truth answer

        Returns:
            ComprehensiveRAGResult with all evaluation metrics
        """
        start_time = time.time()

        ragas_result = None
        multimodal_result = None

        # RAGAS evaluation
        if self.enable_ragas:
            try:
                ragas_result = self.ragas_evaluator.evaluate_single_query(
                    query=query,
                    retrieved_context=retrieved_contexts,
                    generated_answer=generated_answer,
                    ground_truth_answer=ground_truth_answer
                )
            except Exception as e:
                logger.error(f"RAGAS evaluation failed: {e}")

        # Multi-modal evaluation
        if self.enable_multimodal:
            try:
                multimodal_result = self.multimodal_evaluator.evaluate_all_metrics(
                    answer=generated_answer,
                    sources=sources,
                    query=query
                )
            except Exception as e:
                logger.error(f"Multi-modal evaluation failed: {e}")

        # Calculate overall quality score
        overall_quality = self._calculate_overall_quality(ragas_result, multimodal_result)

        evaluation_time = time.time() - start_time

        return ComprehensiveRAGResult(
            ragas_result=ragas_result,
            multimodal_result=multimodal_result,
            overall_rag_quality=overall_quality,
            total_evaluation_time=evaluation_time,
            metadata={
                "query": query,
                "answer_length": len(generated_answer),
                "num_contexts": len(retrieved_contexts),
                "num_sources": len(sources),
                "has_ground_truth": ground_truth_answer is not None
            }
        )

    def evaluate_batch(
        self,
        evaluation_data: List[Dict[str, Any]],
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of queries with comprehensive RAG metrics

        Args:
            evaluation_data: List of dictionaries containing query, contexts, answer, sources, and optional ground_truth
            save_path: Optional path to save results

        Returns:
            Dictionary with aggregated evaluation results
        """
        start_time = time.time()
        results = []

        total_queries = len(evaluation_data)
        logger.info(f"Starting comprehensive RAG evaluation for {total_queries} queries")

        for i, item in enumerate(evaluation_data):
            try:
                result = self.evaluate_single_query(
                    query=item["query"],
                    retrieved_contexts=item.get("contexts", []),
                    generated_answer=item["answer"],
                    sources=item.get("sources", []),
                    ground_truth_answer=item.get("ground_truth")
                )
                results.append(result)

                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{total_queries} queries evaluated")

            except Exception as e:
                logger.error(f"Failed to evaluate query {i + 1}: {e}")

        total_time = time.time() - start_time

        # Calculate aggregated metrics
        aggregated = self._aggregate_results(results)

        # Add metadata
        aggregated["total_evaluation_time"] = total_time
        aggregated["num_queries_evaluated"] = len(results)
        aggregated["success_rate"] = len(results) / total_queries if total_queries > 0 else 0

        # Save results if path provided
        if save_path:
            self._save_results(aggregated, results, save_path)
            logger.info(f"Results saved to {save_path}")

        return aggregated

    def _calculate_overall_quality(
        self,
        ragas_result: Optional[RAGASEvaluationResult],
        multimodal_result: Optional[MultiModalMetricsResult]
    ) -> float:
        """
        Calculate overall RAG quality score combining both metric types

        Args:
            ragas_result: RAGAS evaluation result
            multimodal_result: Multi-modal metrics result

        Returns:
            Overall quality score (0-1)
        """
        scores = []

        if ragas_result:
            scores.append(ragas_result.get_average_score())

        if multimodal_result:
            scores.append(multimodal_result.get_average_score())

        if not scores:
            return 0.0

        # Weight RAGAS slightly higher (0.6) than custom metrics (0.4)
        if len(scores) == 2:
            return (scores[0] * 0.6) + (scores[1] * 0.4)
        else:
            return scores[0]

    def _aggregate_results(self, results: List[ComprehensiveRAGResult]) -> Dict[str, Any]:
        """
        Aggregate results from multiple evaluations

        Args:
            results: List of ComprehensiveRAGResult objects

        Returns:
            Dictionary with aggregated metrics
        """
        if not results:
            return {
                "overall_average_quality": 0.0,
                "num_results": 0
            }

        # RAGAS aggregates
        ragas_aggregates = {}
        if any(r.ragas_result for r in results):
            ragas_results = [r.ragas_result for r in results if r.ragas_result]
            ragas_aggregates = {
                "ragas_average_faithfulness": sum(r.faithfulness for r in ragas_results) / len(ragas_results),
                "ragas_average_answer_relevancy": sum(r.answer_relevancy for r in ragas_results) / len(ragas_results),
                "ragas_average_context_relevancy": sum(r.context_relevancy for r in ragas_results) / len(ragas_results),
                "ragas_average_context_precision": sum(r.context_precision for r in ragas_results) / len(ragas_results),
            }

        # Multi-modal aggregates
        multimodal_aggregates = {}
        if any(r.multimodal_result for r in results):
            multimodal_results = [r.multimodal_result for r in results if r.multimodal_result]
            multimodal_aggregates = {
                "multimodal_average_cross_modal_consistency": sum(r.cross_modal_consistency for r in multimodal_results) / len(multimodal_results),
                "multimodal_average_temporal_coherence": sum(r.temporal_coherence_quality for r in multimodal_results) / len(multimodal_results),
                "multimodal_average_context_utilization": sum(r.multimodal_context_utilization for r in multimodal_results) / len(multimodal_results),
                "multimodal_average_source_diversity": sum(r.source_diversity for r in multimodal_results) / len(multimodal_results),
                "multimodal_average_citation_accuracy": sum(r.citation_accuracy for r in multimodal_results) / len(multimodal_results),
            }

        # Overall quality
        overall_quality = sum(r.overall_rag_quality for r in results) / len(results)

        return {
            "overall_average_quality": overall_quality,
            "ragas_metrics": ragas_aggregates,
            "multimodal_metrics": multimodal_aggregates,
            "num_results": len(results),
            "timestamp": datetime.now().isoformat()
        }

    def _save_results(
        self,
        aggregated: Dict[str, Any],
        results: List[ComprehensiveRAGResult],
        save_path: Path
    ) -> None:
        """
        Save evaluation results to file

        Args:
            aggregated: Aggregated metrics dictionary
            results: List of individual results
            save_path: Path to save results
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "aggregated_metrics": aggregated,
            "individual_results": [r.to_dict() for r in results],
            "summary": results[0].get_summary() if results else ""
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)


def create_comprehensive_rag_evaluator(
    enable_ragas: bool = True,
    enable_multimodal: bool = True
) -> ComprehensiveRAGEvaluator:
    """
    Factory function to create comprehensive RAG evaluator

    Args:
        enable_ragas: Enable RAGAS framework evaluation
        enable_multimodal: Enable custom multi-modal metrics

    Returns:
        ComprehensiveRAGEvaluator instance
    """
    return ComprehensiveRAGEvaluator(enable_ragas, enable_multimodal)


if __name__ == "__main__":
    # Test comprehensive evaluator
    logging.basicConfig(level=logging.INFO)

    try:
        evaluator = create_comprehensive_rag_evaluator()

        # Test data
        test_query = "What is machine learning?"
        test_contexts = [
            "Machine learning is a subset of artificial intelligence...",
            "ML algorithms use statistical techniques to learn from data."
        ]
        test_answer = "Machine learning is a branch of AI that enables systems to learn from data."
        test_sources = [
            {"source_type": "video", "source_name": "Stanford ML", "metadata": {"video_id": "abc123"}},
            {"source_type": "pdf", "source_name": "ML Textbook", "metadata": {"chapter": "1"}}
        ]

        # Evaluate
        result = evaluator.evaluate_single_query(
            query=test_query,
            retrieved_contexts=test_contexts,
            generated_answer=test_answer,
            sources=test_sources
        )

        print(result.get_summary())

    except Exception as e:
        print(f"Error testing comprehensive evaluator: {e}")
        print("Some evaluators may not be available due to missing dependencies")