"""
RAGAS Framework Integration for Multi-Modal RAG System
Integrates RAGAS (Retrieval Augmented Generation Assessment) for comprehensive RAG evaluation
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import time

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_relevancy,
        context_precision,
        context_recall
    )
    from ragas.dataset import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    # Create dummy classes for when RAGAS is not available
    Dataset = None
    logging.warning("RAGAS not installed. Install with: pip install ragas")

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RAG_EVALUATION_CONFIG, LLM_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class RAGASEvaluationResult:
    """Container for RAGAS evaluation results"""
    faithfulness: float
    answer_relevancy: float
    context_relevancy: float
    context_precision: float
    context_recall: Optional[float] = None

    # Additional metadata
    evaluation_time: float = 0.0
    num_samples: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_relevancy": self.context_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "evaluation_time": self.evaluation_time,
            "num_samples": self.num_samples,
            "errors": self.errors
        }

    def get_average_score(self) -> float:
        """Calculate average RAGAS score"""
        scores = [
            self.faithfulness,
            self.answer_relevancy,
            self.context_relevancy,
            self.context_precision
        ]
        if self.context_recall is not None:
            scores.append(self.context_recall)

        return sum(scores) / len(scores)


class RAGASIntegration:
    """
    Integration layer for RAGAS evaluation framework with multi-modal RAG system
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize RAGAS integration

        Args:
            config: Optional configuration dictionary (uses RAG_EVALUATION_CONFIG if not provided)
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS is not installed. Install with: pip install ragas")

        self.config = config or RAG_EVALUATION_CONFIG
        self.enabled = self.config.get("enable_ragas", True)

        # Initialize metrics based on configuration
        self.metrics = self._initialize_metrics()

        logger.info("RAGAS Integration initialized successfully")

    def _initialize_metrics(self) -> List:
        """Initialize RAGAS metrics based on configuration"""
        enabled_metrics = self.config.get("ragas_metrics", [])

        # Available RAGAS metrics mapping
        available_metrics = {
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevancy,
            "answer_relevancy": answer_relevancy,  # Alternative name
            "context_relevance": context_relevancy,
            "context_relevancy": context_relevancy,  # Alternative name
            "context_precision": context_precision,
            "context_recall": context_recall
        }

        metrics = []
        for metric_name in enabled_metrics:
            if metric_name in available_metrics:
                metrics.append(available_metrics[metric_name])
                logger.info(f"Enabled RAGAS metric: {metric_name}")
            else:
                logger.warning(f"Unknown RAGAS metric: {metric_name}")

        return metrics

    def prepare_ragas_dataset(
        self,
        queries: List[str],
        retrieved_contexts: List[List[str]],
        generated_answers: List[str],
        ground_truth_answers: Optional[List[str]] = None
    ) -> Dataset:
        """
        Prepare RAGAS dataset from multi-modal RAG results

        Args:
            queries: List of user queries
            retrieved_contexts: List of retrieved contexts for each query (each context is a list of strings)
            generated_answers: List of generated answers from RAG system
            ground_truth_answers: Optional list of ground truth answers

        Returns:
            RAGAS Dataset object
        """
        # Validate input lengths
        if not (len(queries) == len(retrieved_contexts) == len(generated_answers)):
            raise ValueError("Queries, contexts, and answers must have the same length")

        # Prepare RAGAS dataset format
        dataset_dict = {
            "question": queries,
            "contexts": retrieved_contexts,
            "answer": generated_answers
        }

        # Add ground truth if available (for context_recall)
        if ground_truth_answers and len(ground_truth_answers) == len(queries):
            dataset_dict["ground_truth"] = ground_truth_answers

        try:
            dataset = Dataset.from_dict(dataset_dict)
            logger.info(f"Created RAGAS dataset with {len(queries)} samples")
            return dataset
        except Exception as e:
            logger.error(f"Error creating RAGAS dataset: {e}")
            raise

    def evaluate_rag_results(
        self,
        queries: List[str],
        retrieved_contexts: List[List[str]],
        generated_answers: List[str],
        ground_truth_answers: Optional[List[str]] = None
    ) -> RAGASEvaluationResult:
        """
        Evaluate RAG results using RAGAS metrics

        Args:
            queries: List of user queries
            retrieved_contexts: List of retrieved contexts for each query
            generated_answers: List of generated answers from RAG system
            ground_truth_answers: Optional list of ground truth answers

        Returns:
            RAGASEvaluationResult with all metrics
        """
        if not self.enabled:
            logger.warning("RAGAS evaluation is disabled")
            return RAGASEvaluationResult(
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_relevancy=0.0,
                context_precision=0.0
            )

        try:
            # Prepare dataset
            dataset = self.prepare_ragas_dataset(
                queries=queries,
                retrieved_contexts=retrieved_contexts,
                generated_answers=generated_answers,
                ground_truth_answers=ground_truth_answers
            )

            # Run RAGAS evaluation
            start_time = time.time()
            result = evaluate(dataset=dataset, metrics=self.metrics)
            evaluation_time = time.time() - start_time

            # Extract scores from result
            scores_dict = result.to_pandas().to_dict('records')

            # Calculate averages
            faithfulness_scores = [s.get('faithfulness', 0) for s in scores_dict if not pd.isna(s.get('faithfulness', 0))]
            answer_relevancy_scores = [s.get('answer_relevancy', 0) for s in scores_dict if not pd.isna(s.get('answer_relevancy', 0))]
            context_relevancy_scores = [s.get('context_relevancy', 0) for s in scores_dict if not pd.isna(s.get('context_relevancy', 0))]
            context_precision_scores = [s.get('context_precision', 0) for s in scores_dict if not pd.isna(s.get('context_precision', 0))]

            avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
            avg_answer_relevancy = sum(answer_relevancy_scores) / len(answer_relevancy_scores) if answer_relevancy_scores else 0.0
            avg_context_relevancy = sum(context_relevancy_scores) / len(context_relevancy_scores) if context_relevancy_scores else 0.0
            avg_context_precision = sum(context_precision_scores) / len(context_precision_scores) if context_precision_scores else 0.0

            # Handle context_recall (may not be available without ground truth)
            context_recall_scores = [s.get('context_recall', 0) for s in scores_dict if 'context_recall' in s and not pd.isna(s.get('context_recall', 0))]
            avg_context_recall = sum(context_recall_scores) / len(context_recall_scores) if context_recall_scores else None

            evaluation_result = RAGASEvaluationResult(
                faithfulness=avg_faithfulness,
                answer_relevancy=avg_answer_relevancy,
                context_relevancy=avg_context_relevancy,
                context_precision=avg_context_precision,
                context_recall=avg_context_recall,
                evaluation_time=evaluation_time,
                num_samples=len(queries)
            )

            logger.info(f"RAGAS evaluation completed in {evaluation_time:.2f}s")
            logger.info(f"Average RAGAS Score: {evaluation_result.get_average_score():.3f}")

            return evaluation_result

        except Exception as e:
            logger.error(f"Error during RAGAS evaluation: {e}")
            return RAGASEvaluationResult(
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_relevancy=0.0,
                context_precision=0.0,
                errors=[str(e)]
            )

    def evaluate_single_query(
        self,
        query: str,
        retrieved_context: List[str],
        generated_answer: str,
        ground_truth_answer: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single query using RAGAS metrics

        Args:
            query: User query
            retrieved_context: Retrieved context for the query
            generated_answer: Generated answer from RAG system
            ground_truth_answer: Optional ground truth answer

        Returns:
            Dictionary with metric scores
        """
        # Wrap single query in lists
        queries = [query]
        retrieved_contexts = [retrieved_context]
        generated_answers = [generated_answer]
        ground_truth_answers = [ground_truth_answer] if ground_truth_answer else None

        # Evaluate
        result = self.evaluate_rag_results(
            queries=queries,
            retrieved_contexts=retrieved_contexts,
            generated_answers=generated_answers,
            ground_truth_answers=ground_truth_answers
        )

        return result.to_dict()

    def batch_evaluate(
        self,
        evaluation_data: List[Dict[str, Any]],
        batch_size: Optional[int] = None
    ) -> List[RAGASEvaluationResult]:
        """
        Batch evaluate multiple queries

        Args:
            evaluation_data: List of dictionaries containing query, contexts, answer, and optional ground_truth
            batch_size: Optional batch size (uses config default if not specified)

        Returns:
            List of RAGASEvaluationResult objects
        """
        batch_size = batch_size or self.config.get("batch_size", 10)

        results = []
        for i in range(0, len(evaluation_data), batch_size):
            batch = evaluation_data[i:i + batch_size]

            queries = [item["query"] for item in batch]
            contexts = [item["contexts"] for item in batch]
            answers = [item["answer"] for item in batch]
            ground_truths = [item.get("ground_truth") for item in batch]

            # Remove None values from ground_truths
            if all(gt is None for gt in ground_truths):
                ground_truths = None

            result = self.evaluate_rag_results(
                queries=queries,
                retrieved_contexts=contexts,
                generated_answers=answers,
                ground_truth_answers=ground_truths
            )

            results.append(result)
            logger.info(f"Batch {i//batch_size + 1} completed")

        return results


# Import pandas for handling NaN values
try:
    import pandas as pd
except ImportError:
    pd = None
    logger.warning("Pandas not available. Some RAGAS features may not work properly.")


def create_ragas_integration(config: Optional[Dict[str, Any]] = None) -> RAGASIntegration:
    """
    Factory function to create RAGAS integration

    Args:
        config: Optional configuration dictionary

    Returns:
        RAGASIntegration instance
    """
    return RAGASIntegration(config=config)


if __name__ == "__main__":
    # Test RAGAS integration
    logging.basicConfig(level=logging.INFO)

    try:
        ragas_integration = create_ragas_integration()

        # Sample test data
        test_queries = [
            "What is machine learning?",
            "How do neural networks work?"
        ]

        test_contexts = [
            ["Machine learning is a subset of artificial intelligence...",
             "ML algorithms use statistical techniques to learn from data."],
            ["Neural networks are computing systems inspired by biological neural networks...",
             "Deep neural networks contain multiple hidden layers."]
        ]

        test_answers = [
            "Machine learning is a branch of AI that enables systems to learn from data.",
            "Neural networks work by processing information through layers of interconnected nodes."
        ]

        # Run evaluation
        result = ragas_integration.evaluate_rag_results(
            queries=test_queries,
            retrieved_contexts=test_contexts,
            generated_answers=test_answers
        )

        print("RAGAS Evaluation Results:")
        print(f"Faithfulness: {result.faithfulness:.3f}")
        print(f"Answer Relevancy: {result.answer_relevancy:.3f}")
        print(f"Context Relevancy: {result.context_relevancy:.3f}")
        print(f"Context Precision: {result.context_precision:.3f}")
        print(f"Average Score: {result.get_average_score():.3f}")

    except Exception as e:
        print(f"Error testing RAGAS integration: {e}")
        print("Make sure RAGAS is properly installed and configured")