"""
Retrieval Quality Evaluator for Multi-Modal RAG System

Evaluates the effectiveness of multi-modal retrieval across PDFs,
videos, and web content using standard information retrieval metrics.
"""

from typing import Dict, List, Any, Set
import numpy as np

from src.evaluation.base_evaluator import BaseEvaluator, BenchmarkValidationError
from src.evaluation.metrics.retrieval_metrics import RetrievalMetrics


class RetrievalQualityEvaluator(BaseEvaluator):
    """
    Evaluator for multi-modal retrieval quality.

    Validates the system's ability to retrieve relevant content across
    different modalities using standard IR metrics like Precision@K,
    Recall@K, MAP, NDCG, and MRR.
    """

    def __init__(self, pipeline: Any, config: Dict = None):
        """
        Initialize retrieval quality evaluator.

        Args:
            pipeline: UnifiedMultiModalRAGPipeline instance
            config: Optional configuration dictionary
        """
        super().__init__(pipeline, config)
        self.target_relevance = 0.90  # Claim: 90%+ top-5 relevance
        self.k_values = config.get('k_values', [1, 3, 5, 10])

    def get_required_benchmark_fields(self) -> List[str]:
        """
        Return required fields for retrieval quality benchmark.

        Returns:
            List of required field names
        """
        return ['queries', 'relevant_documents']

    def evaluate(self, benchmark_data: Dict) -> Dict:
        """
        Run retrieval quality evaluation.

        Args:
            benchmark_data: Dictionary containing:
                - queries: List of test queries
                - relevant_documents: List of relevant document IDs for each query
                - relevance_grades: Optional dict of relevance grades (0-3) for graded relevance
                - top_k: Number of documents to retrieve per query (default: 10)

        Returns:
            Dictionary containing evaluation metrics
        """
        self.setup_evaluation(benchmark_data)

        queries = benchmark_data['queries']
        relevant_docs = benchmark_data['relevant_documents']
        top_k = benchmark_data.get('top_k', 10)
        relevance_grades = benchmark_data.get('relevance_grades', None)

        if len(queries) != len(relevant_docs):
            raise BenchmarkValidationError(
                f"Number of queries ({len(queries)}) must match "
                f"number of relevant document lists ({len(relevant_docs)})"
            )

        self.logger.info(f"Evaluating retrieval quality for {len(queries)} queries (top_k={top_k})")

        query_results = []

        with Timer('retrieval_quality_evaluation') as timer:
            for i, query in enumerate(queries):
                try:
                    # Perform retrieval for this query
                    result = self.pipeline.query(query, top_k=top_k)

                    # Extract retrieved document IDs from sources
                    retrieved_ids = self._extract_retrieved_ids(result)

                    # Prepare relevance information
                    relevant_set = set(relevant_docs[i]) if relevant_docs[i] else set()

                    query_result = {
                        'retrieved': retrieved_ids,
                        'relevant': relevant_set
                    }

                    # Add relevance grades if available
                    if relevance_grades and i in relevance_grades:
                        query_result['relevance'] = relevance_grades[i]

                    query_results.append(query_result)

                except Exception as e:
                    self.logger.error(f"Error retrieving for query '{query}': {e}")
                    # Add empty result
                    query_results.append({
                        'retrieved': [],
                        'relevant': set(relevant_docs[i]) if relevant_docs[i] else set()
                    })

        # Calculate comprehensive retrieval metrics
        all_metrics = RetrievalMetrics.calculate_all_metrics(query_results, self.k_values)

        # Validate against claimed relevance (Precision@5)
        precision_at_5 = all_metrics['precision_at_k'].get('P@5', 0.0)
        validation_result = self.validate_claim(
            metric_value=precision_at_5,
            claim_value=self.target_relevance,
            threshold=self.config.get('relevance_threshold', 0.05)
        )

        # Calculate per-modality retrieval statistics
        modality_stats = self._calculate_modality_stats(query_results, queries)

        # Compile results
        self.results = {
            'num_queries': len(queries),
            'top_k': top_k,
            'retrieval_metrics': all_metrics,
            'validation': validation_result,
            'modality_statistics': modality_stats,
            'evaluation_time_seconds': timer.elapsed_time
        }

        self.logger.info(f"Retrieval evaluation complete: P@5={precision_at_5:.2%}")

        return self.results

    def _extract_retrieved_ids(self, query_result: Dict) -> List[str]:
        """
        Extract retrieved document IDs from query result.

        Args:
            query_result: Result dictionary from pipeline.query()

        Returns:
            List of document IDs in retrieval order
        """
        retrieved_ids = []

        # Extract from sources
        sources = query_result.get('sources', [])
        for source in sources:
            source_id = source.get('name', source.get('id', ''))
            source_type = source.get('type', 'unknown')

            # Create unique ID combining type and name
            unique_id = f"{source_type}_{source_id}"
            retrieved_ids.append(unique_id)

        return retrieved_ids

    def _calculate_modality_stats(self, query_results: List[Dict], queries: List[str]) -> Dict:
        """
        Calculate retrieval statistics per modality.

        Args:
            query_results: List of query retrieval results
            queries: List of queries

        Returns:
            Dictionary containing per-modality statistics
        """
        modality_counts = {'video': 0, 'pdf': 0, 'aman': 0, 'unknown': 0}
        modality_relevance = {'video': [], 'pdf': [], 'aman': []}

        for query_result in query_results:
            retrieved = query_result.get('retrieved', [])
            relevant = query_result.get('relevant', set())

            for doc_id in retrieved:
                # Extract modality from document ID
                modality = 'unknown'
                if doc_id.startswith('video_'):
                    modality = 'video'
                elif doc_id.startswith('pdf_'):
                    modality = 'pdf'
                elif doc_id.startswith('aman_'):
                    modality = 'aman'

                modality_counts[modality] += 1

                # Track relevance per modality
                if modality in modality_relevance:
                    is_relevant = 1 if doc_id in relevant else 0
                    modality_relevance[modality].append(is_relevant)

        # Calculate average relevance per modality
        modality_performance = {}
        for modality, relevance_list in modality_relevance.items():
            if relevance_list:
                modality_performance[modality] = {
                    'count': len(relevance_list),
                    'relevant_count': sum(relevance_list),
                    'precision': sum(relevance_list) / len(relevance_list)
                }
            else:
                modality_performance[modality] = {
                    'count': 0,
                    'relevant_count': 0,
                    'precision': 0.0
                }

        return {
            'modality_counts': modality_counts,
            'modality_performance': modality_performance
        }