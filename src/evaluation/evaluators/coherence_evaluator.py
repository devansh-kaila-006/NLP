"""
Temporal Coherence Evaluator for Multi-Modal RAG System

Evaluates the temporal coherence and logical progression of video chunks
to ensure retrieved content maintains proper temporal flow.
"""

from typing import Dict, List, Any
import numpy as np

from src.evaluation.base_evaluator import BaseEvaluator, BenchmarkValidationError
from src.evaluation.metrics.coherence_metrics import CoherenceMetrics


class TemporalCoherenceEvaluator(BaseEvaluator):
    """
    Evaluator for temporal coherence in video RAG systems.

    Validates the system's ability to maintain temporal consistency
    and logical progression across retrieved video chunks.
    """

    def __init__(self, pipeline: Any, config: Dict = None):
        """
        Initialize temporal coherence evaluator.

        Args:
            pipeline: UnifiedMultiModalRAGPipeline instance
            config: Optional configuration dictionary
        """
        super().__init__(pipeline, config)
        self.target_coherence = 0.95  # Claim: 95%+ temporal coherence precision

    def get_required_benchmark_fields(self) -> List[str]:
        """
        Return required fields for temporal coherence benchmark.

        Returns:
            List of required field names
        """
        return ['sequential_queries', 'expected_orders']

    def evaluate(self, benchmark_data: Dict) -> Dict:
        """
        Run temporal coherence evaluation.

        Args:
            benchmark_data: Dictionary containing:
                - sequential_queries: List of sequential query sets
                - expected_orders: List of expected chunk orders for each query set
                - query_types: Optional list of query type categories

        Returns:
            Dictionary containing evaluation metrics
        """
        self.setup_evaluation(benchmark_data)

        sequential_queries = benchmark_data['sequential_queries']
        expected_orders = benchmark_data['expected_orders']

        if len(sequential_queries) != len(expected_orders):
            raise BenchmarkValidationError(
                f"Number of query sets ({len(sequential_queries)}) must match "
                f"number of expected orders ({len(expected_orders)})"
            )

        self.logger.info(f"Evaluating temporal coherence for {len(sequential_queries)} query sets")

        retrieved_sequences = []
        expected_sequences = []

        with Timer('temporal_coherence_evaluation') as timer:
            for i, (query_set, expected_order) in enumerate(zip(sequential_queries, expected_orders)):
                try:
                    # Process each query in the sequential set
                    sequence_retrieved = []
                    sequence_expected = []

                    for j, query in enumerate(query_set):
                        # Retrieve chunks for this query
                        result = self.pipeline.query(query, top_k=5)

                        # Extract video chunks with temporal information
                        video_chunks = self._extract_video_chunks(result)

                        # Add chunk IDs for expected order
                        if j < len(expected_order):
                            expected_chunks = self._create_expected_chunks(expected_order[j], video_chunks)
                            sequence_expected.extend(expected_chunks)

                        sequence_retrieved.extend(video_chunks)

                    if sequence_retrieved and sequence_expected:
                        retrieved_sequences.append(sequence_retrieved)
                        expected_sequences.append(sequence_expected)

                except Exception as e:
                    self.logger.error(f"Error processing query set {i}: {e}")

        # Calculate comprehensive coherence metrics
        coherence_results = CoherenceMetrics.calculate_all_metrics(
            retrieved_sequences, expected_sequences
        )

        # Validate against claimed coherence precision
        coherence_precision = coherence_results['summary']['coherence_precision']['mean']
        validation_result = self.validate_claim(
            metric_value=coherence_precision,
            claim_value=self.target_coherence,
            threshold=self.config.get('coherence_threshold', 0.03)
        )

        # Compile results
        self.results = {
            'num_query_sets': len(sequential_queries),
            'num_successful_sets': len(retrieved_sequences),
            'coherence_metrics': coherence_results,
            'validation': validation_result,
            'evaluation_time_seconds': timer.elapsed_time
        }

        self.logger.info(f"Temporal coherence evaluation complete: {coherence_precision:.2%} precision")

        return self.results

    def _extract_video_chunks(self, query_result: Dict) -> List[Dict]:
        """
        Extract video chunks with temporal information from query result.

        Args:
            query_result: Result dictionary from pipeline.query()

        Returns:
            List of video chunks with temporal metadata
        """
        video_chunks = []

        sources = query_result.get('sources', [])
        for source in sources:
            if source.get('type') == 'video':
                chunk = {
                    'chunk_id': source.get('name', ''),
                    'timestamp_start': source.get('timestamp_start', 0),
                    'timestamp_end': source.get('timestamp_end', 0),
                    'text': source.get('text', ''),
                    'video_url': source.get('video_url', ''),
                    'playlist': source.get('playlist', '')
                }
                video_chunks.append(chunk)

        return video_chunks

    def _create_expected_chunks(self, expected_order: List[str],
                               retrieved_chunks: List[Dict]) -> List[Dict]:
        """
        Create expected chunk sequence based on expected order.

        Args:
            expected_order: List of chunk IDs in expected order
            retrieved_chunks: Actually retrieved chunks for reference

        Returns:
            List of expected chunks with temporal information
        """
        expected_chunks = []

        # Create mapping of chunk IDs to chunk data
        chunk_map = {chunk['chunk_id']: chunk for chunk in retrieved_chunks}

        for chunk_id in expected_order:
            if chunk_id in chunk_map:
                expected_chunks.append(chunk_map[chunk_id])
            else:
                # Create placeholder chunk with expected ID
                expected_chunks.append({
                    'chunk_id': chunk_id,
                    'timestamp_start': 0,
                    'timestamp_end': 0,
                    'text': '',
                    'video_url': '',
                    'playlist': ''
                })

        return expected_chunks