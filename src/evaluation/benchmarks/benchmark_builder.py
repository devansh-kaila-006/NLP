"""
Benchmark Builder for Multi-Modal RAG System

Creates comprehensive ground truth datasets with relevance judgments
for evaluation of the multi-modal RAG system.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Set
from datetime import datetime

from src.evaluation.benchmarks.query_generator import QueryGenerator
from src.pipeline.unified_multimodal_pipeline import UnifiedMultiModalRAGPipeline
from src.utils.logger import LoggerMixin


class BenchmarkBuilder(LoggerMixin):
    """
    Build comprehensive benchmark datasets for RAG evaluation.

    Creates ground truth datasets with queries, relevance judgments,
    and expected outcomes for systematic evaluation.
    """

    def __init__(self, pipeline: UnifiedMultiModalRAGPipeline):
        """
        Initialize benchmark builder.

        Args:
            pipeline: UnifiedMultiModalRAGPipeline instance
        """
        self.pipeline = pipeline
        self.query_generator = QueryGenerator()

    def build_modality_benchmark(self, num_queries: int = 500,
                                output_path: Path = None) -> Dict:
        """
        Build benchmark for modality prediction evaluation.

        Args:
            num_queries: Number of queries to generate
            output_path: Optional path to save benchmark

        Returns:
            Dictionary containing modality prediction benchmark
        """
        self.logger.info(f"Building modality prediction benchmark with {num_queries} queries")

        # Generate diverse queries
        queries_data = self.query_generator.generate_queries(
            domains=None,  # All domains
            query_types=None,  # All query types
            difficulty_levels=['easy', 'medium', 'hard'],
            num_queries_per_category=num_queries // 12
        )

        # Ensure we have exactly num_queries
        queries_data = queries_data[:num_queries]

        # Extract benchmark components
        queries = [q['text'] for q in queries_data]
        expected_modalities = [q['expected_modality'] for q in queries_data]
        query_types = [q['query_type'] for q in queries_data]
        domains = [q['domain'] for q in queries_data]

        # Create benchmark dictionary
        benchmark = {
            'benchmark_type': 'modality_prediction',
            'num_queries': len(queries),
            'creation_date': datetime.now().isoformat(),
            'queries': queries,
            'expected_modality': expected_modalities,
            'query_types': query_types,
            'domains': domains,
            'metadata': {
                'query_generator_version': '1.0',
                'difficulty_distribution': self._calculate_distribution(
                    [q.get('difficulty', 'medium') for q in queries_data]
                ),
                'domain_distribution': self._calculate_distribution(domains),
                'query_type_distribution': self._calculate_distribution(query_types)
            }
        }

        # Save benchmark if path provided
        if output_path:
            self._save_benchmark(benchmark, output_path)
            self.logger.info(f"Modality benchmark saved to {output_path}")

        return benchmark

    def build_retrieval_benchmark(self, num_queries: int = 200,
                                 top_k: int = 10,
                                 output_path: Path = None) -> Dict:
        """
        Build benchmark for retrieval quality evaluation.

        Args:
            num_queries: Number of queries to generate
            top_k: Number of relevant documents to identify per query
            output_path: Optional path to save benchmark

        Returns:
            Dictionary containing retrieval quality benchmark
        """
        self.logger.info(f"Building retrieval benchmark with {num_queries} queries (top_k={top_k})")

        # Generate queries
        queries_data = self.query_generator.generate_queries(
            domains=None,
            query_types=None,
            difficulty_levels=['medium'],
            num_queries_per_category=num_queries // 12
        )

        queries_data = queries_data[:num_queries]

        # Perform retrieval and collect relevance judgments
        queries = []
        relevant_documents = []
        relevance_grades = {}

        for i, query_data in enumerate(queries_data):
            query_text = query_data['text']
            queries.append(query_text)

            try:
                # Retrieve documents for this query
                result = self.pipeline.query(query_text, top_k=top_k)

                # Extract document IDs and create relevance judgments
                retrieved_docs = self._extract_document_ids(result)

                # For automatic benchmark, use retrieved docs as "relevant"
                # (in practice, this would be human-annotated)
                relevant_documents.append(retrieved_docs)

                # Create relevance grades (higher rank = higher relevance)
                doc_grades = {doc_id: 3 - rank for rank, doc_id in enumerate(retrieved_docs)}
                relevance_grades[i] = doc_grades

            except Exception as e:
                self.logger.error(f"Error retrieving for query '{query_text}': {e}")
                relevant_documents.append([])
                relevance_grades[i] = {}

        # Create benchmark dictionary
        benchmark = {
            'benchmark_type': 'retrieval_quality',
            'num_queries': len(queries),
            'top_k': top_k,
            'creation_date': datetime.now().isoformat(),
            'queries': queries,
            'relevant_documents': relevant_documents,
            'relevance_grades': relevance_grades,
            'metadata': {
                'query_generator_version': '1.0',
                'domain_distribution': self._calculate_distribution(
                    [q['domain'] for q in queries_data]
                ),
                'query_type_distribution': self._calculate_distribution(
                    [q['query_type'] for q in queries_data]
                )
            }
        }

        # Save benchmark if path provided
        if output_path:
            self._save_benchmark(benchmark, output_path)
            self.logger.info(f"Retrieval benchmark saved to {output_path}")

        return benchmark

    def build_coherence_benchmark(self, num_sequences: int = 50,
                                 sequence_length: int = 5,
                                 output_path: Path = None) -> Dict:
        """
        Build benchmark for temporal coherence evaluation.

        Args:
            num_sequences: Number of sequential query sets
            sequence_length: Number of queries per sequence
            output_path: Optional path to save benchmark

        Returns:
            Dictionary containing temporal coherence benchmark
        """
        self.logger.info(f"Building coherence benchmark with {num_sequences} sequences")

        # Generate sequential query sets
        sequential_queries = self.query_generator.generate_sequential_queries(
            num_sequences=num_sequences,
            sequence_length=sequence_length
        )

        # Extract query texts and create expected orders
        query_sets = []
        expected_orders = []

        for sequence in sequential_queries:
            query_texts = [q['text'] for q in sequence]
            query_sets.append(query_texts)

            # For expected order, use the sequence position
            # (in practice, this would be based on actual video timestamps)
            expected_order = [f"chunk_{q['sequence_position']}" for q in sequence]
            expected_orders.append(expected_order)

        # Create benchmark dictionary
        benchmark = {
            'benchmark_type': 'temporal_coherence',
            'num_query_sets': len(query_sets),
            'sequence_length': sequence_length,
            'creation_date': datetime.now().isoformat(),
            'sequential_queries': query_sets,
            'expected_orders': expected_orders,
            'metadata': {
                'query_generator_version': '1.0',
                'domain_distribution': self._calculate_distribution(
                    [seq[0]['domain'] for seq in sequential_queries if seq]
                )
            }
        }

        # Save benchmark if path provided
        if output_path:
            self._save_benchmark(benchmark, output_path)
            self.logger.info(f"Coherence benchmark saved to {output_path}")

        return benchmark

    def build_performance_benchmark(self, num_queries: int = 100,
                                   iterations: int = 3,
                                   output_path: Path = None) -> Dict:
        """
        Build benchmark for performance evaluation.

        Args:
            num_queries: Number of performance test queries
            iterations: Number of times to run each query
            output_path: Optional path to save benchmark

        Returns:
            Dictionary containing performance benchmark
        """
        self.logger.info(f"Building performance benchmark with {num_queries} queries")

        # Generate performance queries
        queries_data = self.query_generator.generate_performance_queries(num_queries)
        queries = [q['text'] for q in queries_data]

        # Create warmup queries (subset of actual queries)
        warmup_queries = queries[:min(5, len(queries))]

        # Create benchmark dictionary
        benchmark = {
            'benchmark_type': 'performance',
            'num_queries': len(queries),
            'iterations': iterations,
            'creation_date': datetime.now().isoformat(),
            'queries': queries,
            'warmup_queries': warmup_queries,
            'metadata': {
                'query_generator_version': '1.0',
                'domain_distribution': self._calculate_distribution(
                    [q['domain'] for q in queries_data]
                )
            }
        }

        # Save benchmark if path provided
        if output_path:
            self._save_benchmark(benchmark, output_path)
            self.logger.info(f"Performance benchmark saved to {output_path}")

        return benchmark

    def build_comprehensive_benchmark(self, output_dir: Path) -> Dict:
        """
        Build all benchmark datasets for comprehensive evaluation.

        Args:
            output_dir: Directory to save benchmark files

        Returns:
            Dictionary containing paths to all benchmarks
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Building comprehensive benchmark suite in {output_dir}")

        # Build all benchmarks
        modality_benchmark = self.build_modality_benchmark(
            num_queries=500,
            output_path=output_dir / 'modality_benchmark.json'
        )

        retrieval_benchmark = self.build_retrieval_benchmark(
            num_queries=200,
            top_k=10,
            output_path=output_dir / 'retrieval_benchmark.json'
        )

        coherence_benchmark = self.build_coherence_benchmark(
            num_sequences=50,
            sequence_length=5,
            output_path=output_dir / 'coherence_benchmark.json'
        )

        performance_benchmark = self.build_performance_benchmark(
            num_queries=100,
            iterations=3,
            output_path=output_dir / 'performance_benchmark.json'
        )

        return {
            'modality_benchmark': str(output_dir / 'modality_benchmark.json'),
            'retrieval_benchmark': str(output_dir / 'retrieval_benchmark.json'),
            'coherence_benchmark': str(output_dir / 'coherence_benchmark.json'),
            'performance_benchmark': str(output_dir / 'performance_benchmark.json'),
            'creation_date': datetime.now().isoformat()
        }

    def _extract_document_ids(self, query_result: Dict) -> List[str]:
        """
        Extract document IDs from query result.

        Args:
            query_result: Result dictionary from pipeline.query()

        Returns:
            List of document IDs
        """
        doc_ids = []
        sources = query_result.get('sources', [])

        for source in sources:
            source_id = source.get('name', source.get('id', ''))
            source_type = source.get('type', 'unknown')
            unique_id = f"{source_type}_{source_id}"
            doc_ids.append(unique_id)

        return doc_ids

    def _calculate_distribution(self, items: List[str]) -> Dict:
        """
        Calculate distribution of items.

        Args:
            items: List of items

        Returns:
            Dictionary with item counts and percentages
        """
        distribution = {}
        total = len(items)

        for item in items:
            distribution[item] = distribution.get(item, 0) + 1

        # Add percentages
        for item in distribution:
            distribution[item] = {
                'count': distribution[item],
                'percentage': distribution[item] / total if total > 0 else 0.0
            }

        return distribution

    def _save_benchmark(self, benchmark: Dict, output_path: Path) -> None:
        """
        Save benchmark to file.

        Args:
            benchmark: Benchmark dictionary
            output_path: Path to save benchmark
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark, f, indent=2)

        self.logger.info(f"Benchmark saved to {output_path}")