"""
Performance Evaluator for Multi-Modal RAG System

Evaluates system performance including query latency, throughput,
and resource utilization to ensure production-ready performance.
"""

from typing import Dict, List, Any
import numpy as np
import time
import psutil
import os

from src.evaluation.base_evaluator import BaseEvaluator, BenchmarkValidationError
from src.utils.helpers import Timer


class PerformanceEvaluator(BaseEvaluator):
    """
    Evaluator for system performance profiling.

    Validates query latency, throughput, and resource usage to ensure
    the system meets production performance requirements.
    """

    def __init__(self, pipeline: Any, config: Dict = None):
        """
        Initialize performance evaluator.

        Args:
            pipeline: UnifiedMultiModalRAGPipeline instance
            config: Optional configuration dictionary
        """
        super().__init__(pipeline, config)
        self.target_query_time = 4.0  # Target: 4 seconds average query time

    def get_required_benchmark_fields(self) -> List[str]:
        """
        Return required fields for performance benchmark.

        Returns:
            List of required field names
        """
        return ['queries']

    def evaluate(self, benchmark_data: Dict) -> Dict:
        """
        Run performance evaluation.

        Args:
            benchmark_data: Dictionary containing:
                - queries: List of test queries
                - warmup_queries: Optional list of warmup queries (default: 3)
                - iterations: Number of times to run each query (default: 1)

        Returns:
            Dictionary containing performance metrics
        """
        self.setup_evaluation(benchmark_data)

        queries = benchmark_data['queries']
        warmup_queries = benchmark_data.get('warmup_queries', queries[:3])
        iterations = benchmark_data.get('iterations', 1)

        self.logger.info(f"Evaluating performance for {len(queries)} queries ({iterations} iterations each)")

        # Warmup phase
        self.logger.info("Running warmup queries...")
        for query in warmup_queries:
            try:
                self.pipeline.query(query)
            except Exception as e:
                self.logger.warning(f"Warmup query failed: {e}")

        # Performance measurement phase
        query_times = []
        component_times = []
        memory_usage = []
        cpu_usage = []

        process = psutil.Process(os.getpid())

        with Timer('performance_evaluation') as timer:
            for iteration in range(iterations):
                for i, query in enumerate(queries):
                    try:
                        # Measure memory and CPU before query
                        mem_before = process.memory_info().rss / 1024 / 1024  # MB
                        cpu_before = process.cpu_percent()

                        # Time the query execution
                        query_start = time.time()

                        with Timer(f'query_{i}') as query_timer:
                            result = self.pipeline.query(query)

                        query_time = time.time() - query_start
                        query_times.append(query_time)

                        # Measure memory and CPU after query
                        mem_after = process.memory_info().rss / 1024 / 1024  # MB
                        cpu_after = process.cpu_percent()

                        memory_usage.append(mem_after - mem_before)
                        cpu_usage.append((cpu_before + cpu_after) / 2)

                        # Extract component timing if available
                        if hasattr(query_timer, 'component_times'):
                            component_times.append(query_timer.component_times)

                        self.logger.debug(f"Query {i+1} completed in {query_time:.2f}s")

                    except Exception as e:
                        self.logger.error(f"Error executing query '{query}': {e}")

        # Calculate performance statistics
        performance_stats = self._calculate_performance_stats(
            query_times, memory_usage, cpu_usage
        )

        # Calculate percentiles
        percentiles = self._calculate_percentiles(query_times)

        # Validate against target query time
        mean_query_time = performance_stats['mean']
        validation_result = self.validate_claim(
            metric_value=mean_query_time,
            claim_value=self.target_query_time,
            threshold=self.config.get('performance_threshold', 0.25)  # 25% tolerance
        )

        # Compile results
        self.results = {
            'num_queries': len(queries),
            'iterations': iterations,
            'total_queries_executed': len(query_times),
            'query_time_stats': performance_stats,
            'query_time_percentiles': percentiles,
            'memory_usage_stats': self._calculate_performance_stats(memory_usage, [], []),
            'cpu_usage_stats': self._calculate_performance_stats(cpu_usage, [], []),
            'validation': validation_result,
            'evaluation_time_seconds': timer.elapsed_time
        }

        # Add component timing analysis if available
        if component_times:
            self.results['component_timing'] = self._analyze_component_timing(component_times)

        self.logger.info(f"Performance evaluation complete: {mean_query_time:.2f}s average query time")

        return self.results

    def _calculate_performance_stats(self, primary_values: List[float],
                                    secondary_values: List[float],
                                    tertiary_values: List[float]) -> Dict:
        """
        Calculate comprehensive performance statistics.

        Args:
            primary_values: Primary metric values (e.g., query times)
            secondary_values: Secondary metric values (e.g., memory usage)
            tertiary_values: Tertiary metric values (e.g., CPU usage)

        Returns:
            Dictionary containing performance statistics
        """
        if not primary_values:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'count': 0
            }

        values_array = np.array(primary_values)

        return {
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'median': float(np.median(values_array)),
            'count': len(values_array)
        }

    def _calculate_percentiles(self, values: List[float]) -> Dict:
        """
        Calculate percentile statistics for performance metrics.

        Args:
            values: List of metric values

        Returns:
            Dictionary containing percentile values
        """
        if not values:
            return {
                'p50': 0.0,
                'p75': 0.0,
                'p90': 0.0,
                'p95': 0.0,
                'p99': 0.0
            }

        values_array = np.array(values)

        return {
            'p50': float(np.percentile(values_array, 50)),
            'p75': float(np.percentile(values_array, 75)),
            'p90': float(np.percentile(values_array, 90)),
            'p95': float(np.percentile(values_array, 95)),
            'p99': float(np.percentile(values_array, 99))
        }

    def _analyze_component_timing(self, component_times: List[Dict]) -> Dict:
        """
        Analyze timing breakdown across pipeline components.

        Args:
            component_times: List of component timing dictionaries

        Returns:
            Dictionary containing component timing statistics
        """
        component_stats = {}

        # Aggregate timing by component
        for timing_dict in component_times:
            for component, time_value in timing_dict.items():
                if component not in component_stats:
                    component_stats[component] = []
                component_stats[component].append(time_value)

        # Calculate statistics for each component
        component_analysis = {}
        for component, times in component_stats.items():
            times_array = np.array(times)
            component_analysis[component] = {
                'mean': float(np.mean(times_array)),
                'std': float(np.std(times_array)),
                'min': float(np.min(times_array)),
                'max': float(np.max(times_array)),
                'total': float(np.sum(times_array))
            }

        return component_analysis