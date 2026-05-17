"""
Comprehensive Evaluation Script for Multi-Modal RAG System

Runs complete evaluation suite including modality prediction, retrieval quality,
temporal coherence, and performance testing.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.unified_multimodal_pipeline import UnifiedMultiModalRAGPipeline
from src.evaluation.evaluators import (
    ModalityPredictionEvaluator,
    RetrievalQualityEvaluator,
    TemporalCoherenceEvaluator,
    PerformanceEvaluator
)
from src.evaluation.benchmarks import BenchmarkBuilder
from src.evaluation.reporting import ReportGenerator, VisualizationGenerator
from src.utils.logger import LoggerMixin


class ComprehensiveEvaluation(LoggerMixin):
    """
    Run comprehensive evaluation of the Multi-Modal RAG system.

    Evaluates all three novel innovations and system performance
    with detailed reporting and visualization.
    """

    def __init__(self, use_reranker: bool = True, include_aman: bool = True):
        """
        Initialize comprehensive evaluation.

        Args:
            use_reranker: Whether to use reranking in pipeline
            include_aman: Whether to include Aman.ai content
        """
        self.use_reranker = use_reranker
        self.include_aman = include_aman
        self.pipeline = None
        self.benchmark_builder = None

    def setup(self):
        """Initialize pipeline and benchmark builder."""
        self.logger.info("Initializing Multi-Modal RAG Pipeline for evaluation...")

        self.pipeline = UnifiedMultiModalRAGPipeline(
            use_reranker=self.use_reranker,
            include_aman=self.include_aman
        )

        self.benchmark_builder = BenchmarkBuilder(self.pipeline)

        self.logger.info("Pipeline initialization complete")

    def run_modality_evaluation(self, num_queries: int = 100,
                               benchmark_path: Path = None) -> dict:
        """
        Run modality prediction evaluation.

        Args:
            num_queries: Number of test queries
            benchmark_path: Optional path to existing benchmark

        Returns:
            Evaluation results
        """
        self.logger.info("Running modality prediction evaluation...")

        # Create or load benchmark
        if benchmark_path and Path(benchmark_path).exists():
            with open(benchmark_path, 'r') as f:
                benchmark_data = json.load(f)
        else:
            benchmark_data = self.benchmark_builder.build_modality_benchmark(
                num_queries=num_queries,
                output_path=benchmark_path
            )

        # Run evaluation
        evaluator = ModalityPredictionEvaluator(self.pipeline)
        results = evaluator.evaluate(benchmark_data)

        self.logger.info(f"Modality prediction accuracy: {results['overall_accuracy']:.2%}")

        return results

    def run_retrieval_evaluation(self, num_queries: int = 50,
                                top_k: int = 10,
                                benchmark_path: Path = None) -> dict:
        """
        Run retrieval quality evaluation.

        Args:
            num_queries: Number of test queries
            top_k: Number of documents to retrieve
            benchmark_path: Optional path to existing benchmark

        Returns:
            Evaluation results
        """
        self.logger.info("Running retrieval quality evaluation...")

        # Create or load benchmark
        if benchmark_path and Path(benchmark_path).exists():
            with open(benchmark_path, 'r') as f:
                benchmark_data = json.load(f)
        else:
            benchmark_data = self.benchmark_builder.build_retrieval_benchmark(
                num_queries=num_queries,
                top_k=top_k,
                output_path=benchmark_path
            )

        # Run evaluation
        evaluator = RetrievalQualityEvaluator(self.pipeline)
        results = evaluator.evaluate(benchmark_data)

        precision_5 = results['retrieval_metrics']['precision_at_k']['P@5']
        self.logger.info(f"Retrieval precision@5: {precision_5:.2%}")

        return results

    def run_coherence_evaluation(self, num_sequences: int = 20,
                                sequence_length: int = 5,
                                benchmark_path: Path = None) -> dict:
        """
        Run temporal coherence evaluation.

        Args:
            num_sequences: Number of sequential query sets
            sequence_length: Number of queries per sequence
            benchmark_path: Optional path to existing benchmark

        Returns:
            Evaluation results
        """
        self.logger.info("Running temporal coherence evaluation...")

        # Create or load benchmark
        if benchmark_path and Path(benchmark_path).exists():
            with open(benchmark_path, 'r') as f:
                benchmark_data = json.load(f)
        else:
            benchmark_data = self.benchmark_builder.build_coherence_benchmark(
                num_sequences=num_sequences,
                sequence_length=sequence_length,
                output_path=benchmark_path
            )

        # Run evaluation
        evaluator = TemporalCoherenceEvaluator(self.pipeline)
        results = evaluator.evaluate(benchmark_data)

        coherence_precision = results['coherence_metrics']['summary']['coherence_precision']['mean']
        self.logger.info(f"Temporal coherence precision: {coherence_precision:.2%}")

        return results

    def run_performance_evaluation(self, num_queries: int = 30,
                                  iterations: int = 3,
                                  benchmark_path: Path = None) -> dict:
        """
        Run performance evaluation.

        Args:
            num_queries: Number of test queries
            iterations: Number of times to run each query
            benchmark_path: Optional path to existing benchmark

        Returns:
            Evaluation results
        """
        self.logger.info("Running performance evaluation...")

        # Create or load benchmark
        if benchmark_path and Path(benchmark_path).exists():
            with open(benchmark_path, 'r') as f:
                benchmark_data = json.load(f)
        else:
            benchmark_data = self.benchmark_builder.build_performance_benchmark(
                num_queries=num_queries,
                iterations=iterations,
                output_path=benchmark_path
            )

        # Run evaluation
        evaluator = PerformanceEvaluator(self.pipeline)
        results = evaluator.evaluate(benchmark_data)

        query_time = results['query_time_stats']['mean']
        self.logger.info(f"Average query time: {query_time:.2f}s")

        return results

    def run_comprehensive_evaluation(self,
                                    output_dir: Path = None,
                                    modality_queries: int = 100,
                                    retrieval_queries: int = 50,
                                    coherence_sequences: int = 20,
                                    performance_queries: int = 30,
                                    generate_benchmarks: bool = True) -> dict:
        """
        Run complete evaluation suite.

        Args:
            output_dir: Directory to save results
            modality_queries: Number of modality prediction test queries
            retrieval_queries: Number of retrieval quality test queries
            coherence_sequences: Number of temporal coherence test sequences
            performance_queries: Number of performance test queries
            generate_benchmarks: Whether to generate new benchmarks

        Returns:
            Complete evaluation results
        """
        self.logger.info("Starting comprehensive evaluation...")

        # Setup output directory
        if output_dir is None:
            output_dir = Path(f"data/evaluation/results/comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        benchmark_dir = output_dir / "benchmarks"
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        # Initialize pipeline
        self.setup()

        all_results = {}

        try:
            # Run modality prediction evaluation
            self.logger.info("=" * 60)
            modality_results = self.run_modality_evaluation(
                num_queries=modality_queries,
                benchmark_path=benchmark_dir / 'modality_benchmark.json' if generate_benchmarks else None
            )
            all_results['modality_prediction'] = modality_results

        except Exception as e:
            self.logger.error(f"Modality prediction evaluation failed: {e}")
            all_results['modality_prediction'] = {'error': str(e)}

        try:
            # Run retrieval quality evaluation
            self.logger.info("=" * 60)
            retrieval_results = self.run_retrieval_evaluation(
                num_queries=retrieval_queries,
                top_k=10,
                benchmark_path=benchmark_dir / 'retrieval_benchmark.json' if generate_benchmarks else None
            )
            all_results['retrieval_quality'] = retrieval_results

        except Exception as e:
            self.logger.error(f"Retrieval quality evaluation failed: {e}")
            all_results['retrieval_quality'] = {'error': str(e)}

        try:
            # Run temporal coherence evaluation
            self.logger.info("=" * 60)
            coherence_results = self.run_coherence_evaluation(
                num_sequences=coherence_sequences,
                sequence_length=5,
                benchmark_path=benchmark_dir / 'coherence_benchmark.json' if generate_benchmarks else None
            )
            all_results['temporal_coherence'] = coherence_results

        except Exception as e:
            self.logger.error(f"Temporal coherence evaluation failed: {e}")
            all_results['temporal_coherence'] = {'error': str(e)}

        try:
            # Run performance evaluation
            self.logger.info("=" * 60)
            performance_results = self.run_performance_evaluation(
                num_queries=performance_queries,
                iterations=3,
                benchmark_path=benchmark_dir / 'performance_benchmark.json' if generate_benchmarks else None
            )
            all_results['performance'] = performance_results

        except Exception as e:
            self.logger.error(f"Performance evaluation failed: {e}")
            all_results['performance'] = {'error': str(e)}

        # Save raw results
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        self.logger.info(f"Raw results saved to {results_path}")

        # Generate comprehensive report
        try:
            report_generator = ReportGenerator(output_dir)
            report_path = report_generator.generate_full_report(all_results)
            self.logger.info(f"Evaluation report generated: {report_path}")
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")

        # Generate visualizations
        try:
            viz_generator = VisualizationGenerator(output_dir / "visualizations")
            visualizations = viz_generator.create_all_visualizations(all_results)
            self.logger.info(f"Generated {len(visualizations)} visualizations")
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")

        # Print summary
        self.print_evaluation_summary(all_results)

        self.logger.info("Comprehensive evaluation complete!")

        return all_results

    def print_evaluation_summary(self, results: dict):
        """
        Print evaluation summary.

        Args:
            results: Complete evaluation results
        """
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("=" * 80)

        # Modality Prediction
        if 'modality_prediction' in results:
            modality = results['modality_prediction']
            if 'error' not in modality:
                accuracy = modality.get('overall_accuracy', 0.0)
                validation = modality.get('validation', {})
                passed = validation.get('passed', False)
                status = "PASS" if passed else "FAIL"

                self.logger.info(f"Modality Prediction Accuracy: {accuracy:.2%} [{status}]")
                self.logger.info(f"  Target: 85%, Actual: {accuracy:.2%}")

        # Retrieval Quality
        if 'retrieval_quality' in results:
            retrieval = results['retrieval_quality']
            if 'error' not in retrieval:
                precision_5 = retrieval['retrieval_metrics']['precision_at_k']['P@5']
                validation = retrieval.get('validation', {})
                passed = validation.get('passed', False)
                status = "PASS" if passed else "FAIL"

                self.logger.info(f"Retrieval Precision@5: {precision_5:.2%} [{status}]")
                self.logger.info(f"  Target: 90%, Actual: {precision_5:.2%}")

        # Temporal Coherence
        if 'temporal_coherence' in results:
            coherence = results['temporal_coherence']
            if 'error' not in coherence:
                coherence_precision = coherence['coherence_metrics']['summary']['coherence_precision']['mean']
                validation = coherence.get('validation', {})
                passed = validation.get('passed', False)
                status = "PASS" if passed else "FAIL"

                self.logger.info(f"Temporal Coherence: {coherence_precision:.2%} [{status}]")
                self.logger.info(f"  Target: 95%, Actual: {coherence_precision:.2%}")

        # Performance
        if 'performance' in results:
            performance = results['performance']
            if 'error' not in performance:
                query_time = performance['query_time_stats']['mean']
                validation = performance.get('validation', {})
                passed = validation.get('passed', False)
                status = "PASS" if passed else "FAIL"

                self.logger.info(f"Average Query Time: {query_time:.2f}s [{status}]")
                self.logger.info(f"  Target: 4.0s, Actual: {query_time:.2f}s")

        self.logger.info("=" * 80)


def main():
    """Main entry point for comprehensive evaluation."""
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of Multi-Modal RAG system')

    parser.add_argument('--output-dir', type=str, default='data/evaluation/results/comprehensive',
                       help='Output directory for evaluation results')
    parser.add_argument('--no-reranker', action='store_true',
                       help='Disable reranking for evaluation')
    parser.add_argument('--no-aman', action='store_true',
                       help='Exclude Aman.ai content from evaluation')
    parser.add_argument('--modality-queries', type=int, default=100,
                       help='Number of modality prediction test queries')
    parser.add_argument('--retrieval-queries', type=int, default=50,
                       help='Number of retrieval quality test queries')
    parser.add_argument('--coherence-sequences', type=int, default=20,
                       help='Number of temporal coherence test sequences')
    parser.add_argument('--performance-queries', type=int, default=30,
                       help='Number of performance test queries')
    parser.add_argument('--use-existing-benchmarks', action='store_true',
                       help='Use existing benchmarks instead of generating new ones')

    args = parser.parse_args()

    # Run comprehensive evaluation
    evaluation = ComprehensiveEvaluation(
        use_reranker=not args.no_reranker,
        include_aman=not args.no_aman
    )

    evaluation.run_comprehensive_evaluation(
        output_dir=args.output_dir,
        modality_queries=args.modality_queries,
        retrieval_queries=args.retrieval_queries,
        coherence_sequences=args.coherence_sequences,
        performance_queries=args.performance_queries,
        generate_benchmarks=not args.use_existing_benchmarks
    )


if __name__ == "__main__":
    main()