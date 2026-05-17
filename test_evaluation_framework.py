"""
Test script for the comprehensive evaluation framework

Runs a quick evaluation to validate that all components are working correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.unified_multimodal_pipeline import UnifiedMultiModalRAGPipeline
from src.evaluation.benchmarks import BenchmarkBuilder, QueryGenerator
from src.evaluation.evaluators import ModalityPredictionEvaluator
from src.evaluation.reporting import ReportGenerator


def test_evaluation_framework():
    """Test the evaluation framework with a small sample."""

    print("=" * 80)
    print("TESTING COMPREHENSIVE EVALUATION FRAMEWORK")
    print("=" * 80)

    try:
        # 1. Test Pipeline Initialization
        print("\n1. Testing Pipeline Initialization...")
        pipeline = UnifiedMultiModalRAGPipeline(use_reranker=True, include_aman=True)
        print(f"   Pipeline initialized successfully!")
        print(f"   PDF chunks: {len(pipeline.pdf_chunks)}")
        print(f"   Video chunks: {len(pipeline.video_chunks)}")
        print(f"   Aman.ai chunks: {len(pipeline.aman_chunks)}")

        # 2. Test Query Generator
        print("\n2. Testing Query Generator...")
        query_generator = QueryGenerator()
        test_queries = query_generator.generate_queries(
            domains=['machine_learning'],
            query_types=['conceptual'],
            num_queries_per_category=5
        )
        print(f"   Generated {len(test_queries)} test queries")
        print(f"   Sample query: {test_queries[0]['text']}")

        # 3. Test Benchmark Builder
        print("\n3. Testing Benchmark Builder...")
        benchmark_builder = BenchmarkBuilder(pipeline)
        modality_benchmark = benchmark_builder.build_modality_benchmark(
            num_queries=20,
            output_path=Path("data/evaluation/test_modality_benchmark.json")
        )
        print(f"   Created modality benchmark with {modality_benchmark['num_queries']} queries")

        # 4. Test Modality Evaluator
        print("\n4. Testing Modality Prediction Evaluator...")
        evaluator = ModalityPredictionEvaluator(pipeline)
        results = evaluator.evaluate(modality_benchmark)
        print(f"   Modality prediction accuracy: {results['overall_accuracy']:.2%}")

        # 5. Test Report Generator
        print("\n5. Testing Report Generator...")
        report_gen = ReportGenerator(Path("data/evaluation/test_reports"))

        # Create minimal results for testing
        test_results = {
            'modality_prediction': results,
            'retrieval_quality': {
                'retrieval_metrics': {
                    'precision_at_k': {'P@5': 0.85},
                    'map': 0.78,
                    'mrr': 0.82
                },
                'validation': {'passed': True}
            },
            'temporal_coherence': {
                'coherence_metrics': {
                    'summary': {
                        'coherence_precision': {'mean': 0.92}
                    }
                },
                'validation': {'passed': True}
            },
            'performance': {
                'query_time_stats': {'mean': 3.5},
                'query_time_percentiles': {'p50': 3.2, 'p95': 4.8, 'p99': 5.2},
                'validation': {'passed': True}
            }
        }

        report_path = report_gen.generate_full_report(test_results, "test_report")
        print(f"   Generated test report: {report_path}")

        print("\n" + "=" * 80)
        print("EVALUATION FRAMEWORK TEST COMPLETE")
        print("=" * 80)
        print("\nAll components tested successfully!")
        print("\nNext steps:")
        print("1. Run comprehensive evaluation: python scripts/evaluation/run_comprehensive_evaluation.py")
        print("2. Review generated reports and visualizations")
        print("3. Analyze results and identify areas for improvement")

    except Exception as e:
        print(f"\nERROR: Evaluation framework test failed!")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = test_evaluation_framework()
    sys.exit(0 if success else 1)