"""
Simple test script for the evaluation framework without reranker
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.benchmarks import QueryGenerator
from src.evaluation.metrics import RetrievalMetrics, AccuracyMetrics, CoherenceMetrics


def test_metrics_framework():
    """Test the metrics calculation framework."""

    print("=" * 80)
    print("TESTING EVALUATION METRICS FRAMEWORK")
    print("=" * 80)

    try:
        # 1. Test Query Generator
        print("\n1. Testing Query Generator...")
        query_generator = QueryGenerator()
        test_queries = query_generator.generate_queries(
            domains=['machine_learning'],
            query_types=['conceptual'],
            num_queries_per_category=10
        )
        print(f"   Generated {len(test_queries)} test queries")
        print(f"   Sample query: {test_queries[0]['text']}")
        print(f"   Expected modality: {test_queries[0]['expected_modality']}")

        # 2. Test Accuracy Metrics
        print("\n2. Testing Accuracy Metrics...")
        predictions = ['video', 'pdf', 'video', 'aman', 'video']
        ground_truth = ['video', 'pdf', 'pdf', 'aman', 'video']
        confidences = [0.9, 0.8, 0.7, 0.85, 0.95]

        accuracy = AccuracyMetrics.calculate_accuracy(predictions, ground_truth)
        print(f"   Accuracy: {accuracy:.2%}")

        confusion_result = AccuracyMetrics.calculate_confusion_matrix(
            predictions, ground_truth, ['video', 'pdf', 'aman']
        )
        print(f"   Confusion matrix calculated successfully")
        print(f"   Per-class accuracy: {confusion_result['per_class_accuracy']}")

        # 3. Test Retrieval Metrics
        print("\n3. Testing Retrieval Metrics...")
        query_results = [
            {
                'retrieved': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5'],
                'relevant': ['doc1', 'doc3', 'doc5']
            },
            {
                'retrieved': ['doc6', 'doc7', 'doc8', 'doc9', 'doc10'],
                'relevant': ['doc6', 'doc7']
            }
        ]

        precision_5 = RetrievalMetrics.precision_at_k(
            query_results[0]['retrieved'],
            set(query_results[0]['relevant']),
            5
        )
        print(f"   Precision@5: {precision_5:.2%}")

        map_score = RetrievalMetrics.mean_average_precision(query_results)
        print(f"   MAP: {map_score:.2%}")

        mrr_score = RetrievalMetrics.mean_reciprocal_rank(query_results)
        print(f"   MRR: {mrr_score:.2%}")

        # 4. Test Coherence Metrics
        print("\n4. Testing Coherence Metrics...")
        retrieved_chunks = [
            {'chunk_id': 'chunk_0', 'timestamp_start': 0, 'timestamp_end': 30, 'text': 'Introduction to ML'},
            {'chunk_id': 'chunk_1', 'timestamp_start': 30, 'timestamp_end': 60, 'text': 'ML algorithms overview'},
            {'chunk_id': 'chunk_2', 'timestamp_start': 60, 'timestamp_end': 90, 'text': 'Deep learning basics'}
        ]
        expected_chunks = [
            {'chunk_id': 'chunk_0', 'timestamp_start': 0, 'timestamp_end': 30, 'text': 'Introduction to ML'},
            {'chunk_id': 'chunk_1', 'timestamp_start': 30, 'timestamp_end': 60, 'text': 'ML algorithms overview'},
            {'chunk_id': 'chunk_2', 'timestamp_start': 60, 'timestamp_end': 90, 'text': 'Deep learning basics'}
        ]

        flow_score = CoherenceMetrics.flow_score(retrieved_chunks)
        print(f"   Flow score: {flow_score:.2%}")

        coherence_result = CoherenceMetrics.coherence_precision(
            retrieved_chunks,
            ['chunk_0', 'chunk_1', 'chunk_2']
        )
        print(f"   Coherence precision: {coherence_result['coherence_precision']:.2%}")

        # 5. Test Comprehensive Metrics
        print("\n5. Testing Comprehensive Metrics Calculation...")
        all_retrieval_metrics = RetrievalMetrics.calculate_all_metrics(query_results, [1, 3, 5])
        print(f"   Calculated {len(all_retrieval_metrics)} metric categories")
        print(f"   Precision@K: {all_retrieval_metrics['precision_at_k']}")
        print(f"   NDCG@K: {all_retrieval_metrics['ndcg_at_k']}")

        print("\n" + "=" * 80)
        print("METRICS FRAMEWORK TEST COMPLETE")
        print("=" * 80)
        print("\nAll metrics components tested successfully!")
        print("\nKey Features:")
        print(" - Query generation across domains and query types")
        print(" - Classification metrics (accuracy, confusion matrix, calibration)")
        print(" - Information retrieval metrics (Precision@K, MAP, MRR, NDCG)")
        print(" - Temporal coherence metrics (flow score, coherence precision)")
        print(" - Comprehensive metric calculation with statistical analysis")

        return True

    except Exception as e:
        print(f"\nERROR: Metrics framework test failed!")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_metrics_framework()
    sys.exit(0 if success else 1)