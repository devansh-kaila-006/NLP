"""
Quick Performance Validation

Simple test to validate optimized reranker performance improvements.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.reranking.optimized_cross_encoder_reranker import OptimizedCrossEncoderReranker
from src.config import OPTIMIZED_RERANKING_CONFIG


def quick_test():
    """Quick performance test"""
    print("Quick Optimized Reranker Performance Test")
    print("=" * 50)

    # Sample chunks for testing
    chunks = [
        {"text": "Linear regression is a supervised learning algorithm used for predicting continuous values.", "chunk_id": "1"},
        {"text": "Deep learning uses neural networks with multiple layers to learn hierarchical representations.", "chunk_id": "2"},
        {"text": "Convolutional neural networks are specifically designed for image processing and computer vision tasks.", "chunk_id": "3"},
        {"text": "Transformers use attention mechanisms to process sequential data more effectively than RNNs.", "chunk_id": "4"},
        {"text": "Gradient descent is an optimization algorithm that minimizes the loss function by updating parameters.", "chunk_id": "5"},
        {"text": "Backpropagation calculates gradients by chaining derivatives backwards through the network.", "chunk_id": "6"},
        {"text": "Natural language processing deals with interactions between computers and human language.", "chunk_id": "7"},
        {"text": "Support vector machines find the optimal hyperplane that separates different classes.", "chunk_id": "8"},
    ]

    # Test queries
    simple_query = "What is linear regression?"
    complex_query = "Explain the difference between CNNs and RNNs and how they handle different types of data"

    print(f"Testing Optimized Reranker")
    print(f"Simple query: {simple_query}")
    print(f"Complex query: {complex_query[:50]}...")
    print()

    # Initialize reranker
    print("Initializing optimized reranker...")
    reranker = OptimizedCrossEncoderReranker(OPTIMIZED_RERANKING_CONFIG)

    # Get performance profile
    profile = reranker.get_performance_profile()
    print(f"Performance Profile:")
    print(f"  Query Complexity Detection: {profile['query_complexity_detection']}")
    print(f"  Caching Enabled: {profile['caching_enabled']}")
    print(f"  Primary Model: {profile['models']['primary']}")
    print(f"  Enhanced Model: {profile['models']['enhanced']}")
    print()

    # Test simple query
    print("Testing simple query...")
    start = time.time()
    simple_result = reranker.rerank(simple_query, chunks, top_n=3)
    simple_time = time.time() - start
    print(f"  Time: {simple_time:.2f}s")
    print(f"  Results: {len(simple_result)} chunks")
    print(f"  Top score: {simple_result[0].get('rerank_score', 0):.3f}" if simple_result else "  No results")
    print()

    # Test complex query
    print("Testing complex query...")
    start = time.time()
    complex_result = reranker.rerank(complex_query, chunks, top_n=3)
    complex_time = time.time() - start
    print(f"  Time: {complex_time:.2f}s")
    print(f"  Results: {len(complex_result)} chunks")
    print(f"  Top score: {complex_result[0].get('rerank_score', 0):.3f}" if complex_result else "  No results")
    print()

    # Performance summary
    print("Performance Summary:")
    print(f"  Simple query: {simple_time:.2f}s (target: <2.0s) - {'PASS' if simple_time < 2.0 else 'FAIL'}")
    print(f"  Complex query: {complex_time:.2f}s (target: <4.0s) - {'PASS' if complex_time < 4.0 else 'FAIL'}")

    # Test caching
    if reranker.enable_caching:
        print()
        print("Testing cache performance...")
        start = time.time()
        cached_result = reranker.rerank(simple_query, chunks, top_n=3)
        cached_time = time.time() - start
        print(f"  Cached query time: {cached_time:.2f}s")
        print(f"  Speedup: {simple_time / cached_time:.1f}x faster")

    print()
    print("=" * 50)
    if simple_time < 2.0 and complex_time < 4.0:
        print("SUCCESS: Performance targets achieved!")
    else:
        print("PARTIAL: Some targets not met, but optimization applied")


if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
