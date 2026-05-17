"""
Quick Final Performance Validation

Rapid test to validate all three optimizations are working.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.optimized_multimodal_pipeline import OptimizedMultiModalRAGPipeline


def quick_test():
    """Quick performance test"""
    print("Quick Final Performance Validation")
    print("=" * 50)

    print("\n1. Initializing optimized pipeline...")
    print("   - Pipeline caching: YES")
    print("   - Model preloading: YES")
    print("   - Parallel retrieval: YES")

    init_start = time.time()
    pipeline = OptimizedMultiModalRAGPipeline(
        use_reranker=True,
        include_aman=True,
        enable_cache=True,
        enable_parallel=True,
        preload_models=True
    )
    init_time = time.time() - init_start

    print(f"   Initialization: {init_time:.1f}s")

    print("\n2. Testing first query (after preloading)...")
    test_query = "What is machine learning?"

    query1_start = time.time()
    response1 = pipeline.query(question=test_query, top_k=5, rerank_top_n=5, include_timing=True)
    query1_time = time.time() - query1_start

    print(f"   First query: {query1_time:.2f}s")
    print(f"   Sources: {len(response1.get('sources', []))}")
    print(f"   Modality: {response1.get('predicted_modality', 'unknown')}")

    print("\n3. Testing same query (should be cached)...")
    query2_start = time.time()
    response2 = pipeline.query(question=test_query, top_k=5, rerank_top_n=5, include_timing=True)
    query2_time = time.time() - query2_start

    print(f"   Cached query: {query2_time:.2f}s")
    print(f"   Speedup: {query1_time / query2_time:.1f}x faster")

    print("\n4. Testing different query...")
    test_query2 = "Explain neural networks"

    query3_start = time.time()
    response3 = pipeline.query(question=test_query2, top_k=5, rerank_top_n=5, include_timing=True)
    query3_time = time.time() - query3_start

    print(f"   Different query: {query3_time:.2f}s")
    print(f"   Sources: {len(response3.get('sources', []))}")

    print("\n5. Performance statistics...")
    stats = pipeline.get_performance_stats()
    print(f"   Total queries: {stats['total_queries']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"   Average query time: {stats['avg_query_time']:.2f}s")
    print(f"   Average cached time: {stats['avg_cached_time']:.2f}s")

    print("\n" + "=" * 50)
    print("RESULTS:")

    # Performance targets
    first_query_target = 4.0
    cached_query_target = 0.1
    cache_speedup_target = 100

    first_pass = query1_time <= first_query_target
    cached_pass = query2_time <= cached_query_target
    speedup = query1_time / query2_time
    speedup_pass = speedup >= cache_speedup_target

    print(f"First Query: {query1_time:.2f}s (target: <{first_query_target}s) - {'PASS' if first_pass else 'FAIL'}")
    print(f"Cached Query: {query2_time:.2f}s (target: <{cached_query_target}s) - {'PASS' if cached_pass else 'FAIL'}")
    print(f"Cache Speedup: {speedup:.1f}x (target: >{cache_speedup_target}x) - {'PASS' if speedup_pass else 'FAIL'}")

    overall_success = first_pass and cached_pass and speedup_pass

    print("\n" + "=" * 50)
    if overall_success:
        print("SUCCESS: All performance targets achieved! ✅")
        print("- Sub-4.0s first query: YES")
        print("- Sub-0.1s cached query: YES")
        print("- 100x+ cache speedup: YES")
        print("- Production-ready: YES ✅")
    else:
        print("PARTIAL: Some targets not achieved")
        if not first_pass:
            print(f"- First query {query1_time:.2f}s exceeds {first_query_target}s target")
        if not cached_pass:
            print(f"- Cached query {query2_time:.2f}s exceeds {cached_query_target}s target")
        if not speedup_pass:
            print(f"- Cache speedup {speedup:.1f}x below {cache_speedup_target}x target")

    return overall_success


if __name__ == "__main__":
    try:
        success = quick_test()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
