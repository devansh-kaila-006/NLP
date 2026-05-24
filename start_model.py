"""
Start the Optimized Multi-Modal RAG Model
"""
import sys
sys.path.insert(0, '.')

print('INITIALIZING Optimized Multi-Modal RAG Pipeline...')

from src.pipeline.optimized_multimodal_pipeline import OptimizedMultiModalRAGPipeline

try:
    # Initialize pipeline with all optimizations enabled
    print('Loading pipeline with models...')
    pipeline = OptimizedMultiModalRAGPipeline(
        use_reranker=True,
        include_aman=True,
        enable_cache=True,
        enable_parallel=True,
        preload_models=True
    )

    print('Pipeline initialized successfully!')
    print('Model Components:')
    print(f'   - PDF Retriever: {pipeline.pdf_retriever is not None}')
    print(f'   - Video Retriever: {pipeline.video_retriever is not None}')
    print(f'   - Aman.ai Retriever: {pipeline.aman_retriever is not None}')
    print(f'   - Reranker: {pipeline.reranker is not None}')
    print(f'   - Cache Enabled: {pipeline.enable_cache}')
    print(f'   - Parallel Retrieval: {pipeline.enable_parallel}')

    # Test a quick query
    print('Testing pipeline with sample query...')
    result = pipeline.query('What is machine learning?', top_k=3, rerank_top_n=2)
    print('Test query successful!')
    print(f'   - Answer length: {len(result["answer"])} chars')
    print(f'   - Sources: {len(result["sources"])}')
    print(f'   - Predicted modality: {result["predicted_modality"]}')

    print('')
    print('=== MODEL IS LIVE AND READY ===')
    print('Frontend: http://localhost:7861')
    print('')
    print('Performance Stats:')
    stats = pipeline.get_performance_stats()
    for key, value in stats.items():
        print(f'   {key}: {value}')

    # Keep the pipeline alive
    print('')
    print('Pipeline is running. Press Ctrl+C to stop.')
    import time
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print('Pipeline stopped by user.')
except Exception as e:
    print(f'Error initializing pipeline: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)