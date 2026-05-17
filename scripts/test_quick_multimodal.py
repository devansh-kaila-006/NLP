"""
Quick Test of Multi-Modal RAG Pipeline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.multimodal_rag_pipeline import MultiModalRAGPipeline


def quick_test():
    """Quick test of multi-modal functionality"""

    print("="*60)
    print("QUICK MULTI-MODAL RAG TEST")
    print("="*60)

    # Initialize pipeline
    print("\n[INITIALIZING]")
    pipeline = MultiModalRAGPipeline(
        pdf_index_path="data/processed/indices/vector_index.faiss",
        pdf_chunks_path="data/processed/indices/chunks_metadata.pkl",
        video_index_path="data/processed/video_chunks/video_index.faiss",
        video_chunks_path="data/processed/video_chunks/video_metadata.pkl",
        use_reranker=True
    )

    # Test query
    query = "What is machine learning?"
    print(f"\n[TEST QUERY]")
    print(f"Query: {query}")

    response = pipeline.query(query, include_timing=True)

    print(f"\n[RESPONSE]")
    print(f"Answer: {response['answer'][:200]}...")

    print(f"\n[SOURCES]")
    for source in response['sources'][:3]:
        if source['type'] == 'video':
            print(f"  [VIDEO] {source['name']} - {source['timestamp']/60:.0f}min")
            print(f"          URL: {source['timestamp_url']}")
        else:
            print(f"  [PDF] {source['name']}")

    print(f"\n[NOVELTY 3: Cross-Modal Prediction]")
    print(f"  Modality Scores: {response['modality_scores']}")
    print(f"  PDF Chunks: {response['pdf_chunks_used']}")
    print(f"  Video Chunks: {response['video_chunks_used']}")

    if 'timings' in response:
        timings = response['timings']
        print(f"\n[TIMING]")
        print(f"  Total: {timings['total']:.2f}s")
        print(f"  Modality Prediction: {timings['modality_prediction']:.2f}s")
        print(f"  Retrieval: {timings['retrieval']:.2f}s")
        if 'reranking' in timings:
            print(f"  Reranking: {timings['reranking']:.2f}s")
        print(f"  Generation: {timings['generation']:.2f}s")

    print(f"\n[VIDEO LINKS]")
    if response.get('video_links'):
        for link in response['video_links'][:2]:
            print(f"  {link['timestamp']}: {link['url']}")
    else:
        print("  No video links in this response")

    print(f"\n{'='*60}")
    print(f"SUCCESS: Multi-Modal RAG Working!")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
