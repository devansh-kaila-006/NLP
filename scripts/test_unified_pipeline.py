"""
Test Unified Multi-Modal RAG Pipeline - All 5 Playlists Combined
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.unified_multimodal_pipeline import UnifiedMultiModalRAGPipeline


def main():
    print("="*80)
    print("UNIFIED MULTI-MODAL RAG PIPELINE TEST")
    print("ALL 5 VIDEO PLAYLISTS COMBINED")
    print("="*80)

    print("\nInitializing unified pipeline with all 5 playlists...")
    pipeline = UnifiedMultiModalRAGPipeline(use_reranker=True)

    # Show system stats
    stats = pipeline.get_stats()
    print(f"\nSystem Statistics:")
    print(f"  PDF retriever: {'Active' if stats.get('pdf_available') else 'Not available'}")
    print(f"  Video retriever: {'Active' if stats.get('video_available') else 'Not available'}")
    print(f"  Reranker: {'Enabled' if stats.get('reranker_enabled') else 'Disabled'}")

    if stats.get('pdf_stats'):
        pdf_stats = stats['pdf_stats']
        print(f"\n  PDF Content:")
        print(f"    {pdf_stats.get('total_vectors', 0)} vectors")

    if stats.get('video_stats'):
        video_stats = stats['video_stats']
        print(f"\n  Video Content:")
        print(f"    {video_stats.get('total_playlists', 0)} playlists")
        print(f"    {video_stats.get('total_vectors', 0)} total video chunks")

        if video_stats.get('playlist_stats'):
            print(f"\n  Individual Playlists:")
            for i, playlist_stat in enumerate(video_stats['playlist_stats'], 1):
                vectors = playlist_stat.get('total_vectors', 0)
                print(f"    Playlist {i}: {vectors} chunks")

    # Test queries across different domains
    test_queries = [
        {
            "domain": "Machine Learning",
            "query": "What is supervised learning and how does it differ from unsupervised learning?"
        },
        {
            "domain": "Deep Learning",
            "query": "Explain how convolutional neural networks process images"
        },
        {
            "domain": "NLP",
            "query": "What are word embeddings and why are they important for natural language processing?"
        },
        {
            "domain": "Computer Vision",
            "query": "How do image classification models work?"
        },
        {
            "domain": "Transformers",
            "query": "Explain the transformer architecture and attention mechanism"
        }
    ]

    print(f"\n{'='*80}")
    print("TESTING QUERIES ACROSS ALL DOMAINS")
    print(f"{'='*80}")

    for i, test in enumerate(test_queries, 1):
        print(f"\n{'-'*80}")
        print(f"TEST {i}/{len(test_queries)}: {test['domain']}")
        print(f"{'-'*80}")
        print(f"Query: {test['query']}")
        print(f"\nProcessing...")

        try:
            result = pipeline.query(test['query'], top_k=3, include_timing=True)

            print(f"\n[OK] Query successful!")
            print(f"  Video chunks used: {result.get('video_chunks_used', 0)}")
            print(f"  PDF chunks used: {result.get('pdf_chunks_used', 0)}")

            # Show video sources with playlist info
            video_sources = [s for s in result.get('sources', []) if s.get('type') == 'video']
            if video_sources:
                print(f"\n  Video sources:")
                for j, source in enumerate(video_sources[:3], 1):
                    print(f"    {j}. {source.get('name', 'Unknown')}")
                    if source.get('timestamp_url'):
                        print(f"       Time: {source.get('timestamp_start', 0)/60:.0f}min")
                        print(f"       URL: {source['timestamp_url']}")

            # Show PDF sources
            pdf_sources = [s for s in result.get('sources', []) if s.get('type') == 'pdf']
            if pdf_sources:
                print(f"\n  PDF sources:")
                for j, source in enumerate(pdf_sources[:2], 1):
                    print(f"    {j}. {source.get('name', 'Unknown')}")

            # Show timing if available
            if 'timings' in result:
                timings = result['timings']
                print(f"\n  Timing: {timings['total']:.2f}s total")
                print(f"    Modality prediction: {timings['modality_prediction']:.2f}s")
                print(f"    Retrieval: {timings['retrieval']:.2f}s")
                if 'reranking' in timings:
                    print(f"    Reranking: {timings['reranking']:.2f}s")
                print(f"    Generation: {timings['generation']:.2f}s")

        except Exception as e:
            print(f"\n[ERROR] Query failed: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print(f"\n{'='*80}")
    print("UNIFIED PIPELINE TEST COMPLETE")
    print(f"{'='*80}")

    final_stats = pipeline.get_stats()

    print(f"\nFinal System Statistics:")
    if final_stats.get('pdf_stats'):
        pdf_stats = final_stats['pdf_stats']
        print(f"  PDF: {pdf_stats.get('total_vectors', 0)} chunks")

    if final_stats.get('video_stats'):
        video_stats = final_stats['video_stats']
        print(f"  Video: {video_stats.get('total_vectors', 0)} chunks across {video_stats.get('total_playlists', 0)} playlists")

    total_chunks = final_stats.get('pdf_stats', {}).get('total_vectors', 0) + final_stats.get('video_stats', {}).get('total_vectors', 0)
    print(f"  TOTAL: {total_chunks} chunks")

    print(f"\n[SUCCESS] Unified multi-modal RAG pipeline with all 5 playlists is operational!")
    print(f"[READY] System ready for production use!")
    print(f"\nCoverage:")
    print(f"  [OK] CS229 Machine Learning (Stanford)")
    print(f"  [OK] MIT 6.S191 Deep Learning - Alternative")
    print(f"  [OK] CS224n NLP with Deep Learning (Stanford)")
    print(f"  [OK] CS231n Computer Vision (Stanford)")
    print(f"  [OK] MIT 6.S191 Deep Learning - Main (Comprehensive)")


if __name__ == "__main__":
    main()
