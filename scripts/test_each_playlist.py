"""
Test Each Video Playlist Individually
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.multimodal_rag_pipeline import MultiModalRAGPipeline


def test_playlist(playlist_name, video_index_path, video_chunks_path, test_query):
    """Test a single playlist"""

    print(f"\n{'='*80}")
    print(f"TESTING: {playlist_name}")
    print(f"{'='*80}")

    try:
        # Initialize with PDF + this playlist
        pipeline = MultiModalRAGPipeline(
            pdf_index_path="data/processed/indices/vector_index.faiss",
            pdf_chunks_path="data/processed/indices/chunks_metadata.pkl",
            video_index_path=video_index_path,
            video_chunks_path=video_chunks_path,
            use_reranker=True
        )

        print(f"\nQuery: {test_query}")

        result = pipeline.query(test_query, top_k=3)

        print(f"\n[OK] Test successful!")
        print(f"  Video chunks used: {result.get('video_chunks_used', 0)}")
        print(f"  PDF chunks used: {result.get('pdf_chunks_used', 0)}")

        # Show video sources
        video_sources = [s for s in result.get('sources', []) if s.get('type') == 'video']
        if video_sources:
            print(f"\n  Video sources:")
            for j, source in enumerate(video_sources[:3], 1):
                print(f"    {j}. {source.get('name', 'Unknown')}")
                if source.get('timestamp_url'):
                    print(f"       URL: {source['timestamp_url']}")

        # Get stats
        stats = pipeline.get_stats()
        if stats.get('video_stats'):
            video_stats = stats['video_stats']
            print(f"\n  Playlist stats: {video_stats.get('total_vectors', 0)} video chunks")

        return True, playlist_name

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        return False, playlist_name


def main():
    print("="*80)
    print("INDIVIDUAL PLAYLIST TESTING")
    print("="*80)

    # Define all 5 playlists
    playlists = [
        {
            "name": "CS229 Machine Learning (Stanford)",
            "video_index": "data/processed/video_chunks/video_index.faiss",
            "video_chunks": "data/processed/video_chunks/video_metadata.pkl",
            "query": "What is supervised learning?"
        },
        {
            "name": "MIT 6.S191 Deep Learning - Alternative (11 lectures)",
            "video_index": "data/processed/video_chunks_mit_dl/mit_dl_video_index.faiss",
            "video_chunks": "data/processed/video_chunks_mit_dl/mit_dl_video_metadata.pkl",
            "query": "Explain convolutional neural networks"
        },
        {
            "name": "CS224n NLP with Deep Learning (Stanford)",
            "video_index": "data/processed/video_chunks_cs224n/cs224n_video_index.faiss",
            "video_chunks": "data/processed/video_chunks_cs224n/cs224n_video_metadata.pkl",
            "query": "What are word embeddings?"
        },
        {
            "name": "CS231n Computer Vision (Stanford)",
            "video_index": "data/processed/video_chunks_cs231n/cs231n_video_index.faiss",
            "video_chunks": "data/processed/video_chunks_cs231n/cs231n_video_metadata.pkl",
            "query": "How do image classification models work?"
        },
        {
            "name": "MIT 6.S191 Deep Learning - Main (24 lectures + tutorial)",
            "video_index": "data/processed/video_chunks_mit_dl_main/mit_dl_main_video_index.faiss",
            "video_chunks": "data/processed/video_chunks_mit_dl_main/mit_dl_main_video_metadata.pkl",
            "query": "Explain the transformer architecture"
        }
    ]

    # Test each playlist
    results = []
    for playlist in playlists:
        success, name = test_playlist(
            playlist['name'],
            playlist['video_index'],
            playlist['video_chunks'],
            playlist['query']
        )
        results.append((name, success))

    # Print summary
    print(f"\n{'='*80}")
    print("PLAYLIST TEST SUMMARY")
    print(f"{'='*80}")

    for name, success in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"{status} {name}")

    successful = sum(1 for _, success in results if success)
    print(f"\nTotal: {successful}/{len(results)} playlists tested successfully")

    if successful == len(results):
        print(f"\n[SUCCESS] All 5 playlists are operational!")
        print(f"[READY] Ready to create unified multi-playlist pipeline")
    else:
        print(f"\n[WARNING] Some playlists failed - check errors above")


if __name__ == "__main__":
    main()
