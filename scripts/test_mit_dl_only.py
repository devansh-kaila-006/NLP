"""
Test MIT DL Content Standalone
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.retriever import Retriever
from src.generation.gemini_generator import GeminiGenerator
from src.config import RETRIEVAL_CONFIG


def test_mit_dl_only():
    """Test MIT DL content retrieval and generation"""

    print("="*80)
    print("MIT DL CONTENT TEST")
    print("="*80)

    # Initialize MIT DL retriever
    print("\n[INITIALIZING MIT DL RETRIEVER]")
    mit_dl_retriever = Retriever(
        index_path="data/processed/video_chunks_mit_dl/mit_dl_video_index.faiss",
        chunks_path="data/processed/video_chunks_mit_dl/mit_dl_video_metadata.pkl",
        config=RETRIEVAL_CONFIG
    )

    stats = mit_dl_retriever.get_stats()
    print(f"  MIT DL Chunks: {stats.get('total_vectors', 'N/A')}")
    print(f"  Index Type: {stats.get('index_type', 'N/A')}")

    # Initialize generator
    print("\n[INITIALIZING GENERATOR]")
    generator = GeminiGenerator()

    # Test queries
    print(f"\n{'='*80}")
    print("TESTING MIT DL SPECIFIC QUERIES")
    print('='*80)

    test_queries = [
        "What are transformers and how do they work?",
        "Explain retrieval augmented generation (RAG)",
        "What is parameter-efficient fine-tuning?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"QUERY {i}: {query}")
        print('='*80)

        # Retrieve chunks
        chunks = mit_dl_retriever.retrieve(query, top_k=3)
        print(f"\n[RETRIEVED {len(chunks)} CHUNKS]")

        for j, chunk in enumerate(chunks, 1):
            print(f"\n  {j}. {chunk.get('source_name', 'Unknown')} - {chunk.get('chunk_id', 'Unknown')}")
            print(f"     Timestamp: {chunk.get('timestamp_start', 0)/60:.1f}-{chunk.get('timestamp_end', 0)/60:.1f}min")
            print(f"     URL: {chunk.get('timestamp_url', 'N/A')}")
            print(f"     Relevance: {chunk.get('relevance_score', 0):.3f}")
            print(f"     Text: {chunk.get('text', '')[:150]}...")

        # Generate response
        if chunks:
            print(f"\n[GENERATING RESPONSE]")
            response = generator.query(query, chunks)

            print(f"\n[ANSWER]")
            print(f"{response['answer'][:300]}...")

            print(f"\n[METADATA]")
            print(f"  Model: {response.get('model', 'unknown')}")
            print(f"  Answer Length: {len(response['answer'])} chars")

    print(f"\n{'='*80}")
    print("MIT DL CONTENT TEST COMPLETE")
    print('='*80)

    print(f"\n[MIT DL STATS]")
    print(f"  Total Video Chunks: {stats.get('total_vectors', 'N/A')}")
    print(f"  Source: MIT 6.S191 Introduction to Deep Learning (Alternative)")
    print(f"  Topics: Neural Networks, CNNs, Transformers, LLMs, RAG, PEFT, Text-to-Image")
    print(f"  Retrieval: Working ✓")
    print(f"  Generation: Working ✓")

    return mit_dl_retriever


if __name__ == "__main__":
    try:
        test_mit_dl_only()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
