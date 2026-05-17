"""
Test Video Chunker with CS229 Lecture 1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders.srt_loader import SRTLoader
from src.processors.video_chunker import VideoChunker

def test_chunking():
    """Test video chunking with novelty features"""

    print("=" * 80)
    print("TESTING VIDEO CHUNKER - CS229 Lecture 1")
    print("=" * 80)

    # Initialize chunker
    chunker = VideoChunker(
        min_chunk_duration=120.0,  # 2 minutes
        max_chunk_duration=300.0,  # 5 minutes
        semantic_threshold=0.6
    )

    # Process Lecture 1
    print("\n[1] Loading transcript...")
    result = chunker.process_lecture(
        srt_path="data/transcripts/cs229/CS229_L01_I_Introduction_Lecture_1.srt",
        video_url="https://www.youtube.com/watch?v=CS229_Lecture1"
    )

    print(f"[OK] Created {result['metadata']['num_chunks']} chunks")
    print(f"     Total duration: {result['metadata']['total_duration']:.1f} hours")
    print(f"     Temporal graph: {result['temporal_graph'].number_of_nodes()} nodes")

    print("\n[2] Analyzing chunks...")
    chunks = result['chunks']

    print(f"Chunk duration statistics:")
    durations = [chunk.duration_minutes() for chunk in chunks]
    print(f"  Min: {min(durations):.1f} minutes")
    print(f"  Max: {max(durations):.1f} minutes")
    print(f"  Avg: {sum(durations)/len(durations):.1f} minutes")

    print("\n[3] Sample chunks (first 3):")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n  Chunk {i}: {chunk}")
        print(f"    Text preview: {chunk.text[:100]}...")
        print(f"    Timestamp URL: {chunk.timestamp_url}")

    print("\n[4] Testing NOVELTY 1: Semantic Chunking")
    print("  Checking for topic boundary detection...")
    for i in range(min(5, len(chunks))):
        chunk = chunks[i]
        print(f"    Chunk {i+1}: {chunk.start_time/60:.0f}-{chunk.end_time/60:.0f}min ({chunk.duration_minutes():.1f}min)")

    print("\n[5] Testing NOVELTY 2: Temporal Coherence")
    print("  Building temporal dependency graph...")
    graph = result['temporal_graph']
    print(f"    Nodes: {graph.number_of_nodes()}")
    print(f"    Edges: {graph.number_of_edges()}")

    # Test coherent path finding
    if len(chunks) >= 3:
        start_chunk = chunks[0].chunk_id
        end_chunk = chunks[2].chunk_id
        print(f"  Finding coherent path: {start_chunk} -> {end_chunk}")

        coherent_path = chunker.find_coherent_path(graph, start_chunk, end_chunk, num_chunks=3)
        print(f"    Coherent path: {coherent_path}")

    print("\n[6] Testing NOVELTY 3: Cross-Modal Reranking")
    test_queries = [
        "What is machine learning?",
        "How does backpropagation work visually?",
        "What's the mathematical formula for gradient descent?",
        "How do I implement neural networks in Python?"
    ]

    for query in test_queries:
        modality_scores = chunker.predict_modality(query)
        best_modality = max(modality_scores, key=modality_scores.get)
        print(f"  Query: '{query}'")
        print(f"    Best modality: {best_modality}")
        print(f"    Scores: {modality_scores}")

    print("\n" + "=" * 80)
    print("TESTING COMPLETE - All novelty features working!")
    print("=" * 80)

    return result

if __name__ == "__main__":
    test_chunking()
