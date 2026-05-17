"""
Create FAISS Index for CS224n Video Chunks (using already processed data)
"""

import sys
from pathlib import Path
import pickle
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store.faiss_manager import FAISSManager


def create_cs224n_index():
    """Create FAISS index from processed CS224n video chunks"""

    print("="*80)
    print("CREATING CS224N VIDEO FAISS INDEX")
    print("="*80)

    chunks_dir = Path("data/processed/video_chunks_cs224n")

    # Load chunks
    chunks_path = chunks_dir / "cs224n_video_chunks.pkl"
    print(f"\n[1] Loading chunks from {chunks_path}")

    if not chunks_path.exists():
        print(f"ERROR: Chunks file not found at {chunks_path}")
        print("Please run process_cs224n_pipeline.py first")
        return None

    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)

    print(f"    Loaded {len(chunks)} chunks")

    # Load embeddings
    embeddings_path = chunks_dir / "cs224n_video_embeddings.npy"
    print(f"\n[2] Loading embeddings from {embeddings_path}")

    if not embeddings_path.exists():
        print(f"ERROR: Embeddings file not found at {embeddings_path}")
        print("Please run process_cs224n_pipeline.py first")
        return None

    embeddings = np.load(embeddings_path)
    print(f"    Loaded embeddings: {embeddings.shape}")

    # Create FAISS index
    print(f"\n[3] Creating FAISS index...")

    faiss_manager = FAISSManager()

    # Build index
    faiss_manager.create_index(embeddings, chunks)
    print(f"    Index created: {faiss_manager.get_index_info()}")

    # Save index
    index_path = chunks_dir / "cs224n_video_index.faiss"
    metadata_path = chunks_dir / "cs224n_video_metadata.pkl"

    print(f"\n[4] Saving FAISS index...")
    faiss_manager.save_index(str(index_path), str(metadata_path))

    print(f"    Saved index: {index_path}")
    print(f"    Saved metadata: {metadata_path}")

    print(f"\n{'='*80}")
    print(f"CS224N VIDEO INDEX CREATED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"Total chunks indexed: {len(chunks)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Count tutorials vs lectures
    tutorial_count = sum(1 for chunk in chunks if chunk.get('is_tutorial', False))
    lecture_count = len(chunks) - tutorial_count

    print(f"\nContent breakdown:")
    print(f"  Lectures: {lecture_count} chunks")
    print(f"  Tutorials: {tutorial_count} chunks")

    # Sample chunks
    print(f"\nSample chunks (first 3):")
    for i, chunk in enumerate(chunks[:3], 1):
        tutorial_marker = " [TUTORIAL]" if chunk.get('is_tutorial') else ""
        print(f"\n  {i}. {chunk['source_name']}{tutorial_marker} - {chunk['chunk_id']}")
        print(f"     Timestamp: {chunk['timestamp_start']/60:.1f}-{chunk['timestamp_end']/60:.1f}min")
        print(f"     URL: {chunk['timestamp_url']}")
        print(f"     Text: {chunk['text'][:100]}...")

    return faiss_manager


if __name__ == "__main__":
    create_cs224n_index()
