"""
Index Building Script - Generate embeddings and build FAISS index
Main entry point for Phase 3
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vector_store.faiss_manager import FAISSManager
from src.utils.logger import setup_logger
from src.utils.helpers import Timer, load_pickle
from src.config import EMBEDDINGS_DIR, INDICES_DIR, CHUNKS_DIR, EMBEDDING_CONFIG, FAISS_CONFIG


def main():
    """Main index building pipeline"""
    logger = setup_logger("index_building")

    logger.info("=" * 60)
    logger.info("Starting Phase 3: Embeddings + Vector Index")
    logger.info("=" * 60)

    # Phase 1: Load chunks
    logger.info("\nPhase 1: Loading chunks...")
    chunks_path = CHUNKS_DIR / "chunks.pkl"

    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        logger.error("Please run ingestion first: python scripts/01_ingest_pdfs.py")
        return 1

    with Timer("Chunk loading"):
        chunks = load_pickle(chunks_path)

    logger.info(f"Loaded {len(chunks)} chunks")

    # Phase 2: Generate embeddings
    logger.info("\nPhase 2: Generating embeddings...")
    generator = EmbeddingGenerator(EMBEDDING_CONFIG)

    # Display model info
    model_info = generator.get_embedding_info()
    logger.info(f"Model: {model_info['model_name']}")
    logger.info(f"Dimension: {model_info['embedding_dimension']}")
    logger.info(f"Device: {model_info['device']}")
    logger.info(f"Max sequence length: {model_info['max_sequence_length']}")

    with Timer("Embedding generation"):
        embeddings = generator.generate_embeddings_from_chunks(
            chunks,
            show_progress=True
        )

    logger.info(f"Generated embeddings shape: {embeddings.shape}")

    # Save embeddings
    embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"
    generator.save_embeddings(embeddings, embeddings_path)

    # Phase 3: Build FAISS index
    logger.info("\nPhase 3: Building FAISS index...")
    manager = FAISSManager(FAISS_CONFIG)

    # Update dimension based on actual embeddings
    FAISS_CONFIG["dimension"] = embeddings.shape[1]

    with Timer("Index building"):
        index = manager.create_index(embeddings, chunks)

    logger.info(f"Index created with {index.ntotal} vectors")

    # Phase 4: Save index
    logger.info("\nPhase 4: Saving index...")
    index_path = INDICES_DIR / "vector_index.faiss"
    chunks_path = INDICES_DIR / "chunks_metadata.pkl"

    manager.save_index(index_path, chunks_path)

    # Phase 5: Test search
    logger.info("\nPhase 5: Testing search...")

    # Load index back
    manager.load_index(index_path, chunks_path)

    # Test query
    test_queries = [
        "What is gradient descent?",
        "How does backpropagation work?",
        "Explain convolutional neural networks"
    ]

    logger.info(f"\nTesting {len(test_queries)} sample queries:")

    for query in test_queries:
        # Generate query embedding
        query_embedding = generator.generate_embeddings([query], show_progress=False)

        # Search
        distances, indices = manager.search(query_embedding, top_k=3)

        # Retrieve chunks
        results = manager.retrieve_chunks(indices, distances)

        logger.info(f"\nQuery: '{query}'")
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. {result['source_name']} - {result.get('chapter', 'N/A')} "
                       f"(score: {result['relevance_score']:.4f})")
            logger.info(f"     {result['text'][:100]}...")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Index Building Summary")
    logger.info("=" * 60)
    logger.info(f"Chunks processed: {len(chunks):,}")
    logger.info(f"Embeddings generated: {embeddings.shape[0]:,}")
    logger.info(f"Embedding dimension: {embeddings.shape[1]}")
    logger.info(f"Index vectors: {index.ntotal:,}")
    logger.info(f"Index type: {FAISS_CONFIG['index_type']}")

    # File sizes
    embeddings_size_mb = embeddings_path.stat().st_size / (1024 * 1024)
    index_size_mb = index_path.stat().st_size / (1024 * 1024)
    chunks_size_mb = chunks_path.stat().st_size / (1024 * 1024)

    logger.info(f"\nFile sizes:")
    logger.info(f"  Embeddings: {embeddings_size_mb:.2f}MB")
    logger.info(f"  Index: {index_size_mb:.2f}MB")
    logger.info(f"  Chunks metadata: {chunks_size_mb:.2f}MB")
    logger.info(f"  Total: {embeddings_size_mb + index_size_mb + chunks_size_mb:.2f}MB")

    logger.info("\n" + "=" * 60)
    logger.info("Phase 3 Complete!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
