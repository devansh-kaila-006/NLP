"""
Ingestion Script - Load all data sources and create chunks
This is the main entry point for data ingestion
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders.document_loader import DocumentLoader
from src.processors.semantic_chunker import SemanticChunker
from src.utils.logger import setup_logger
from src.utils.helpers import Timer
from src.config import CHUNKING_CONFIG, CHUNKS_DIR


def main():
    """Main ingestion pipeline"""
    logger = setup_logger("ingestion")

    logger.info("=" * 60)
    logger.info("Starting Data Ingestion Pipeline")
    logger.info("=" * 60)

    # Initialize components
    loader = DocumentLoader()
    chunker = SemanticChunker(CHUNKING_CONFIG)

    # Phase 1: Load documents
    logger.info("\nPhase 1: Loading documents...")
    with Timer("Document loading"):
        documents = loader.load_all_sources()

    if not documents:
        logger.error("No documents loaded! Exiting.")
        return 1

    logger.info(f"Loaded {len(documents)} documents")

    # Phase 2: Normalize documents
    logger.info("\nPhase 2: Normalizing documents...")
    with Timer("Document normalization"):
        normalized_docs = loader.normalize_documents(documents)

    logger.info(f"Normalized {len(normalized_docs)} documents")

    # Log statistics by source
    logger.info("\nDocument statistics by source:")
    from collections import Counter
    source_counts = Counter(doc['source_name'] for doc in normalized_docs)
    for source, count in source_counts.most_common():
        logger.info(f"  {source}: {count} documents")

    # Phase 3: Chunk documents
    logger.info("\nPhase 3: Chunking documents...")
    chunks = chunker.chunk_hierarchical(normalized_docs)

    if not chunks:
        logger.error("No chunks created! Exiting.")
        return 1

    # Phase 4: Save chunks
    logger.info("\nPhase 4: Saving chunks...")
    output_path = CHUNKS_DIR / "chunks.pkl"
    chunker.save_chunks(chunks, output_path)

    # Phase 5: Display summary
    logger.info("\n" + "=" * 60)
    logger.info("Ingestion Summary")
    logger.info("=" * 60)
    logger.info(f"Documents processed: {len(normalized_docs)}")
    logger.info(f"Chunks created: {len(chunks)}")

    # Chunk statistics
    token_counts = [c['token_count'] for c in chunks]
    logger.info(f"\nChunk token statistics:")
    logger.info(f"  Min: {min(token_counts)} tokens")
    logger.info(f"  Max: {max(token_counts)} tokens")
    logger.info(f"  Avg: {sum(token_counts) // len(token_counts)} tokens")

    # Sample chunks
    logger.info(f"\nSample chunks (first 3):")
    for i, chunk in enumerate(chunks[:3]):
        logger.info(f"\nChunk {i + 1}:")
        logger.info(f"  ID: {chunk['chunk_id']}")
        logger.info(f"  Source: {chunk['source_name']}")
        logger.info(f"  Tokens: {chunk['token_count']}")
        logger.info(f"  Preview: {chunk['text'][:100]}...")

    logger.info("\n" + "=" * 60)
    logger.info("Ingestion Complete!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
