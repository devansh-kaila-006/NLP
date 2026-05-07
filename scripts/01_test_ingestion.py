"""
Test Ingestion Script - Test PDF loading and chunking
Run this to verify the ingestion pipeline is working
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders.document_loader import DocumentLoader
from src.processors.semantic_chunker import SemanticChunker
from src.utils.logger import setup_logger
from src.utils.helpers import Timer
from src.config import CHUNKS_DIR


def test_pdf_sources_only():
    """Test with PDF sources only (faster, no downloads)"""
    logger = setup_logger("test_ingestion")

    logger.info("=" * 60)
    logger.info("Testing PDF Sources Only")
    logger.info("=" * 60)

    # Check if PDF files exist
    pdf_dir = Path("data/pdfs")
    ml_pdf = pdf_dir / "ML.pdf"
    dl_pdf = pdf_dir / "DL.pdf"

    available_sources = []
    if ml_pdf.exists():
        available_sources.append("ML_Course_Notes")
        logger.info(f"[OK] Found: {ml_pdf.name}")
    else:
        logger.warning(f"[MISSING] {ml_pdf.name}")

    if dl_pdf.exists():
        available_sources.append("DL_Textbook")
        logger.info(f"[OK] Found: {dl_pdf.name}")
    else:
        logger.warning(f"[MISSING] {dl_pdf.name}")

    if not available_sources:
        logger.error("\nNo PDF files found in data/pdfs/")
        logger.info("Please place your PDF files in the data/pdfs/ directory:")
        logger.info("  - ML.pdf (Stanford CS229 notes)")
        logger.info("  - DL.pdf (Deep Learning textbook)")
        return 1

    # Initialize components
    loader = DocumentLoader()
    chunker = SemanticChunker()

    # Load only PDF sources
    logger.info(f"\nLoading PDF sources: {available_sources}")
    with Timer("PDF loading"):
        documents = loader.load_all_sources(source_filter=available_sources)

    if not documents:
        logger.error("No documents loaded!")
        return 1

    logger.info(f"Loaded {len(documents)} documents")

    # Display document info
    logger.info("\nDocument Information:")
    for doc in documents[:3]:  # Show first 3
        logger.info(f"\n  Source: {doc['source_name']}")
        logger.info(f"  Title: {doc.get('title', 'N/A')}")
        logger.info(f"  Type: {doc['source_type']}")
        logger.info(f"  Length: {doc.get('char_count', 0)} chars")
        logger.info(f"  Preview: {doc.get('text', '')[:100]}...")

    # Normalize documents
    logger.info("\nNormalizing documents...")
    normalized = loader.normalize_documents(documents)
    logger.info(f"Normalized {len(normalized)} documents")

    # Chunk documents
    logger.info("\nChunking documents...")
    chunks = chunker.chunk_hierarchical(normalized)

    if not chunks:
        logger.error("No chunks created!")
        return 1

    # Save chunks
    logger.info("\nSaving chunks...")
    output_path = CHUNKS_DIR / "chunks.pkl"
    chunker.save_chunks(chunks, output_path)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    logger.info(f"Documents: {len(normalized)}")
    logger.info(f"Chunks: {len(chunks)}")

    # Sample chunks
    logger.info("\nSample chunks (first 2):")
    for i, chunk in enumerate(chunks[:2]):
        logger.info(f"\nChunk {i + 1}:")
        logger.info(f"  ID: {chunk['chunk_id']}")
        logger.info(f"  Source: {chunk['source_name']}")
        logger.info(f"  Type: {chunk['source_type']}")
        logger.info(f"  Tokens: {chunk['token_count']}")
        logger.info(f"  Preview: {chunk['text'][:150]}...")

    logger.info("\n" + "=" * 60)
    logger.info("[SUCCESS] Ingestion test passed!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit_code = test_pdf_sources_only()
    sys.exit(exit_code)
