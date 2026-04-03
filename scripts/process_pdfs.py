"""
Process PDF files and create vector index.

This script processes PDF files from data/pdfs/, generates embeddings,
and creates a searchable vector index.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.base_rag.pdf_processor import PDFProcessor
from src.base_rag.retriever import MultiModalRetriever
from src.utils.logger import setup_logging, get_logger
from src.utils.helpers import ensure_dir

# Setup logging
setup_logging(log_level="INFO", log_file="logs/processing.log")
logger = get_logger(__name__)


def save_chunks_to_json(chunks, output_path):
    """
    Save chunks to JSON file for later use.

    Args:
        chunks: List of chunks with metadata
        output_path: Path to save JSON file
    """
    logger.info(f"Saving {len(chunks)} chunks to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Chunks saved successfully")


def process_pdfs():
    """
    Process all PDFs from data/pdfs directory.

    Returns:
        List of all processed chunks
    """
    logger.info("Starting PDF processing")

    # Get PDF directory (use config/data for PDFs)
    project_root = Path(__file__).parent.parent
    pdf_dir = project_root / "config" / "data" / "pdfs"

    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        return []

    # Find all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Initialize PDF processor
    processor = PDFProcessor(
        chunk_size=800,
        chunk_overlap=100,
        min_chunk_size=400
    )

    # Process all PDFs
    all_chunks = []

    for pdf_file in pdf_files:
        logger.info(f"Processing: {pdf_file.name}")

        try:
            chunks = processor.process_pdf(str(pdf_file))
            all_chunks.extend(chunks)
            logger.info(f"  Generated {len(chunks)} chunks from {pdf_file.name}")

        except Exception as e:
            logger.error(f"  Failed to process {pdf_file.name}: {e}")
            continue

    logger.info(f"Total chunks created: {len(all_chunks)}")

    # Save chunks to JSON
    chunks_dir = project_root / "config" / "data" / "chunks"
    ensure_dir(str(chunks_dir))
    chunks_path = chunks_dir / "pdf_chunks.json"
    save_chunks_to_json(all_chunks, str(chunks_path))

    return all_chunks


def create_index(chunks, index_dir=None):
    """
    Create vector index from chunks.

    Args:
        chunks: List of chunks with text and metadata
        index_dir: Directory to save index
    """
    logger.info("Creating vector index")

    # Use project root for index directory and chunks directory
    if index_dir is None:
        project_root = Path(__file__).parent.parent
        index_dir = project_root / "config" / "data" / "pdf_index"

    # Ensure index directory exists
    ensure_dir(str(index_dir))

    # Initialize retriever (don't load existing index)
    retriever = MultiModalRetriever(
        embedding_model="all-MiniLM-L6-v2",
        index_dir=None,  # Don't load existing index
        embedding_dimension=384
    )

    # Index chunks
    logger.info(f"Indexing {len(chunks)} chunks...")
    retriever.index_chunks(chunks, save_index=False)

    # Save index to specified directory
    logger.info(f"Saving index to {index_dir}")
    retriever.save_index(str(index_dir))

    # Get and display index statistics
    stats = retriever.get_index_stats()
    logger.info(f"Index statistics:")
    logger.info(f"  Total chunks: {stats['total_chunks']}")
    logger.info(f"  PDF chunks: {stats['pdf_chunks']}")
    logger.info(f"  Video chunks: {stats['video_chunks']}")
    logger.info(f"  Embedding dimension: {stats['embedding_dimension']}")

    return retriever


def test_retrieval(retriever, test_queries):
    """
    Test retrieval with sample queries.

    Args:
        retriever: MultiModalRetriever instance
        test_queries: List of test queries
    """
    logger.info("\n" + "="*50)
    logger.info("Testing retrieval with sample queries")
    logger.info("="*50)

    for query in test_queries:
        logger.info(f"\nQuery: {query}")

        try:
            results = retriever.retrieve(query, k=5)

            logger.info(f"Retrieved {len(results)} results:")

            for i, result in enumerate(results, start=1):
                logger.info(f"\n  Result {i}:")
                logger.info(f"    Score: {result.get('score', 0):.4f}")
                logger.info(f"    Source: {result.get('source', 'Unknown')}")
                logger.info(f"    Page: {result.get('page', 'N/A')}")
                logger.info(f"    Topic: {result.get('topic', 'N/A')}")
                logger.info(f"    Difficulty: {result.get('difficulty', 'N/A')}")
                logger.info(f"    Text: {result.get('text', '')[:150]}...")

        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")


def main():
    """
    Main processing function.
    """
    logger.info("="*50)
    logger.info("PDF Processing and Index Creation")
    logger.info("="*50)

    try:
        # Process PDFs
        chunks = process_pdfs()

        if not chunks:
            logger.error("No chunks were created. Exiting.")
            return

        # Create index
        retriever = create_index(chunks)

        # Test queries
        test_queries = [
            "What is a neural network?",
            "Explain backpropagation",
            "What is overfitting?",
            "How does gradient descent work?",
            "What are convolutional neural networks?",
            "Explain the concept of regularization"
        ]

        test_retrieval(retriever, test_queries)

        logger.info("\n" + "="*50)
        logger.info("Processing completed successfully!")
        logger.info("="*50)
        logger.info(f"\nIndex saved to: indices/pdf_index")
        logger.info(f"Total chunks indexed: {len(chunks)}")
        logger.info("\nYou can now use the retriever for queries:")
        logger.info("  from src.base_rag.retriever import MultiModalRetriever")
        logger.info("  retriever = MultiModalRetriever(index_dir='indices/pdf_index')")
        logger.info("  results = retriever.retrieve('your query here', k=5)")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
