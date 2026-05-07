"""
Interactive Query CLI - Test the RAG pipeline
Command-line interface for querying the system
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.rag_pipeline import RAGPipeline
from src.utils.logger import setup_logger


def main():
    """Main CLI application"""
    logger = setup_logger("query_cli")

    logger.info("=" * 60)
    logger.info("RAG Query CLI")
    logger.info("=" * 60)

    # Initialize pipeline
    try:
        pipeline = RAGPipeline(
            index_path="data/processed/indices/vector_index.faiss",
            chunks_path="data/processed/indices/chunks_metadata.pkl",
            use_reranker=True
        )

        logger.info("Pipeline initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        logger.error("Make sure you've run: python scripts/02_build_index.py")
        return 1

    # Show pipeline stats
    stats = pipeline.get_stats()
    logger.info("\nPipeline Statistics:")
    logger.info(f"  Index vectors: {stats['retriever']['total_vectors']:,}")
    logger.info(f"  Top K: {stats['retriever']['top_k']}")
    logger.info(f"  Reranker: {stats['reranker_enabled']}")
    logger.info(f"  Generator model: {stats['generator_model']}")

    # Run interactive mode
    pipeline.interactive_mode()

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
