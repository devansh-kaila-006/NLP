"""
Build unified vector index from PDF and video chunks.

This script combines PDF chunks and video chunks into a single
vector index for the multi-modal RAG system.

Usage:
    python scripts/build_unified_index.py

Input:
    config/data/chunks/pdf_chunks.json
    config/data/chunks/video_chunks.json

Output:
    config/data/unified_index/ (FAISS index with all chunks)
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.base_rag.retriever import MultiModalRetriever
from src.utils.logger import setup_logging, get_logger
from src.utils.helpers import ensure_dir

# Setup logging
setup_logging(log_level="INFO", log_file="logs/unified_index.log")
logger = get_logger(__name__)


class UnifiedIndexBuilder:
    """Build unified index from PDF and video chunks."""

    def __init__(self):
        """Initialize unified index builder."""
        project_root = Path(__file__).parent.parent
        self.chunks_dir = project_root / "config" / "data" / "chunks"
        self.index_dir = project_root / "config" / "data" / "unified_index"

        # Ensure directories exist
        ensure_dir(str(self.chunks_dir))
        ensure_dir(str(self.index_dir))

    def load_pdf_chunks(self) -> list:
        """
        Load PDF chunks.

        Returns:
            List of PDF chunks
        """
        pdf_chunks_path = self.chunks_dir / "pdf_chunks.json"

        if not pdf_chunks_path.exists():
            logger.warning(f"PDF chunks not found: {pdf_chunks_path}")
            return []

        logger.info(f"Loading PDF chunks from: {pdf_chunks_path}")

        with open(pdf_chunks_path, 'r', encoding='utf-8') as f:
            pdf_chunks = json.load(f)

        logger.info(f"Loaded {len(pdf_chunks)} PDF chunks")
        return pdf_chunks

    def load_video_chunks(self) -> list:
        """
        Load video chunks.

        Returns:
            List of video chunks
        """
        video_chunks_path = self.chunks_dir / "video_chunks.json"

        if not video_chunks_path.exists():
            logger.warning(f"Video chunks not found: {video_chunks_path}")
            logger.warning("Run scripts/load_video_chunks.py first")
            return []

        logger.info(f"Loading video chunks from: {video_chunks_path}")

        with open(video_chunks_path, 'r', encoding='utf-8') as f:
            video_chunks = json.load(f)

        logger.info(f"Loaded {len(video_chunks)} video chunks")
        return video_chunks

    def combine_chunks(self, pdf_chunks: list, video_chunks: list) -> list:
        """
        Combine PDF and video chunks.

        Args:
            pdf_chunks: List of PDF chunks
            video_chunks: List of video chunks

        Returns:
            Combined list of chunks
        """
        logger.info("Combining PDF and video chunks")

        # Add source type if not present
        for chunk in pdf_chunks:
            if "chunk_type" not in chunk:
                chunk["chunk_type"] = "pdf"

        for chunk in video_chunks:
            if "chunk_type" not in chunk:
                chunk["chunk_type"] = "video"

        # Combine chunks
        all_chunks = pdf_chunks + video_chunks

        logger.info(f"Combined {len(pdf_chunks)} PDF + {len(video_chunks)} video = {len(all_chunks)} total chunks")

        return all_chunks

    def build_index(self, all_chunks: list) -> MultiModalRetriever:
        """
        Build unified vector index.

        Args:
            all_chunks: Combined list of chunks

        Returns:
            MultiModalRetriever instance
        """
        logger.info("="*60)
        logger.info("Building Unified Vector Index")
        logger.info("="*60)
        logger.info(f"Total chunks to index: {len(all_chunks)}")

        # Initialize retriever
        retriever = MultiModalRetriever(
            embedding_model="all-MiniLM-L6-v2",
            index_dir=None,  # Don't load existing index
            embedding_dimension=384
        )

        # Index chunks
        logger.info("Generating embeddings and building index...")
        retriever.index_chunks(all_chunks, save_index=False)

        # Save index
        logger.info(f"Saving index to: {self.index_dir}")
        retriever.save_index(str(self.index_dir))

        return retriever

    def print_statistics(self, retriever: MultiModalRetriever, all_chunks: list):
        """
        Print index statistics.

        Args:
            retriever: MultiModalRetriever instance
            all_chunks: List of all chunks
        """
        stats = retriever.get_index_stats()

        logger.info("\n" + "="*60)
        logger.info("Unified Index Statistics")
        logger.info("="*60)
        logger.info(f"Total chunks: {stats['total_chunks']}")
        logger.info(f"PDF chunks: {stats['pdf_chunks']}")
        logger.info(f"Video chunks: {stats['video_chunks']}")
        logger.info(f"Embedding dimension: {stats['embedding_dimension']}")

        # Chunk type breakdown
        pdf_count = sum(1 for c in all_chunks if c.get("chunk_type") == "pdf")
        video_count = sum(1 for c in all_chunks if c.get("chunk_type") == "video")

        logger.info(f"\nChunk type breakdown:")
        logger.info(f"  PDF: {pdf_count} ({pdf_count/len(all_chunks)*100:.1f}%)")
        logger.info(f"  Video: {video_count} ({video_count/len(all_chunks)*100:.1f}%)")

        # Video chunk statistics
        video_chunks = [c for c in all_chunks if c.get("chunk_type") == "video"]
        if video_chunks:
            durations = [c.get("duration", 0) for c in video_chunks]
            total_duration_hours = sum(durations) / 3600
            avg_duration = sum(durations) / len(durations)

            logger.info(f"\nVideo chunk statistics:")
            logger.info(f"  Total video duration: {total_duration_hours:.1f} hours")
            logger.info(f"  Avg chunk duration: {avg_duration:.1f} seconds")

            has_diagrams = sum(1 for c in video_chunks if c.get("has_diagram"))
            logger.info(f"  Chunks with diagrams: {has_diagrams}")

    def save_chunk_manifest(self, all_chunks: list):
        """
        Save manifest of all chunks.

        Args:
            all_chunks: List of all chunks
        """
        manifest_path = self.index_dir / "chunk_manifest.json"

        logger.info(f"Saving chunk manifest to: {manifest_path}")

        # Create manifest summary
        manifest = {
            "total_chunks": len(all_chunks),
            "pdf_chunks": sum(1 for c in all_chunks if c.get("chunk_type") == "pdf"),
            "video_chunks": sum(1 for c in all_chunks if c.get("chunk_type") == "video"),
            "sources": list(set(c.get("source", "unknown") for c in all_chunks)),
            "topics": list(set(c.get("topic", "unknown") for c in all_chunks)),
        }

        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        logger.info("Chunk manifest saved")

    def test_retrieval(self, retriever: MultiModalRetriever):
        """
        Test retrieval with sample queries.

        Args:
            retriever: MultiModalRetriever instance
        """
        logger.info("\n" + "="*60)
        logger.info("Testing Unified Retrieval")
        logger.info("="*60)

        test_queries = [
            "What is a neural network?",
            "Explain backpropagation",
            "What are convolutional neural networks?",
            "How does gradient descent work?"
        ]

        for query in test_queries:
            logger.info(f"\nQuery: {query}")

            try:
                results = retriever.retrieve(query, k=5)

                logger.info(f"Retrieved {len(results)} results:")

                for i, result in enumerate(results[:3], start=1):  # Show top 3
                    chunk_type = result.get('chunk_type', 'unknown')
                    source = result.get('source', 'Unknown')
                    score = result.get('score', 0)

                    if chunk_type == 'video':
                        start_time = result.get('start_time', 0)
                        logger.info(f"  {i}. [{chunk_type.upper()}] {source} @ {start_time:.0f}s (score: {score:.4f})")
                    else:
                        page = result.get('page', 'N/A')
                        logger.info(f"  {i}. [{chunk_type.upper()}] {source} p.{page} (score: {score:.4f})")

            except Exception as e:
                logger.error(f"Retrieval failed: {e}")


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("Unified Index Builder")
    logger.info("="*60)

    # Initialize builder
    builder = UnifiedIndexBuilder()

    # Load chunks
    pdf_chunks = builder.load_pdf_chunks()
    video_chunks = builder.load_video_chunks()

    if not pdf_chunks and not video_chunks:
        logger.error("No chunks found. Exiting.")
        logger.error("Please ensure:")
        logger.error("  1. PDF chunks: Run scripts/process_pdfs.py")
        logger.error("  2. Video chunks: Run scripts/load_video_chunks.py")
        return

    # Combine chunks
    all_chunks = builder.combine_chunks(pdf_chunks, video_chunks)

    # Build index
    retriever = builder.build_index(all_chunks)

    # Print statistics
    builder.print_statistics(retriever, all_chunks)

    # Save manifest
    builder.save_chunk_manifest(all_chunks)

    # Test retrieval
    builder.test_retrieval(retriever)

    logger.info("\n" + "="*60)
    logger.info("Unified index creation complete!")
    logger.info("="*60)
    logger.info(f"\nIndex saved to: {builder.index_dir}")
    logger.info("\nYou can now use the unified retriever:")
    logger.info("  from src.base_rag.retriever import MultiModalRetriever")
    logger.info(f"  retriever = MultiModalRetriever(index_dir='{builder.index_dir}')")
    logger.info("  results = retriever.retrieve('your query here', k=5)")


if __name__ == "__main__":
    main()
