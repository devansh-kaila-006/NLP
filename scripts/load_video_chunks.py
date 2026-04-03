"""
Load and validate video chunks from Kaggle processing.

This script loads video chunks processed on Kaggle and prepares them
for integration into the RAG system.

Usage:
    python scripts/load_video_chunks.py

Input:
    config/data/chunks/ (video chunks from Kaggle)

Output:
    config/data/chunks/video_chunks.json (validated and formatted)
    config/data/chunks/chunk_loading_log.txt
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logging, get_logger
from src.utils.helpers import ensure_dir
from src.utils.exceptions import PDFProcessingError

# Setup logging
setup_logging(log_level="INFO", log_file="logs/chunk_loading.log")
logger = get_logger(__name__)


class VideoChunkLoader:
    """Load and validate video chunks from Kaggle."""

    def __init__(self, chunks_dir: str = None):
        """
        Initialize video chunk loader.

        Args:
            chunks_dir: Directory containing video chunks from Kaggle
        """
        if chunks_dir is None:
            project_root = Path(__file__).parent.parent
            chunks_dir = project_root / "config" / "data" / "chunks"

        self.chunks_dir = Path(chunks_dir)
        self.video_chunks = []

    def find_video_chunks(self) -> List[Path]:
        """
        Find all video chunk files.

        Returns:
            List of paths to chunk files
        """
        logger.info(f"Searching for video chunks in: {self.chunks_dir}")

        # Look for files matching video chunk patterns
        chunk_files = []

        # Pattern 1: {video_id}_chunks.json
        chunk_files.extend(self.chunks_dir.glob("*_chunks.json"))

        # Pattern 2: chunks/*.json
        chunks_subdir = self.chunks_dir / "chunks"
        if chunks_subdir.exists():
            chunk_files.extend(chunks_subdir.glob("*.json"))

        # Filter out non-video chunk files
        video_chunk_files = [
            f for f in chunk_files
            if "pdf" not in f.name.lower() and "video" not in f.name.lower()
        ]

        logger.info(f"Found {len(video_chunk_files)} video chunk files")
        return video_chunk_files

    def load_chunk_file(self, chunk_file: Path) -> Dict[str, Any]:
        """
        Load a single chunk file.

        Args:
            chunk_file: Path to chunk file

        Returns:
            Chunk data dictionary

        Raises:
            PDFProcessingError: If file cannot be loaded
        """
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise PDFProcessingError(
                f"Failed to load chunk file: {chunk_file}",
                details={"file": str(chunk_file), "error": str(e)}
            )

    def validate_chunk(self, chunk: Dict[str, Any]) -> bool:
        """
        Validate a single chunk.

        Args:
            chunk: Chunk dictionary

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["chunk_id", "start_time", "end_time", "transcript"]

        # Check required fields
        for field in required_fields:
            if field not in chunk:
                logger.warning(f"Chunk missing required field: {field}")
                return False

        # Validate time ranges
        if chunk["end_time"] <= chunk["start_time"]:
            logger.warning(f"Chunk has invalid time range: {chunk}")
            return False

        # Validate transcript
        if not chunk["transcript"] or len(chunk["transcript"].strip()) == 0:
            logger.warning(f"Chunk has empty transcript: {chunk['chunk_id']}")
            return False

        return True

    def normalize_chunk(self, chunk: Dict[str, Any], source_file: str) -> Dict[str, Any]:
        """
        Normalize chunk to RAG system format.

        Args:
            chunk: Original chunk dictionary
            source_file: Source file name

        Returns:
            Normalized chunk dictionary
        """
        # Extract video ID from source file
        video_id = chunk.get("video_id", source_file.replace("_chunks.json", ""))

        # Create normalized chunk
        normalized = {
            "chunk_id": f"{video_id}_{chunk['chunk_id']}",
            "text": chunk["transcript"],
            "source": video_id,
            "page": int(chunk["start_time"]),  # Use start_time as page equivalent
            "topic": chunk.get("topic", "video lecture"),
            "difficulty": "intermediate",
            "chunk_type": "video",
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "duration": chunk.get("duration", chunk["end_time"] - chunk["start_time"]),
            "has_diagram": chunk.get("has_diagram", False),
            "ocr_text": chunk.get("ocr_text", ""),
            "slide_number": chunk.get("slide_number"),
            "video_url": chunk.get("video_url", "")
        }

        return normalized

    def load_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Load and validate all video chunks.

        Returns:
            List of normalized video chunks
        """
        # Find all chunk files
        chunk_files = self.find_video_chunks()

        if not chunk_files:
            logger.warning("No video chunk files found")
            return []

        # Load and process chunks
        all_chunks = []
        valid_count = 0
        invalid_count = 0

        for chunk_file in chunk_files:
            logger.info(f"Processing: {chunk_file.name}")

            try:
                # Load chunk file
                data = self.load_chunk_file(chunk_file)

                # Extract chunks (handle different formats)
                chunks = []
                if "chunks" in data:
                    chunks = data["chunks"]
                elif isinstance(data, list):
                    chunks = data
                else:
                    logger.warning(f"Unknown chunk format in: {chunk_file}")
                    continue

                # Validate and normalize chunks
                for chunk in chunks:
                    if self.validate_chunk(chunk):
                        normalized = self.normalize_chunk(chunk, chunk_file.name)
                        all_chunks.append(normalized)
                        valid_count += 1
                    else:
                        invalid_count += 1

                logger.info(f"  Loaded {len(chunks)} chunks from {chunk_file.name}")

            except Exception as e:
                logger.error(f"Failed to process {chunk_file.name}: {e}")
                continue

        logger.info(f"\nChunk loading summary:")
        logger.info(f"  Total chunks: {len(all_chunks)}")
        logger.info(f"  Valid: {valid_count}")
        logger.info(f"  Invalid: {invalid_count}")

        self.video_chunks = all_chunks
        return all_chunks

    def save_chunks(self, output_path: str = None):
        """
        Save loaded chunks to JSON file.

        Args:
            output_path: Path to save chunks
        """
        if output_path is None:
            output_path = self.chunks_dir / "video_chunks.json"

        logger.info(f"Saving {len(self.video_chunks)} video chunks to: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.video_chunks, f, ensure_ascii=False, indent=2)

        logger.info("Video chunks saved successfully")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about loaded chunks.

        Returns:
            Statistics dictionary
        """
        if not self.video_chunks:
            return {}

        durations = [c["duration"] for c in self.video_chunks]
        sources = set(c["source"] for c in self.video_chunks)
        has_diagrams = sum(1 for c in self.video_chunks if c["has_diagram"])

        stats = {
            "total_chunks": len(self.video_chunks),
            "total_duration_hours": sum(durations) / 3600,
            "avg_chunk_duration_seconds": sum(durations) / len(durations),
            "unique_videos": len(sources),
            "chunks_with_diagrams": has_diagrams,
            "sources": list(sources)
        }

        return stats


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("Video Chunk Loader")
    logger.info("="*60)

    # Initialize loader
    loader = VideoChunkLoader()

    # Load all chunks
    chunks = loader.load_all_chunks()

    if not chunks:
        logger.warning("No video chunks loaded. Exiting.")
        return

    # Save chunks
    loader.save_chunks()

    # Print statistics
    stats = loader.get_statistics()
    logger.info("\n" + "="*60)
    logger.info("Video Chunk Statistics")
    logger.info("="*60)
    logger.info(f"Total chunks: {stats['total_chunks']}")
    logger.info(f"Total duration: {stats['total_duration_hours']:.1f} hours")
    logger.info(f"Avg chunk duration: {stats['avg_chunk_duration_seconds']:.1f} seconds")
    logger.info(f"Unique videos: {stats['unique_videos']}")
    logger.info(f"Chunks with diagrams: {stats['chunks_with_diagrams']}")
    logger.info(f"\nVideo sources: {', '.join(stats['sources'])}")

    logger.info("\n" + "="*60)
    logger.info("Video chunk loading complete!")
    logger.info("="*60)
    logger.info(f"\nVideo chunks saved to: config/data/chunks/video_chunks.json")
    logger.info("Next step: Run scripts/build_unified_index.py to create combined index")


if __name__ == "__main__":
    main()
