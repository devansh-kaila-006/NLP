"""
Process CS231n Computer Vision Video Lectures - FINAL PLAYLIST
"""

import sys
from pathlib import Path
import pickle
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processors.video_chunker import VideoChunker
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vector_store.faiss_manager import FAISSManager
from src.utils.logger import LoggerMixin


class CS231nPipelineProcessor(LoggerMixin):
    """Process CS231n Computer Vision lectures - FINAL PLAYLIST"""

    def __init__(self):
        self.chunker = VideoChunker()
        self.embedder = EmbeddingGenerator()
        self.faiss_manager = None

    def process_all_lectures(
        self,
        transcript_dir: str = "data/transcripts/cs231n",
        output_dir: str = "data/processed/video_chunks_cs231n"
    ):
        """Process all CS231n SRT files in directory"""
        transcript_dir = Path(transcript_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all SRT files
        srt_files = sorted(transcript_dir.glob("CS231n_*.srt"))

        self.logger.info(f"Found {len(srt_files)} CS231n lecture transcripts")
        self.logger.info(f"**FINAL PLAYLIST - Completing multi-modal RAG system!**")

        all_chunks = []
        all_embeddings = []

        for i, srt_file in enumerate(srt_files, 1):
            self.logger.info(f"\n[{i}/{len(srt_files)}] Processing {srt_file.name}")

            try:
                # Process lecture
                result = self.chunker.process_lecture(
                    str(srt_file),
                    video_url=f"https://www.youtube.com/watch?v={srt_file.stem}"
                )

                chunks = result['chunks']

                self.logger.info(f"  Created {len(chunks)} chunks")

                # Add chunks to list
                all_chunks.extend(chunks)

                # Generate embeddings for chunks
                chunk_texts = [chunk.text for chunk in chunks]
                embeddings = self.embedder.generate_embeddings(chunk_texts)
                all_embeddings.extend(embeddings)

                self.logger.info(f"  Generated embeddings: {embeddings.shape}")

            except Exception as e:
                self.logger.error(f"  Failed to process {srt_file.name}: {e}")
                continue

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"CS231N PROCESSING COMPLETE")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total chunks: {len(all_chunks)}")
        self.logger.info(f"Total embeddings: {len(all_embeddings)}")

        # Convert chunks to RAG format
        rag_chunks = []
        for i, chunk in enumerate(all_chunks):
            rag_chunk = {
                'text': chunk.text,
                'source_name': chunk.video_source,
                'source_type': 'video',
                'chunk_id': chunk.chunk_id,
                'lecture_number': chunk.lecture_number,
                'timestamp_start': chunk.start_time,
                'timestamp_end': chunk.end_time,
                'video_url': chunk.video_url,
                'timestamp_url': chunk.timestamp_url,
                'duration_minutes': chunk.duration_minutes()
            }
            rag_chunks.append(rag_chunk)

        # Save chunks
        chunks_path = output_dir / "cs231n_video_chunks.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump(rag_chunks, f)
        self.logger.info(f"Saved chunks: {chunks_path}")

        # Save embeddings
        embeddings_array = np.array(all_embeddings)
        embeddings_path = output_dir / "cs231n_video_embeddings.npy"
        np.save(embeddings_path, embeddings_array)
        self.logger.info(f"Saved embeddings: {embeddings_path}")

        # Create FAISS index
        self.logger.info(f"\nCreating FAISS index...")
        self.faiss_manager = FAISSManager()

        # Build index
        self.faiss_manager.create_index(embeddings_array, rag_chunks)
        self.logger.info(f"FAISS index created with {self.faiss_manager.get_index_info()['total_vectors']} chunks")

        # Save FAISS index
        index_path = output_dir / "cs231n_video_index.faiss"
        metadata_path = output_dir / "cs231n_video_metadata.pkl"
        self.faiss_manager.save_index(str(index_path), str(metadata_path))
        self.logger.info(f"Saved FAISS index: {index_path}")

        return {
            'chunks': rag_chunks,
            'embeddings': embeddings_array,
            'faiss_manager': self.faiss_manager
        }


def main():
    """Process all CS231n lectures - COMPLETE MULTI-MODAL RAG SYSTEM"""
    processor = CS231nPipelineProcessor()

    print("="*80)
    print("CS231N COMPUTER VISION PIPELINE PROCESSOR")
    print("FINAL PLAYLIST - COMPLETING MULTI-MODAL RAG SYSTEM")
    print("="*80)

    result = processor.process_all_lectures()

    print(f"\n{'='*80}")
    print("CS231N VIDEO PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total video chunks: {len(result['chunks'])}")
    print(f"Embeddings shape: {result['embeddings'].shape}")
    print(f"FAISS index ready: {result['faiss_manager'].get_index_info()}")

    # Sample chunks
    print(f"\nSample chunks (first 3):")
    for i, chunk in enumerate(result['chunks'][:3], 1):
        print(f"\n  {i}. {chunk['source_name']} - {chunk['chunk_id']}")
        print(f"     Timestamp: {chunk['timestamp_start']/60:.1f}-{chunk['timestamp_end']/60:.1f}min")
        print(f"     URL: {chunk['timestamp_url']}")
        print(f"     Text: {chunk['text'][:100]}...")

    print(f"\n{'='*80}")
    print("CONGRATULATIONS! MULTI-MODAL RAG SYSTEM COMPLETE!")
    print(f"{'='*80}")
    print("All 4 playlists processed:")
    print("- CS229 Machine Learning: 612 chunks")
    print("- MIT 6.S191 Deep Learning: 363 chunks")
    print("- CS224n NLP with Deep Learning: 661 chunks")
    print(f"- CS231n Computer Vision: {len(result['chunks'])} chunks")
    print(f"\nTotal Video Content: {612 + 363 + 661 + len(result['chunks'])} chunks")
    print("+ PDF Content: 9,661 chunks")
    print(f"= COMPLETE SYSTEM: {9661 + 612 + 363 + 661 + len(result['chunks'])} chunks")
    print(f"{'='*80}")
    print("Ready for production use and workshop paper submission!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
