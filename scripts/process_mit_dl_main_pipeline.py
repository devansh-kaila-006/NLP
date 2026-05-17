"""
Process MIT 6.S191 DL Main Video Lectures - COMPREHENSIVE VERSION
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


class MITDLMainPipelineProcessor(LoggerMixin):
    """Process MIT 6.S191 DL comprehensive lectures"""

    def __init__(self):
        self.chunker = VideoChunker()
        self.embedder = EmbeddingGenerator()
        self.faiss_manager = None

    def process_all_lectures(
        self,
        transcript_dir: str = "data/transcripts/dl",
        output_dir: str = "data/processed/video_chunks_mit_dl_main"
    ):
        """Process all MIT DL main SRT files in directory"""
        transcript_dir = Path(transcript_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all SRT files (lectures and tutorials)
        srt_files = sorted(transcript_dir.glob("MIT_DL_*.srt"))

        self.logger.info(f"Found {len(srt_files)} MIT DL main lecture transcripts")
        self.logger.info(f"**COMPREHENSIVE VERSION - 24 lectures + 1 tutorial**")

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
        self.logger.info(f"MIT DL MAIN PROCESSING COMPLETE")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total chunks: {len(all_chunks)}")
        self.logger.info(f"Total embeddings: {len(all_embeddings)}")

        # Convert chunks to RAG format
        rag_chunks = []
        for i, chunk in enumerate(all_chunks):
            # Determine if tutorial or lecture
            source_name = chunk.video_source
            is_tutorial = 'T01' in source_name

            rag_chunk = {
                'text': chunk.text,
                'source_name': source_name,
                'source_type': 'video',
                'chunk_id': chunk.chunk_id,
                'lecture_number': chunk.lecture_number,
                'timestamp_start': chunk.start_time,
                'timestamp_end': chunk.end_time,
                'video_url': chunk.video_url,
                'timestamp_url': chunk.timestamp_url,
                'duration_minutes': chunk.duration_minutes(),
                'is_tutorial': is_tutorial
            }
            rag_chunks.append(rag_chunk)

        # Save chunks
        chunks_path = output_dir / "mit_dl_main_video_chunks.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump(rag_chunks, f)
        self.logger.info(f"Saved chunks: {chunks_path}")

        # Save embeddings
        embeddings_array = np.array(all_embeddings)
        embeddings_path = output_dir / "mit_dl_main_video_embeddings.npy"
        np.save(embeddings_path, embeddings_array)
        self.logger.info(f"Saved embeddings: {embeddings_path}")

        # Create FAISS index
        self.logger.info(f"\nCreating FAISS index...")
        self.faiss_manager = FAISSManager()

        # Build index
        self.faiss_manager.create_index(embeddings_array, rag_chunks)
        self.logger.info(f"FAISS index created with {self.faiss_manager.get_index_info()['total_vectors']} chunks")

        # Save FAISS index
        index_path = output_dir / "mit_dl_main_video_index.faiss"
        metadata_path = output_dir / "mit_dl_main_video_metadata.pkl"
        self.faiss_manager.save_index(str(index_path), str(metadata_path))
        self.logger.info(f"Saved FAISS index: {index_path}")

        return {
            'chunks': rag_chunks,
            'embeddings': embeddings_array,
            'faiss_manager': self.faiss_manager
        }


def main():
    """Process all MIT DL main lectures"""
    processor = MITDLMainPipelineProcessor()

    print("="*80)
    print("MIT 6.S191 DL MAIN PIPELINE PROCESSOR")
    print("COMPREHENSIVE VERSION - 24 LECTURES + 1 TUTORIAL")
    print("="*80)

    result = processor.process_all_lectures()

    print(f"\n{'='*80}")
    print("MIT DL MAIN VIDEO PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total video chunks: {len(result['chunks'])}")
    print(f"Embeddings shape: {result['embeddings'].shape}")
    print(f"FAISS index ready: {result['faiss_manager'].get_index_info()}")

    # Count tutorials vs lectures
    tutorial_count = sum(1 for chunk in result['chunks'] if chunk.get('is_tutorial', False))
    lecture_count = len(result['chunks']) - tutorial_count

    print(f"\nContent breakdown:")
    print(f"  Lectures: {lecture_count} chunks")
    print(f"  Tutorials: {tutorial_count} chunks")

    # Sample chunks
    print(f"\nSample chunks (first 3):")
    for i, chunk in enumerate(result['chunks'][:3], 1):
        tutorial_marker = " [TUTORIAL]" if chunk.get('is_tutorial') else ""
        print(f"\n  {i}. {chunk['source_name']}{tutorial_marker} - {chunk['chunk_id']}")
        print(f"     Timestamp: {chunk['timestamp_start']/60:.1f}-{chunk['timestamp_end']/60:.1f}min")
        print(f"     URL: {chunk['timestamp_url']}")
        print(f"     Text: {chunk['text'][:100]}...")

    print(f"\n[SUCCESS] MIT DL main video pipeline ready for RAG integration!")
    print(f"[COMPREHENSIVE] This is the full 24-lecture MIT DL curriculum!")


if __name__ == "__main__":
    main()
