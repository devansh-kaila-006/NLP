"""
Semantic Chunker for hierarchical text chunking
Supports chapter → section → paragraph hierarchy
"""

import re
from pathlib import Path
from typing import List, Dict, Any
import tiktoken

from src.utils.logger import LoggerMixin
from src.utils.helpers import save_pickle, Timer
from src.config import CHUNKING_CONFIG


class SemanticChunker(LoggerMixin):
    """
    Chunk documents hierarchically while preserving context
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize semantic chunker

        Args:
            config: Chunking configuration (uses CHUNKING_CONFIG if None)
        """
        self.config = config or CHUNKING_CONFIG
        self.chunk_size = self.config.get("chunk_size", 400)
        self.chunk_overlap = self.config.get("chunk_overlap", 50)
        self.min_chunk_size = self.config.get("min_chunk_size", 100)
        self.max_chunk_size = self.config.get("max_chunk_size", 800)

        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.logger.warning("tiktoken not available, using approximate token count")
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text

        Args:
            text: Text to count

        Returns:
            Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximate: 1 token ≈ 4 characters
            return len(text) // 4

    def split_by_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def split_by_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs

        Args:
            text: Input text

        Returns:
            List of paragraphs
        """
        # Split by double newlines or other paragraph markers
        paragraphs = re.split(r'\n\n+|\r\n\r\n+', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def create_chunks_with_overlap(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create chunks from text with overlap

        Args:
            text: Input text
            metadata: Metadata to attach to chunks

        Returns:
            List of chunks
        """
        chunks = []

        # Split into paragraphs first
        paragraphs = self.split_by_paragraphs(text)

        if not paragraphs:
            # Fall back to sentences
            sentences = self.split_by_sentences(text)
            paragraphs = [' '.join(sentences[i:i+5]) for i in range(0, len(sentences), 5)]

        # Build chunks
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0

        for i, paragraph in enumerate(paragraphs):
            paragraph_tokens = self.count_tokens(paragraph)

            # If single paragraph is too large, split it
            if paragraph_tokens > self.max_chunk_size:
                sentences = self.split_by_sentences(paragraph)
                for sentence in sentences:
                    sentence_tokens = self.count_tokens(sentence)
                    new_tokens = current_tokens + sentence_tokens

                    if new_tokens > self.chunk_size and current_chunk:
                        # Save current chunk
                        chunks.append(self._create_chunk(current_chunk, metadata, chunk_id))
                        chunk_id += 1

                        # Start new chunk with overlap
                        overlap_text = self._get_overlap_text(current_chunk)
                        current_chunk = overlap_text + " " + sentence
                        current_tokens = self.count_tokens(current_chunk)
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
                        current_tokens = new_tokens
            else:
                # Check if we need to start a new chunk
                new_tokens = current_tokens + paragraph_tokens

                if new_tokens > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append(self._create_chunk(current_chunk, metadata, chunk_id))
                    chunk_id += 1

                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + "\n\n" + paragraph
                    current_tokens = self.count_tokens(current_chunk)
                else:
                    # Add to current chunk
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                    current_tokens = new_tokens

        # Don't forget the last chunk
        if current_chunk and current_tokens >= self.min_chunk_size:
            chunks.append(self._create_chunk(current_chunk, metadata, chunk_id))

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """
        Get overlap text from end of chunk

        Args:
            text: Source text

        Returns:
            Overlap text
        """
        # Get last few sentences for overlap
        sentences = self.split_by_sentences(text)
        overlap_sentences = sentences[-2:] if len(sentences) > 2 else sentences

        overlap_tokens = sum(self.count_tokens(s) for s in overlap_sentences)

        # Trim if too long
        while overlap_tokens > self.chunk_overlap and overlap_sentences:
            overlap_sentences.pop(0)
            overlap_tokens = sum(self.count_tokens(s) for s in overlap_sentences)

        return ' '.join(overlap_sentences)

    def _create_chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_id: int
    ) -> Dict[str, Any]:
        """
        Create chunk dictionary with metadata

        Args:
            text: Chunk text
            metadata: Source metadata
            chunk_id: Chunk number

        Returns:
            Chunk dictionary
        """
        chunk = {
            'text': text.strip(),
            'chunk_id': f"{metadata.get('chunk_id', 'doc')}_{chunk_id:03d}",
            'char_count': len(text),
            'token_count': self.count_tokens(text),
        }

        # Copy metadata
        for key, value in metadata.items():
            if key not in ['text', 'chunk_id']:
                chunk[key] = value

        return chunk

    def chunk_hierarchical(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Chunk documents hierarchically

        Args:
            documents: List of documents with text

        Returns:
            List of chunks
        """
        self.logger.info(f"Chunking {len(documents)} documents hierarchically")
        all_chunks = []

        with Timer("Hierarchical chunking"):
            for i, doc in enumerate(documents):
                if i % 10 == 0:
                    self.logger.info(f"Chunking document {i + 1}/{len(documents)}")

                try:
                    chunks = self.create_chunks_with_overlap(doc['text'], doc)
                    all_chunks.extend(chunks)

                except Exception as e:
                    self.logger.warning(f"Failed to chunk document {i}: {e}")
                    continue

        self.logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")

        # Log statistics
        if all_chunks:
            token_counts = [c['token_count'] for c in all_chunks]
            self.logger.info(f"Chunk statistics:")
            self.logger.info(f"  Min tokens: {min(token_counts)}")
            self.logger.info(f"  Max tokens: {max(token_counts)}")
            self.logger.info(f"  Avg tokens: {sum(token_counts) // len(token_counts)}")

        return all_chunks

    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: str | Path) -> None:
        """
        Save chunks to disk

        Args:
            chunks: List of chunks
            output_path: Path to save chunks
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_pickle(chunks, output_path)
        self.logger.info(f"Saved {len(chunks)} chunks to {output_path}")

    def load_chunks(self, input_path: str | Path) -> List[Dict[str, Any]]:
        """
        Load chunks from disk

        Args:
            input_path: Path to chunks file

        Returns:
            List of chunks
        """
        from src.utils.helpers import load_pickle

        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {input_path}")

        chunks = load_pickle(input_path)
        self.logger.info(f"Loaded {len(chunks)} chunks from {input_path}")

        return chunks


def chunk_and_save(
    documents: List[Dict[str, Any]],
    output_path: str | Path,
    config: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to chunk documents and save

    Args:
        documents: List of documents
        output_path: Path to save chunks
        config: Chunking configuration

    Returns:
        List of chunks
    """
    chunker = SemanticChunker(config)
    chunks = chunker.chunk_hierarchical(documents)
    chunker.save_chunks(chunks, output_path)
    return chunks


if __name__ == "__main__":
    # Test chunker
    test_text = """
    This is the first paragraph. It contains some introductory text about machine learning.

    This is the second paragraph. It goes into more detail about supervised learning algorithms.

    This is the third paragraph. It discusses unsupervised learning and its applications.

    This is the fourth paragraph. It talks about reinforcement learning and how it works.

    This is the fifth paragraph. It concludes the introduction to machine learning concepts.
    """

    test_metadata = {
        'source_name': 'Test_Source',
        'source_type': 'pdf',
        'chapter': '1',
        'title': 'Introduction',
        'chunk_id': 'test_doc'
    }

    chunker = SemanticChunker()
    chunks = chunker.create_chunks_with_overlap(test_text, test_metadata)

    print(f"\nCreated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  Tokens: {chunk['token_count']}")
        print(f"  Text: {chunk['text'][:150]}...")
