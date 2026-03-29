"""
Module: base_rag.pdf_processor

Description:
    PDF processing pipeline for text extraction and chunking

Inputs:
    - PDF file path

Outputs:
    - Chunks with metadata

Dependencies:
    - pypdf
    - typing
    - src.utils.logger
    - src.utils.exceptions

Usage:
    >>> from src.base_rag.pdf_processor import PDFProcessor
    >>> processor = PDFProcessor()
    >>> chunks = processor.process_pdf("path/to/file.pdf")
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pypdf import PdfReader

from src.utils.exceptions import PDFProcessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PDFProcessor:
    """
    Process PDF files for the RAG system.

    Handles text extraction, chunking, and metadata extraction from PDF files.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        min_chunk_size: int = 400
    ):
        """
        Initialize PDF processor.

        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            min_chunk_size: Minimum tokens per chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def process_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Process a PDF file and extract chunks.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of chunks with metadata

        Raises:
            PDFProcessingError: If PDF processing fails

        Example:
            >>> processor = PDFProcessor()
            >>> chunks = processor.process_pdf("cs229_notes.pdf")
            >>> len(chunks)
            45
        """
        pdf_file = Path(pdf_path)

        if not pdf_file.exists():
            raise PDFProcessingError(
                f"PDF file not found: {pdf_path}",
                details={"pdf_path": pdf_path}
            )

        logger.info(f"Processing PDF: {pdf_path}")

        try:
            # Extract text and metadata
            documents = self._extract_text(pdf_path)

            # Chunk documents
            chunks = self._chunk_documents(documents, pdf_path)

            logger.info(f"Created {len(chunks)} chunks from {pdf_path}")
            return chunks

        except Exception as e:
            raise PDFProcessingError(
                f"Failed to process PDF: {str(e)}",
                details={"pdf_path": pdf_path, "error": str(e)}
            )

    def _extract_text(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of documents with metadata
        """
        reader = PdfReader(pdf_path)
        documents = []

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()

                if text.strip():
                    documents.append({
                        "page": page_num,
                        "text": text,
                        "source": Path(pdf_path).stem
                    })
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num}: {e}")
                continue

        logger.info(f"Extracted text from {len(documents)} pages")
        return documents

    def _chunk_documents(
        self,
        documents: List[Dict[str, any]],
        pdf_path: str
    ) -> List[Dict[str, any]]:
        """
        Chunk documents into smaller pieces with overlap.

        Args:
            documents: List of documents from PDF
            pdf_path: Original PDF path for metadata

        Returns:
            List of chunks with metadata
        """
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0

        # Extract metadata from PDF
        metadata = self._extract_metadata(pdf_path)

        for doc in documents:
            text = doc["text"]
            sentences = self._split_sentences(text)

            for sentence in sentences:
                sentence_length = len(sentence.split())

                if current_length + sentence_length > self.chunk_size:
                    # Save current chunk if it meets minimum size
                    if current_length >= self.min_chunk_size:
                        chunk = self._create_chunk(
                            current_chunk,
                            doc,
                            chunk_id,
                            metadata
                        )
                        chunks.append(chunk)
                        chunk_id += 1

                        # Start new chunk with overlap
                        overlap_text = self._get_overlap_text(current_chunk)
                        current_chunk = [overlap_text] if overlap_text else []
                        current_length = len(overlap_text.split()) if overlap_text else 0

                current_chunk.append(sentence)
                current_length += sentence_length

        # Add final chunk
        if current_length >= self.min_chunk_size:
            chunk = self._create_chunk(
                current_chunk,
                doc,
                chunk_id,
                metadata
            )
            chunks.append(chunk)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting on common delimiters
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_text(self, chunk: List[str]) -> str:
        """
        Get overlap text from previous chunk.

        Args:
            chunk: Previous chunk sentences

        Returns:
            Overlap text
        """
        # Calculate how many sentences to keep for overlap
        # Rough approximation: keep last N sentences to get ~chunk_overlap tokens
        overlap_sentences = []
        overlap_length = 0

        for sentence in reversed(chunk):
            sentence_length = len(sentence.split())

            if overlap_length + sentence_length <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_length += sentence_length
            else:
                break

        return " ".join(overlap_sentences)

    def _create_chunk(
        self,
        sentences: List[str],
        doc: Dict[str, any],
        chunk_id: int,
        metadata: Dict[str, str]
    ) -> Dict[str, any]:
        """
        Create a chunk dictionary with metadata.

        Args:
            sentences: List of sentences in chunk
            doc: Source document
            chunk_id: Chunk identifier
            metadata: Extracted metadata

        Returns:
            Chunk dictionary
        """
        return {
            "chunk_id": chunk_id,
            "text": " ".join(sentences),
            "page": doc.get("page", 0),
            "source": doc.get("source", "unknown"),
            "topic": metadata.get("topic", "general"),
            "difficulty": metadata.get("difficulty", "intermediate"),
            "chunk_type": "pdf"
        }

    def _extract_metadata(self, pdf_path: str) -> Dict[str, str]:
        """
        Extract metadata from PDF filename/path.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Metadata dictionary
        """
        filename = Path(pdf_path).stem.lower()

        # Simple heuristic-based metadata extraction
        metadata = {
            "topic": "general",
            "difficulty": "intermediate"
        }

        # Detect topic from filename
        if "ml" in filename or "machine" in filename:
            metadata["topic"] = "machine learning"
        elif "dl" in filename or "deep" in filename:
            metadata["topic"] = "deep learning"
        elif "nlp" in filename:
            metadata["topic"] = "natural language processing"
        elif "cv" in filename or "vision" in filename:
            metadata["topic"] = "computer vision"
        elif "cnn" in filename or "convolutional" in filename:
            metadata["topic"] = "convolutional neural networks"

        # Detect difficulty from filename
        if "intro" in filename or "basic" in filename:
            metadata["difficulty"] = "beginner"
        elif "adv" in filename or "advanced" in filename:
            metadata["difficulty"] = "advanced"

        return metadata

    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, any]]:
        """
        Process multiple PDF files.

        Args:
            pdf_paths: List of PDF file paths

        Returns:
            Combined list of chunks from all PDFs

        Example:
            >>> processor = PDFProcessor()
            >>> chunks = processor.process_multiple_pdfs([
            ...     "cs229_notes.pdf",
            ...     "deep_learning_book.pdf"
            ... ])
        """
        all_chunks = []

        for pdf_path in pdf_paths:
            try:
                chunks = self.process_pdf(pdf_path)
                all_chunks.extend(chunks)
            except PDFProcessingError as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                continue

        logger.info(f"Processed {len(all_chunks)} chunks from {len(pdf_paths)} PDFs")
        return all_chunks
