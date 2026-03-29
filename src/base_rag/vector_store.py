"""
Module: base_rag.vector_store

Description:
    Vector store implementation using FAISS

Inputs:
    - Embeddings
    - Metadata

Outputs:
    - FAISS index with metadata storage

Dependencies:
    - faiss
    - numpy
    - pickle
    - pathlib.Path
    - typing
    - src.utils.logger
    - src.utils.exceptions

Usage:
    >>> from src.base_rag.vector_store import VectorStore
    >>> store = VectorStore(embedding_dimension=384)
    >>> store.add_embeddings(embeddings, metadata)
    >>> results = store.search(query_embedding, k=5)
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from src.utils.exceptions import VectorStoreError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    FAISS-based vector store for similarity search.
    """

    def __init__(
        self,
        embedding_dimension: int = 384,
        index_type: str = "IndexFlatIP",
        separate_indices: bool = False
    ):
        """
        Initialize vector store.

        Args:
            embedding_dimension: Dimension of embeddings
            index_type: Type of FAISS index (IndexFlatIP for cosine similarity)
            separate_indices: Whether to use separate indices for different modalities
        """
        self.embedding_dimension = embedding_dimension
        self.index_type = index_type
        self.separate_indices = separate_indices

        # Main index
        self._index: Optional[faiss.Index] = None

        # Metadata storage
        self._metadata: List[Dict[str, any]] = []

        # Separate indices if enabled
        self._pdf_index: Optional[faiss.Index] = None
        self._video_index: Optional[faiss.Index] = None
        self._pdf_metadata: List[Dict[str, any]] = []
        self._video_metadata: List[Dict[str, any]] = []

        self._initialize_index()

    def _initialize_index(self) -> None:
        """
        Initialize FAISS index.

        Returns:
            None
        """
        try:
            if self.index_type == "IndexFlatIP":
                # Inner product index for cosine similarity (requires normalized vectors)
                self._index = faiss.IndexFlatIP(self.embedding_dimension)
            else:
                # L2 distance index
                self._index = faiss.IndexFlatL2(self.embedding_dimension)

            if self.separate_indices:
                self._pdf_index = faiss.IndexFlatIP(self.embedding_dimension)
                self._video_index = faiss.IndexFlatIP(self.embedding_dimension)

            logger.info(f"Initialized FAISS index: {self.index_type}")

        except Exception as e:
            raise VectorStoreError(
                f"Failed to initialize FAISS index: {str(e)}",
                details={"index_type": self.index_type, "error": str(e)}
            )

    @property
    def index(self) -> faiss.Index:
        """
        Get the main FAISS index.

        Returns:
            FAISS index
        """
        if self._index is None:
            self._initialize_index()
        return self._index

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, any]]
    ) -> None:
        """
        Add embeddings to the index.

        Args:
            embeddings: Embedding array with shape (n, embedding_dim)
            metadata: List of metadata dictionaries

        Raises:
            VectorStoreError: If add operation fails

        Example:
            >>> store = VectorStore(embedding_dimension=384)
            >>> embeddings = np.random.rand(10, 384)
            >>> metadata = [{"source": "pdf1"} for _ in range(10)]
            >>> store.add_embeddings(embeddings, metadata)
        """
        try:
            # Ensure embeddings are numpy array
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)

            # Add to main index
            self.index.add(embeddings.astype('float32'))

            # Store metadata
            self._metadata.extend(metadata)

            # Add to separate indices if enabled
            if self.separate_indices:
                self._add_to_separate_indices(embeddings, metadata)

            logger.info(f"Added {len(embeddings)} embeddings to index")

        except Exception as e:
            raise VectorStoreError(
                f"Failed to add embeddings: {str(e)}",
                details={"num_embeddings": len(embeddings), "error": str(e)}
            )

    def _add_to_separate_indices(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, any]]
    ) -> None:
        """
        Add embeddings to separate modality-specific indices.

        Args:
            embeddings: Embedding array
            metadata: List of metadata dictionaries

        Returns:
            None
        """
        pdf_embeddings = []
        pdf_metadata = []
        video_embeddings = []
        video_metadata = []

        for emb, meta in zip(embeddings, metadata):
            chunk_type = meta.get("chunk_type", "unknown")

            if chunk_type == "pdf":
                pdf_embeddings.append(emb)
                pdf_metadata.append(meta)
            elif chunk_type == "video":
                video_embeddings.append(emb)
                video_metadata.append(meta)

        # Add to PDF index
        if pdf_embeddings:
            self._pdf_index.add(np.array(pdf_embeddings).astype('float32'))
            self._pdf_metadata.extend(pdf_metadata)

        # Add to video index
        if video_embeddings:
            self._video_index.add(np.array(video_embeddings).astype('float32'))
            self._video_metadata.extend(video_metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        chunk_type: Optional[str] = None
    ) -> Tuple[np.ndarray, List[Dict[str, any]]]:
        """
        Search for similar embeddings.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            chunk_type: Optional filter by chunk type (pdf, video)

        Returns:
            Tuple of (distances/scores, metadata)

        Raises:
            VectorStoreError: If search fails

        Example:
            >>> store = VectorStore(embedding_dimension=384)
            >>> query = np.random.rand(384)
            >>> scores, results = store.search(query, k=5)
        """
        try:
            # Ensure query is numpy array and normalize
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)

            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            faiss.normalize_L2(query_embedding)

            # Search appropriate index
            if self.separate_indices and chunk_type:
                if chunk_type == "pdf":
                    index = self._pdf_index
                    metadata = self._pdf_metadata
                elif chunk_type == "video":
                    index = self._video_index
                    metadata = self._video_metadata
                else:
                    index = self.index
                    metadata = self._metadata
            else:
                index = self.index
                metadata = self._metadata

            # Perform search
            scores, indices = index.search(query_embedding.astype('float32'), k)

            # Retrieve metadata for results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(metadata) and idx >= 0:
                    result_metadata = metadata[idx].copy()
                    result_metadata["score"] = float(score)
                    results.append(result_metadata)

            return scores[0], results

        except Exception as e:
            raise VectorStoreError(
                f"Search failed: {str(e)}",
                details={"k": k, "error": str(e)}
            )

    def save(self, directory: str) -> None:
        """
        Save index and metadata to directory.

        Args:
            directory: Directory to save index

        Raises:
            VectorStoreError: If save fails

        Example:
            >>> store = VectorStore(embedding_dimension=384)
            >>> store.add_embeddings(embeddings, metadata)
            >>> store.save("indices/pdf_index")
        """
        try:
            output_dir = Path(directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save main index
            index_path = output_dir / "index.faiss"
            faiss.write_index(self.index, str(index_path))

            # Save metadata
            metadata_path = output_dir / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(self._metadata, f)

            # Save separate indices if enabled
            if self.separate_indices:
                faiss.write_index(self._pdf_index, str(output_dir / "pdf_index.faiss"))
                faiss.write_index(self._video_index, str(output_dir / "video_index.faiss"))

                with open(output_dir / "pdf_metadata.pkl", 'wb') as f:
                    pickle.dump(self._pdf_metadata, f)
                with open(output_dir / "video_metadata.pkl", 'wb') as f:
                    pickle.dump(self._video_metadata, f)

            logger.info(f"Saved index to {directory}")

        except Exception as e:
            raise VectorStoreError(
                f"Failed to save index: {str(e)}",
                details={"directory": directory, "error": str(e)}
            )

    def load(self, directory: str) -> None:
        """
        Load index and metadata from directory.

        Args:
            directory: Directory containing index files

        Raises:
            VectorStoreError: If load fails

        Example:
            >>> store = VectorStore(embedding_dimension=384)
            >>> store.load("indices/pdf_index")
        """
        try:
            input_dir = Path(directory)

            # Load main index
            index_path = input_dir / "index.faiss"
            if not index_path.exists():
                raise VectorStoreError(
                    f"Index file not found: {index_path}",
                    details={"directory": directory}
                )

            self._index = faiss.read_index(str(index_path))

            # Load metadata
            metadata_path = input_dir / "metadata.pkl"
            with open(metadata_path, 'rb') as f:
                self._metadata = pickle.load(f)

            # Load separate indices if they exist
            if self.separate_indices:
                pdf_index_path = input_dir / "pdf_index.faiss"
                video_index_path = input_dir / "video_index.faiss"

                if pdf_index_path.exists():
                    self._pdf_index = faiss.read_index(str(pdf_index_path))
                    with open(input_dir / "pdf_metadata.pkl", 'rb') as f:
                        self._pdf_metadata = pickle.load(f)

                if video_index_path.exists():
                    self._video_index = faiss.read_index(str(video_index_path))
                    with open(input_dir / "video_metadata.pkl", 'rb') as f:
                        self._video_metadata = pickle.load(f)

            logger.info(f"Loaded index from {directory}")

        except Exception as e:
            raise VectorStoreError(
                f"Failed to load index: {str(e)}",
                details={"directory": directory, "error": str(e)}
            )

    def __len__(self) -> int:
        """
        Get number of embeddings in the index.

        Returns:
            Number of embeddings
        """
        return self.index.ntotal
