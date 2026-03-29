"""
Module: base_rag.retriever

Description:
    Multi-modal retrieval system for PDF and video content

Inputs:
    - Query string
    - Retrieval parameters

Outputs:
    - Retrieved chunks with metadata

Dependencies:
    - numpy
    - typing
    - src.base_rag.embedder
    - src.base_rag.vector_store
    - src.utils.logger
    - src.utils.exceptions

Usage:
    >>> from src.base_rag.retriever import MultiModalRetriever
    >>> retriever = MultiModalRetriever()
    >>> results = retriever.retrieve("What is a CNN?", k=5)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.base_rag.embedder import Embedder
from src.base_rag.vector_store import VectorStore
from src.utils.exceptions import RetrievalError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MultiModalRetriever:
    """
    Multi-modal retrieval system for PDF and video content.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_dir: Optional[str] = None,
        embedding_dimension: int = 384
    ):
        """
        Initialize retriever.

        Args:
            embedding_model: Name of sentence transformer model
            index_dir: Directory to load/save index
            embedding_dimension: Dimension of embeddings
        """
        self.embedding_model = embedding_model
        self.index_dir = index_dir
        self.embedding_dimension = embedding_dimension

        # Initialize components
        self.embedder = Embedder(model_name=embedding_model)
        self.vector_store = VectorStore(
            embedding_dimension=embedding_dimension,
            separate_indices=True
        )

        # Load existing index if directory provided
        if index_dir and Path(index_dir).exists():
            self.load_index(index_dir)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        chunk_type: Optional[str] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, any]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Query string
            k: Number of results to return
            chunk_type: Optional filter by chunk type (pdf, video)
            similarity_threshold: Minimum similarity score

        Returns:
            List of retrieved chunks with metadata and scores

        Raises:
            RetrievalError: If retrieval fails

        Example:
            >>> retriever = MultiModalRetriever()
            >>> results = retriever.retrieve("What is a CNN?", k=5)
            >>> len(results)
            5
        """
        try:
            logger.info(f"Retrieving for query: {query}")

            # Generate query embedding
            query_embedding = self.embedder.embed_text(query)

            # Search vector store
            scores, results = self.vector_store.search(
                query_embedding,
                k=k,
                chunk_type=chunk_type
            )

            # Filter by similarity threshold if provided
            if similarity_threshold is not None:
                results = [
                    r for r in results
                    if r.get("score", 0) >= similarity_threshold
                ]

            logger.info(f"Retrieved {len(results)} chunks")
            return results

        except Exception as e:
            raise RetrievalError(
                f"Retrieval failed: {str(e)}",
                details={"query": query, "k": k, "error": str(e)}
            )

    def retrieve_with_filtering(
        self,
        query: str,
        k: int = 5,
        source: Optional[str] = None,
        topic: Optional[str] = None,
        difficulty: Optional[str] = None,
        chunk_type: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Retrieve with metadata filtering.

        Args:
            query: Query string
            k: Number of results to return
            source: Filter by source name
            topic: Filter by topic
            difficulty: Filter by difficulty level
            chunk_type: Filter by chunk type (pdf, video)

        Returns:
            List of retrieved chunks with metadata

        Example:
            >>> retriever = MultiModalRetriever()
            >>> results = retriever.retrieve_with_filtering(
            ...     "neural networks",
            ...     k=5,
            ...     difficulty="beginner"
            ... )
        """
        # Retrieve more results than needed to account for filtering
        retrieval_k = k * 3
        results = self.retrieve(query, k=retrieval_k, chunk_type=chunk_type)

        # Apply filters
        filtered_results = []

        for result in results:
            # Check source filter
            if source and result.get("source") != source:
                continue

            # Check topic filter
            if topic and result.get("topic") != topic:
                continue

            # Check difficulty filter
            if difficulty and result.get("difficulty") != difficulty:
                continue

            filtered_results.append(result)

            # Return only k results
            if len(filtered_results) >= k:
                break

        logger.info(f"Filtered to {len(filtered_results)} results")
        return filtered_results

    def index_chunks(
        self,
        chunks: List[Dict[str, any]],
        save_index: bool = True
    ) -> None:
        """
        Index chunks for retrieval.

        Args:
            chunks: List of chunks with text and metadata
            save_index: Whether to save index after adding

        Raises:
            RetrievalError: If indexing fails

        Example:
            >>> retriever = MultiModalRetriever()
            >>> chunks = [
            ...     {"text": "Neural networks are...", "source": "pdf1"},
            ...     {"text": "CNNs are used for...", "source": "video1"}
            ... ]
            >>> retriever.index_chunks(chunks)
        """
        try:
            if not chunks:
                logger.warning("No chunks provided for indexing")
                return

            logger.info(f"Indexing {len(chunks)} chunks")

            # Extract texts
            texts = [chunk["text"] for chunk in chunks]

            # Generate embeddings
            embeddings = self.embedder.embed_texts(texts)

            # Add to vector store
            self.vector_store.add_embeddings(embeddings, chunks)

            # Save index if requested
            if save_index and self.index_dir:
                self.save_index(self.index_dir)

            logger.info("Indexing complete")

        except Exception as e:
            raise RetrievalError(
                f"Indexing failed: {str(e)}",
                details={"num_chunks": len(chunks), "error": str(e)}
            )

    def save_index(self, directory: str) -> None:
        """
        Save index to directory.

        Args:
            directory: Directory to save index

        Raises:
            RetrievalError: If save fails

        Example:
            >>> retriever = MultiModalRetriever()
            >>> retriever.save_index("indices/my_index")
        """
        try:
            self.vector_store.save(directory)
            logger.info(f"Index saved to {directory}")
        except Exception as e:
            raise RetrievalError(
                f"Failed to save index: {str(e)}",
                details={"directory": directory, "error": str(e)}
            )

    def load_index(self, directory: str) -> None:
        """
        Load index from directory.

        Args:
            directory: Directory containing index

        Raises:
            RetrievalError: If load fails

        Example:
            >>> retriever = MultiModalRetriever()
            >>> retriever.load_index("indices/my_index")
        """
        try:
            self.vector_store.load(directory)
            logger.info(f"Index loaded from {directory}")
        except Exception as e:
            raise RetrievalError(
                f"Failed to load index: {str(e)}",
                details={"directory": directory, "error": str(e)}
            )

    def get_index_stats(self) -> Dict[str, any]:
        """
        Get statistics about the index.

        Returns:
            Dictionary with index statistics

        Example:
            >>> retriever = MultiModalRetriever()
            >>> stats = retriever.get_index_stats()
            >>> print(stats["total_chunks"])
        """
        stats = {
            "total_chunks": len(self.vector_store),
            "pdf_chunks": len(self.vector_store._pdf_metadata),
            "video_chunks": len(self.vector_store._video_metadata),
            "embedding_dimension": self.embedding_dimension,
            "embedding_model": self.embedding_model
        }

        return stats

    def batch_retrieve(
        self,
        queries: List[str],
        k: int = 5
    ) -> List[List[Dict[str, any]]]:
        """
        Retrieve for multiple queries.

        Args:
            queries: List of query strings
            k: Number of results per query

        Returns:
            List of result lists

        Example:
            >>> retriever = MultiModalRetriever()
            >>> queries = ["What is a CNN?", "Explain backpropagation"]
            >>> results = retriever.batch_retrieve(queries, k=3)
        """
        results = []

        for query in queries:
            try:
                query_results = self.retrieve(query, k=k)
                results.append(query_results)
            except RetrievalError as e:
                logger.error(f"Failed to retrieve for query '{query}': {e}")
                results.append([])

        return results
