"""
Retriever - Vector similarity search wrapper
Handles retrieval of relevant chunks using FAISS index
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vector_store.faiss_manager import FAISSManager
from src.utils.logger import LoggerMixin
from src.config import RETRIEVAL_CONFIG


class Retriever(LoggerMixin):
    """
    Retrieve relevant chunks using vector similarity search
    """

    def __init__(
        self,
        index_path: str | Path = None,
        chunks_path: str | Path = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize retriever

        Args:
            index_path: Path to FAISS index
            chunks_path: Path to chunks metadata
            config: Retrieval configuration
        """
        self.config = config or RETRIEVAL_CONFIG
        self.top_k = self.config.get("top_k", 5)
        self.min_relevance_score = self.config.get("min_relevance_score", 0.3)

        # Initialize components
        self.embedding_generator = EmbeddingGenerator()
        self.faiss_manager = FAISSManager()

        # Load index if paths provided
        if index_path and chunks_path:
            self.load_index(index_path, chunks_path)

    def load_index(
        self,
        index_path: str | Path,
        chunks_path: str | Path
    ) -> None:
        """
        Load FAISS index and chunks

        Args:
            index_path: Path to FAISS index
            chunks_path: Path to chunks metadata
        """
        self.logger.info(f"Loading index from {index_path}")
        self.faiss_manager.load_index(index_path, chunks_path)
        self.logger.info(f"Index loaded with {self.faiss_manager.index.ntotal} vectors")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query

        Args:
            query: User query text
            top_k: Number of chunks to retrieve
            return_scores: Include relevance scores

        Returns:
            List of relevant chunks with metadata
        """
        top_k = top_k or self.top_k

        self.logger.info(f"Retrieving top {top_k} chunks for query: '{query[:100]}...'")

        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embeddings(
            [query],
            show_progress=False
        )

        # Search index
        distances, indices = self.faiss_manager.search(query_embedding, top_k=top_k)

        # Retrieve chunks
        if return_scores:
            results = self.faiss_manager.retrieve_chunks(indices, distances)
        else:
            results = self.faiss_manager.retrieve_chunks(indices)

        # Filter by relevance score if requested
        if self.min_relevance_score > 0 and return_scores:
            results = [
                r for r in results
                if r.get('relevance_score', 0) >= self.min_relevance_score
            ]

        self.logger.info(f"Retrieved {len(results)} chunks")
        return results

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = None,
        return_scores: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve relevant chunks for multiple queries

        Args:
            queries: List of query strings
            top_k: Number of chunks per query
            return_scores: Include relevance scores

        Returns:
            List of result lists (one per query)
        """
        top_k = top_k or self.top_k

        self.logger.info(f"Batch retrieving for {len(queries)} queries")

        # Generate query embeddings
        query_embeddings = self.embedding_generator.generate_embeddings(
            queries,
            show_progress=False
        )

        # Batch search
        distances, indices = self.faiss_manager.batch_search(query_embeddings, top_k=top_k)

        # Retrieve chunks for each query
        all_results = []
        for i in range(len(queries)):
            if return_scores:
                results = self.faiss_manager.retrieve_chunks(
                    distances[i:i+1],
                    indices[i:i+1]
                )
            else:
                results = self.faiss_manager.retrieve_chunks(indices[i:i+1])

            all_results.append(results)

        return all_results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever statistics

        Returns:
            Dictionary with retriever info
        """
        index_info = self.faiss_manager.get_index_info()

        return {
            "index_loaded": index_info["loaded"],
            "total_vectors": index_info.get("total_vectors", 0),
            "top_k": self.top_k,
            "min_relevance_score": self.min_relevance_score,
            "embedding_model": self.embedding_generator.model_name
        }


if __name__ == "__main__":
    # Test retriever
    retriever = Retriever(
        index_path="data/processed/indices/vector_index.faiss",
        chunks_path="data/processed/indices/chunks_metadata.pkl"
    )

    # Test retrieval
    query = "What is gradient descent?"
    results = retriever.retrieve(query, top_k=3)

    print(f"\nQuery: '{query}'")
    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['source_name']} (score: {result['relevance_score']:.4f})")
        print(f"   {result['text'][:150]}...")
