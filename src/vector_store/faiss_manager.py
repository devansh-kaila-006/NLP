"""
FAISS Vector Store Manager - Build and manage FAISS indices
Handles vector similarity search operations
"""

import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pickle

from src.utils.logger import LoggerMixin
from src.utils.helpers import Timer
from src.config import FAISS_CONFIG


class FAISSManager(LoggerMixin):
    """
    Manage FAISS vector index for similarity search
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize FAISS manager

        Args:
            config: FAISS configuration (uses FAISS_CONFIG if None)
        """
        self.config = config or FAISS_CONFIG
        self.index_type = self.config.get("index_type", "IndexFlatIP")
        self.dimension = self.config.get("dimension", 384)
        self.normalize = self.config.get("normalize", True)

        self.index = None
        self.chunks = None

    def create_index(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict[str, Any]] = None
    ) -> faiss.Index:
        """
        Create FAISS index from embeddings

        Args:
            embeddings: numpy array of embeddings (N x dimension)
            chunks: Optional list of chunks to store with index

        Returns:
            FAISS index object
        """
        if len(embeddings) == 0:
            raise ValueError("No embeddings provided")

        # Verify dimensions match
        if embeddings.shape[1] != self.dimension:
            self.logger.warning(
                f"Embedding dimension {embeddings.shape[1]} doesn't match "
                f"configured dimension {self.dimension}. Updating config."
            )
            self.dimension = embeddings.shape[1]

        self.logger.info(f"Creating FAISS index: {self.index_type}")
        self.logger.info(f"  Dimension: {self.dimension}")
        self.logger.info(f"  Vectors: {len(embeddings)}")

        with Timer("Index creation"):
            # Create index based on type
            if self.index_type == "IndexFlatIP":
                # Inner product (cosine similarity if normalized)
                index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "IndexFlatL2":
                # L2 distance
                index = faiss.IndexFlatL2(self.dimension)
            else:
                self.logger.warning(f"Unknown index type {self.index_type}, using IndexFlatIP")
                index = faiss.IndexFlatIP(self.dimension)

            # Normalize embeddings if using inner product (for cosine similarity)
            if self.normalize and self.index_type == "IndexFlatIP":
                self.logger.info("Normalizing embeddings for cosine similarity")
                faiss.normalize_L2(embeddings)

            # Add embeddings to index
            index.add(embeddings.astype('float32'))

        self.index = index
        self.chunks = chunks

        self.logger.info(f"Index created with {index.ntotal} vectors")
        return index

    def save_index(
        self,
        index_path: str | Path,
        chunks_path: str | Path = None
    ) -> None:
        """
        Save index and chunks to disk

        Args:
            index_path: Path to save FAISS index
            chunks_path: Path to save chunks metadata
        """
        if self.index is None:
            raise ValueError("No index to save. Create an index first.")

        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        with Timer("Index saving"):
            faiss.write_index(self.index, str(index_path))

        index_size_mb = index_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"Saved index to {index_path}")
        self.logger.info(f"Index size: {index_size_mb:.2f}MB")
        self.logger.info(f"Vectors: {self.index.ntotal}")

        # Save chunks metadata
        if chunks_path and self.chunks:
            chunks_path = Path(chunks_path)
            with open(chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)

            chunks_size_mb = chunks_path.stat().st_size / (1024 * 1024)
            self.logger.info(f"Saved chunks to {chunks_path}")
            self.logger.info(f"Chunks size: {chunks_size_mb:.2f}MB")

    def load_index(
        self,
        index_path: str | Path,
        chunks_path: str | Path = None
    ) -> faiss.Index:
        """
        Load index and chunks from disk

        Args:
            index_path: Path to FAISS index file
            chunks_path: Path to chunks metadata file

        Returns:
            Loaded FAISS index
        """
        index_path = Path(index_path)

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        # Load FAISS index
        with Timer("Index loading"):
            self.index = faiss.read_index(str(index_path))

        self.logger.info(f"Loaded index from {index_path}")
        self.logger.info(f"Vectors: {self.index.ntotal}")
        self.logger.info(f"Dimension: {self.index.d}")

        # Load chunks metadata
        if chunks_path:
            chunks_path = Path(chunks_path)
            if chunks_path.exists():
                with open(chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                self.logger.info(f"Loaded {len(self.chunks)} chunks")
            else:
                self.logger.warning(f"Chunks file not found: {chunks_path}")

        return self.index

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search index for similar vectors

        Args:
            query_embedding: Query vector (1 x dimension)
            top_k: Number of results to return

        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None:
            raise ValueError("No index loaded. Create or load an index first.")

        if query_embedding.shape[0] != 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize query if using inner product
        if self.normalize and self.index_type == "IndexFlatIP":
            faiss.normalize_L2(query_embedding)

        # Search
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
        )

        return distances, indices

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch search for multiple queries

        Args:
            query_embeddings: Query vectors (N x dimension)
            top_k: Number of results to return per query

        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None:
            raise ValueError("No index loaded. Create or load an index first.")

        # Normalize queries if using inner product
        if self.normalize and self.index_type == "IndexFlatIP":
            faiss.normalize_L2(query_embeddings)

        # Search
        distances, indices = self.index.search(
            query_embeddings.astype('float32'),
            top_k
        )

        return distances, indices

    def retrieve_chunks(
        self,
        indices: np.ndarray,
        scores: np.ndarray = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks by indices

        Args:
            indices: Array of chunk indices
            scores: Optional relevance scores

        Returns:
            List of chunk dictionaries with scores
        """
        if self.chunks is None:
            raise ValueError("No chunks loaded. Load index with chunks first.")

        results = []
        for i, idx in enumerate(indices[0]):  # indices is 2D array
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                if scores is not None:
                    chunk['relevance_score'] = float(scores[0][i])
                results.append(chunk)

        return results

    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the index

        Returns:
            Dictionary with index information
        """
        if self.index is None:
            return {
                "loaded": False,
                "index_type": self.index_type,
                "dimension": self.dimension
            }

        return {
            "loaded": True,
            "index_type": self.index_type,
            "dimension": self.index.d,
            "total_vectors": self.index.ntotal,
            "normalize": self.normalize,
            "chunks_count": len(self.chunks) if self.chunks else 0
        }


def create_and_save_index(
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    index_path: str | Path,
    config: Dict[str, Any] = None
) -> faiss.Index:
    """
    Convenience function to create and save index

    Args:
        embeddings: numpy array of embeddings
        chunks: List of chunks with metadata
        index_path: Path to save index
        config: FAISS configuration

    Returns:
        FAISS index object
    """
    manager = FAISSManager(config)
    index = manager.create_index(embeddings, chunks)

    # Save index and chunks
    chunks_path = Path(str(index_path).replace('.faiss', '_chunks.pkl'))
    manager.save_index(index_path, chunks_path)

    return index


if __name__ == "__main__":
    # Test FAISS manager
    import numpy as np

    # Create dummy embeddings
    dummy_embeddings = np.random.rand(100, 384).astype('float32')
    dummy_chunks = [
        {"text": f"Chunk {i}", "chunk_id": f"chunk_{i}"}
        for i in range(100)
    ]

    # Create and save index
    manager = FAISSManager()
    index = manager.create_index(dummy_embeddings, dummy_chunks)

    print(f"\nIndex created:")
    print(f"  Vectors: {index.ntotal}")
    print(f"  Dimension: {index.d}")

    # Test search
    query = np.random.rand(1, 384).astype('float32')
    distances, indices = manager.search(query, top_k=5)

    print(f"\nSearch results:")
    print(f"  Distances: {distances[0]}")
    print(f"  Indices: {indices[0]}")

    # Test retrieval
    results = manager.retrieve_chunks(indices, distances)
    print(f"\nRetrieved {len(results)} chunks")
    for i, result in enumerate(results[:3]):
        print(f"  {i+1}. {result['chunk_id']} (score: {result['relevance_score']:.4f})")
