"""
Embedding Generator - Create vector embeddings for chunks
Uses Sentence Transformers for semantic embeddings
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from src.utils.logger import LoggerMixin
from src.utils.helpers import save_pickle, load_pickle, Timer
from src.config import EMBEDDING_CONFIG


class EmbeddingGenerator(LoggerMixin):
    """
    Generate embeddings for text chunks using Sentence Transformers
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize embedding generator

        Args:
            config: Embedding configuration (uses EMBEDDING_CONFIG if None)
        """
        self.config = config or EMBEDDING_CONFIG
        self.model_name = self.config.get("model_name", "all-MiniLM-L6-v2")
        self.batch_size = self.config.get("batch_size", 32)
        self.device = self.config.get("device", "cpu")
        self.normalize = self.config.get("normalize", True)

        self.model = None
        self.embedding_dimension = None

    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            with Timer("Model loading"):
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device
                )

            # Get embedding dimension
            test_embedding = self.model.encode(["test"])
            self.embedding_dimension = test_embedding.shape[1]

            self.logger.info(f"Model loaded. Embedding dimension: {self.embedding_dimension}")
            self.logger.info(f"Device: {self.device}")

        return self.model

    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings
            show_progress: Show progress bar

        Returns:
            numpy array of embeddings (N x dimension)
        """
        if not texts:
            self.logger.warning("No texts provided for embedding")
            return np.array([])

        self.logger.info(f"Generating embeddings for {len(texts)} texts...")
        self.load_model()

        with Timer("Embedding generation"):
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize
            )

        self.logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def generate_embeddings_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings from chunk dictionaries

        Args:
            chunks: List of chunk dictionaries with 'text' field
            show_progress: Show progress bar

        Returns:
            numpy array of embeddings
        """
        if not chunks:
            self.logger.warning("No chunks provided for embedding")
            return np.array([])

        # Extract text from chunks
        texts = [chunk["text"] for chunk in chunks]

        self.logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.generate_embeddings(texts, show_progress)

        return embeddings

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        output_path: str | Path
    ) -> None:
        """
        Save embeddings to disk

        Args:
            embeddings: numpy array of embeddings
            output_path: Path to save embeddings (.npy format)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(output_path, embeddings)
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        self.logger.info(f"Saved embeddings to {output_path}")
        self.logger.info(f"File size: {file_size_mb:.2f}MB")
        self.logger.info(f"Shape: {embeddings.shape}")

    def load_embeddings(self, input_path: str | Path) -> np.ndarray:
        """
        Load embeddings from disk

        Args:
            input_path: Path to embeddings file (.npy format)

        Returns:
            numpy array of embeddings
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {input_path}")

        embeddings = np.load(input_path)
        self.logger.info(f"Loaded embeddings from {input_path}")
        self.logger.info(f"Shape: {embeddings.shape}")

        return embeddings

    def get_embedding_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model

        Returns:
            Dictionary with model information
        """
        self.load_model()

        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": self.device,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
            "max_sequence_length": self.model.max_seq_length
        }


def generate_and_save_embeddings(
    chunks: List[Dict[str, Any]],
    output_path: str | Path,
    config: Dict[str, Any] = None
) -> np.ndarray:
    """
    Convenience function to generate embeddings and save to disk

    Args:
        chunks: List of chunk dictionaries
        output_path: Path to save embeddings
        config: Embedding configuration

    Returns:
        numpy array of embeddings
    """
    generator = EmbeddingGenerator(config)
    embeddings = generator.generate_embeddings_from_chunks(chunks)
    generator.save_embeddings(embeddings, output_path)
    return embeddings


if __name__ == "__main__":
    # Test embedding generator
    test_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing deals with text and speech."
    ]

    generator = EmbeddingGenerator()
    embeddings = generator.generate_embeddings(test_texts)

    print(f"\nGenerated embeddings:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Type: {embeddings.dtype}")
    print(f"  First embedding (first 5 dims): {embeddings[0][:5]}")

    # Test model info
    info = generator.get_embedding_info()
    print(f"\nModel info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
