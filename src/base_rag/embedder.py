"""
Module: base_rag.embedder

Description:
    Embedding generation using sentence transformers

Inputs:
    - Text chunks

Outputs:
    - Embedding vectors

Dependencies:
    - sentence_transformers
    - numpy
    - typing
    - src.utils.logger
    - src.utils.exceptions

Usage:
    >>> from src.base_rag.embedder import Embedder
    >>> embedder = Embedder()
    >>> embeddings = embedder.embed_texts(["text1", "text2"])
"""

import pickle
from pathlib import Path
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.exceptions import EmbeddingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """
    Generate embeddings for text chunks using sentence transformers.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        """
        Initialize embedder.

        Args:
            model_name: Name of sentence transformer model
            device: Device to run model on (cpu or cuda)
        """
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """
        Lazy load the sentence transformer model.

        Returns:
            SentenceTransformer instance

        Raises:
            EmbeddingError: If model fails to load
        """
        if self._model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load embedding model: {str(e)}",
                    details={"model_name": self.model_name, "error": str(e)}
                )

        return self._model

    @property
    def embedding_dimension(self) -> int:
        """
        Get the embedding dimension of the model.

        Returns:
            Embedding dimension
        """
        if self._model is None:
            # Load model to get dimension
            _ = self.model

        return self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)

        Raises:
            EmbeddingError: If embedding generation fails

        Example:
            >>> embedder = Embedder()
            >>> embeddings = embedder.embed_texts(["hello world", "foo bar"])
            >>> embeddings.shape
            (2, 384)
        """
        if not texts:
            logger.warning("Empty text list provided for embedding")
            return np.array([])

        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100,
                normalize_embeddings=True  # L2 normalize for cosine similarity
            )

            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate embeddings: {str(e)}",
                details={"num_texts": len(texts), "error": str(e)}
            )

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text string

        Returns:
            Embedding vector with shape (embedding_dim,)

        Example:
            >>> embedder = Embedder()
            >>> embedding = embedder.embed_text("hello world")
            >>> embedding.shape
            (384,)
        """
        return self.embed_texts([text])[0]

    def save_embeddings(self, embeddings: np.ndarray, path: str) -> None:
        """
        Save embeddings to file.

        Args:
            embeddings: Embedding array
            path: Path to save embeddings

        Raises:
            EmbeddingError: If save fails

        Example:
            >>> embedder = Embedder()
            >>> embeddings = embedder.embed_texts(["text1", "text2"])
            >>> embedder.save_embeddings(embeddings, "data/embeddings.pkl")
        """
        try:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'wb') as f:
                pickle.dump(embeddings, f)

            logger.info(f"Saved embeddings to {path}")

        except Exception as e:
            raise EmbeddingError(
                f"Failed to save embeddings: {str(e)}",
                details={"path": path, "error": str(e)}
            )

    def load_embeddings(self, path: str) -> np.ndarray:
        """
        Load embeddings from file.

        Args:
            path: Path to embeddings file

        Returns:
            Embedding array

        Raises:
            EmbeddingError: If load fails

        Example:
            >>> embedder = Embedder()
            >>> embeddings = embedder.load_embeddings("data/embeddings.pkl")
        """
        try:
            with open(path, 'rb') as f:
                embeddings = pickle.load(f)

            logger.info(f"Loaded embeddings from {path}, shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            raise EmbeddingError(
                f"Failed to load embeddings: {str(e)}",
                details={"path": path, "error": str(e)}
            )

    def compute_similarity(
        self,
        embedding1: Union[np.ndarray, List[np.ndarray]],
        embedding2: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[float, np.ndarray]:
        """
        Compute cosine similarity between embeddings.

        Args:
            embedding1: First embedding or list of embeddings
            embedding2: Second embedding or list of embeddings

        Returns:
            Similarity score(s)

        Example:
            >>> embedder = Embedder()
            >>> emb1 = embedder.embed_text("hello")
            >>> emb2 = embedder.embed_text("hi")
            >>> similarity = embedder.compute_similarity(emb1, emb2)
        """
        # Ensure embeddings are numpy arrays
        if not isinstance(embedding1, np.ndarray):
            embedding1 = np.array(embedding1)
        if not isinstance(embedding2, np.ndarray):
            embedding2 = np.array(embedding2)

        # Compute dot product (embeddings are already normalized)
        return np.dot(embedding1, embedding2.T)
