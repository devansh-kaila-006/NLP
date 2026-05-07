"""
Cross-Encoder Reranker - Improve retrieval precision
Re-ranks retrieved chunks using cross-encoder model
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

from src.utils.logger import LoggerMixin
from src.utils.helpers import Timer
from src.config import RERANKING_CONFIG


class CrossEncoderReranker(LoggerMixin):
    """
    Rerank retrieved chunks using cross-encoder for better precision
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize reranker

        Args:
            config: Reranking configuration
        """
        self.config = config or RERANKING_CONFIG
        self.model_name = self.config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.top_n = self.config.get("top_n", 3)
        self.batch_size = self.config.get("batch_size", 32)
        self.threshold = self.config.get("threshold", 0.5)

        self.model = None

    def load_model(self):
        """Load cross-encoder model"""
        if self.model is None:
            self.logger.info(f"Loading reranker model: {self.model_name}")
            with Timer("Model loading"):
                self.model = CrossEncoder(self.model_name)

            self.logger.info("Reranker model loaded")
        return self.model

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks based on query relevance

        Args:
            query: User query
            chunks: List of retrieved chunks
            top_n: Number of top chunks to return

        Returns:
            Re-ranked list of chunks
        """
        if not chunks:
            self.logger.warning("No chunks to rerank")
            return []

        top_n = top_n or self.top_n

        self.logger.info(f"Reranking {len(chunks)} chunks for query")
        self.load_model()

        # Prepare query-chunk pairs
        pairs = [[query, chunk["text"]] for chunk in chunks]

        # Score pairs
        with Timer("Reranking"):
            scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Sort by score
        scored_chunks = list(zip(scores, chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # Select top N
        reranked = []
        for score, chunk in scored_chunks[:top_n]:
            chunk_copy = chunk.copy()
            chunk_copy['rerank_score'] = float(score)
            reranked.append(chunk_copy)

        # Log top scores
        if reranked:
            scores = [f"{r['rerank_score']:.4f}" for r in reranked[:3]]
            self.logger.info(f"Top rerank scores: {scores}")

        return reranked

    def batch_rerank(
        self,
        queries: List[str],
        chunks_list: List[List[Dict[str, Any]]],
        top_n: int = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Rerank multiple query-chunk lists

        Args:
            queries: List of queries
            chunks_list: List of chunk lists (one per query)
            top_n: Number of top chunks per query

        Returns:
            List of re-ranked chunk lists
        """
        top_n = top_n or self.top_n

        self.logger.info(f"Batch reranking {len(queries)} queries")
        self.load_model()

        all_reranked = []

        for query, chunks in zip(queries, chunks_list):
            reranked = self.rerank(query, chunks, top_n=top_n)
            all_reranked.append(reranked)

        return all_reranked

    def filter_by_threshold(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter chunks by relevance threshold

        Args:
            chunks: List of chunks with rerank_score

        Returns:
            Filtered list of chunks
        """
        if not chunks:
            return []

        filtered = [
            chunk for chunk in chunks
            if chunk.get('rerank_score', 0) >= self.threshold
        ]

        self.logger.info(f"Filtered to {len(filtered)} chunks (threshold: {self.threshold})")
        return filtered

    def get_rerank_improvement(
        self,
        original_scores: List[float],
        rerank_scores: List[float]
    ) -> Dict[str, float]:
        """
        Calculate reranking improvement metrics

        Args:
            original_scores: Original relevance scores
            rerank_scores: Rerank scores

        Returns:
            Dictionary with improvement metrics
        """
        if not original_scores or not rerank_scores:
            return {}

        return {
            "original_max": max(original_scores),
            "rerank_max": max(rerank_scores),
            "original_avg": sum(original_scores) / len(original_scores),
            "rerank_avg": sum(rerank_scores) / len(rerank_scores),
            "improvement": (max(rerank_scores) - max(original_scores)) / max(original_scores) if max(original_scores) > 0 else 0
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get reranker model information

        Returns:
            Dictionary with model info
        """
        return {
            "model_name": self.model_name,
            "top_n": self.top_n,
            "batch_size": self.batch_size,
            "threshold": self.threshold
        }


if __name__ == "__main__":
    # Test reranker
    reranker = CrossEncoderReranker()

    # Dummy data
    query = "What is machine learning?"
    chunks = [
        {"text": "Machine learning is a subset of AI.", "chunk_id": "1"},
        {"text": "Deep learning uses neural networks.", "chunk_id": "2"},
        {"text": "Random forests are ensemble methods.", "chunk_id": "3"},
    ]

    # Rerank
    reranked = reranker.rerank(query, chunks, top_n=2)

    print(f"\nQuery: '{query}'")
    print(f"\nReranked results:")
    for i, chunk in enumerate(reranked):
        print(f"{i+1}. [{chunk['rerank_score']:.4f}] {chunk['text'][:60]}...")
