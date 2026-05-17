"""
Retrieval Metrics for Multi-Modal RAG System

Implements standard information retrieval metrics for evaluating
retrieval effectiveness: Precision@K, Recall@K, MAP, NDCG, MRR.
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict


class RetrievalMetrics:
    """
    Calculate standard information retrieval metrics for RAG systems.

    Supports both binary relevance (relevant/non-relevant) and graded relevance
    (highly relevant, somewhat relevant, not relevant).
    """

    @staticmethod
    def precision_at_k(retrieved_items: List[str], relevant_items: Set[str], k: int) -> float:
        """
        Calculate Precision@K - fraction of retrieved items that are relevant.

        Args:
            retrieved_items: List of retrieved item IDs (ranked by relevance)
            relevant_items: Set of relevant item IDs
            k: Number of top items to consider

        Returns:
            Precision@K score (0.0 to 1.0)
        """
        if k <= 0 or not retrieved_items:
            return 0.0

        top_k_items = retrieved_items[:k]
        relevant_count = sum(1 for item in top_k_items if item in relevant_items)

        return relevant_count / k

    @staticmethod
    def recall_at_k(retrieved_items: List[str], relevant_items: Set[str], k: int) -> float:
        """
        Calculate Recall@K - fraction of relevant items that are retrieved in top K.

        Args:
            retrieved_items: List of retrieved item IDs (ranked by relevance)
            relevant_items: Set of relevant item IDs
            k: Number of top items to consider

        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if not relevant_items or k <= 0:
            return 0.0

        top_k_items = retrieved_items[:k]
        relevant_count = sum(1 for item in top_k_items if item in relevant_items)

        return relevant_count / len(relevant_items)

    @staticmethod
    def average_precision(retrieved_items: List[str], relevant_items: Set[str]) -> float:
        """
        Calculate Average Precision - average of precision scores at each relevant item.

        Args:
            retrieved_items: List of retrieved item IDs (ranked by relevance)
            relevant_items: Set of relevant item IDs

        Returns:
            Average Precision score (0.0 to 1.0)
        """
        if not relevant_items or not retrieved_items:
            return 0.0

        precision_scores = []
        relevant_found = 0

        for i, item in enumerate(retrieved_items, start=1):
            if item in relevant_items:
                relevant_found += 1
                precision_at_i = relevant_found / i
                precision_scores.append(precision_at_i)

        if not precision_scores:
            return 0.0

        return np.mean(precision_scores)

    @staticmethod
    def mean_average_precision(queries_results: List[Dict]) -> float:
        """
        Calculate Mean Average Precision (MAP) across multiple queries.

        Args:
            queries_results: List of dictionaries containing:
                - 'retrieved': List of retrieved item IDs
                - 'relevant': Set of relevant item IDs

        Returns:
            MAP score (0.0 to 1.0)
        """
        if not queries_results:
            return 0.0

        ap_scores = []
        for query_result in queries_results:
            retrieved = query_result.get('retrieved', [])
            relevant = set(query_result.get('relevant', []))

            ap = RetrievalMetrics.average_precision(retrieved, relevant)
            ap_scores.append(ap)

        return np.mean(ap_scores)

    @staticmethod
    def ndcg_at_k(retrieved_items: List[str], relevance_grades: Dict[str, int], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K (NDCG@K).

        Args:
            retrieved_items: List of retrieved item IDs (ranked by relevance)
            relevance_grades: Dictionary mapping item IDs to relevance grades (0-3)
                0: not relevant, 1: somewhat relevant, 2: relevant, 3: highly relevant
            k: Number of top items to consider

        Returns:
            NDCG@K score (0.0 to 1.0)
        """
        if k <= 0 or not retrieved_items:
            return 0.0

        # Calculate DCG@K
        dcg = 0.0
        for i, item in enumerate(retrieved_items[:k]):
            relevance = relevance_grades.get(item, 0)
            # DCG formula: relevance / log2(position + 1)
            dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0

        # Calculate ideal DCG@K (perfect ranking)
        ideal_grades = sorted(relevance_grades.values(), reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(ideal_grades):
            idcg += relevance / np.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def mean_ndcg(queries_results: List[Dict], k: int) -> float:
        """
        Calculate Mean NDCG@K across multiple queries.

        Args:
            queries_results: List of dictionaries containing:
                - 'retrieved': List of retrieved item IDs
                - 'relevance': Dictionary of item IDs to relevance grades
            k: Number of top items to consider

        Returns:
            Mean NDCG@K score (0.0 to 1.0)
        """
        if not queries_results:
            return 0.0

        ndcg_scores = []
        for query_result in queries_results:
            retrieved = query_result.get('retrieved', [])
            relevance = query_result.get('relevance', {})

            ndcg = RetrievalMetrics.ndcg_at_k(retrieved, relevance, k)
            ndcg_scores.append(ndcg)

        return np.mean(ndcg_scores)

    @staticmethod
    def mean_reciprocal_rank(queries_results: List[Dict]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR) - average of reciprocal of first relevant item.

        Args:
            queries_results: List of dictionaries containing:
                - 'retrieved': List of retrieved item IDs
                - 'relevant': Set of relevant item IDs

        Returns:
            MRR score (0.0 to 1.0)
        """
        if not queries_results:
            return 0.0

        reciprocal_ranks = []
        for query_result in queries_results:
            retrieved = query_result.get('retrieved', [])
            relevant = set(query_result.get('relevant', []))

            if not relevant:
                continue

            # Find position of first relevant item
            for i, item in enumerate(retrieved, start=1):
                if item in relevant:
                    reciprocal_ranks.append(1.0 / i)
                    break
            else:
                # No relevant items found
                reciprocal_ranks.append(0.0)

        if not reciprocal_ranks:
            return 0.0

        return np.mean(reciprocal_ranks)

    @staticmethod
    def calculate_all_metrics(queries_results: List[Dict], k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """
        Calculate comprehensive retrieval metrics across multiple queries.

        Args:
            queries_results: List of dictionaries containing query results
            k_values: List of K values for Precision@K and Recall@K

        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {
            'precision_at_k': {},
            'recall_at_k': {},
            'ndcg_at_k': {},
            'map': RetrievalMetrics.mean_average_precision(queries_results),
            'mrr': RetrievalMetrics.mean_reciprocal_rank(queries_results)
        }

        # Calculate Precision@K and Recall@K for each K
        for k in k_values:
            precision_scores = []
            recall_scores = []

            for query_result in queries_results:
                retrieved = query_result.get('retrieved', [])
                relevant = set(query_result.get('relevant', []))

                precision_scores.append(
                    RetrievalMetrics.precision_at_k(retrieved, relevant, k)
                )
                recall_scores.append(
                    RetrievalMetrics.recall_at_k(retrieved, relevant, k)
                )

            metrics['precision_at_k'][f'P@{k}'] = np.mean(precision_scores)
            metrics['recall_at_k'][f'R@{k}'] = np.mean(recall_scores)

        # Calculate NDCG@K for each K
        for k in k_values:
            ndcg_scores = []

            for query_result in queries_results:
                retrieved = query_result.get('retrieved', [])
                relevance = query_result.get('relevance', {})

                # Convert relevant set to relevance grades if needed
                if not relevance and 'relevant' in query_result:
                    relevant_set = set(query_result.get('relevant', []))
                    relevance = {item: 2 for item in retrieved if item in relevant_set}
                    relevance.update({item: 0 for item in retrieved if item not in relevant_set})

                ndcg_scores.append(
                    RetrievalMetrics.ndcg_at_k(retrieved, relevance, k)
                )

            metrics['ndcg_at_k'][f'NDCG@{k}'] = np.mean(ndcg_scores)

        return metrics