"""
Coherence Metrics for Multi-Modal RAG System

Implements temporal coherence and flow metrics for evaluating
the temporal consistency and logical progression of video chunks.
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from scipy.stats import kendalltau, spearmanr


class CoherenceMetrics:
    """
    Calculate temporal coherence and flow metrics for video RAG systems.

    Evaluates how well retrieved video chunks maintain temporal consistency
    and logical progression across timestamps.
    """

    @staticmethod
    def temporal_ordering_accuracy(retrieved_sequence: List[Dict],
                                 expected_sequence: List[Dict]) -> float:
        """
        Calculate how well retrieved chunks follow expected temporal order.

        Args:
            retrieved_sequence: List of retrieved chunks with timestamps
            expected_sequence: List of expected chunks in correct temporal order

        Returns:
            Temporal ordering accuracy (0.0 to 1.0)
        """
        if not retrieved_sequence or not expected_sequence:
            return 0.0

        # Extract timestamps from sequences
        retrieved_timestamps = [chunk.get('timestamp_start', 0) for chunk in retrieved_sequence]
        expected_timestamps = [chunk.get('timestamp_start', 0) for chunk in expected_sequence]

        # Calculate rank correlation using Kendall's Tau
        correlation, p_value = kendalltau(retrieved_timestamps, expected_timestamps)

        # Normalize correlation to 0-1 range (correlation is -1 to 1)
        normalized_correlation = (correlation + 1) / 2

        return max(0.0, normalized_correlation)

    @staticmethod
    def flow_score(chunks: List[Dict], embedding_model=None) -> float:
        """
        Calculate semantic flow score - average similarity between consecutive chunks.

        Args:
            chunks: List of chunks with text content
            embedding_model: Optional embedding model for semantic similarity

        Returns:
            Flow score (0.0 to 1.0)
        """
        if len(chunks) < 2:
            return 0.0

        flow_scores = []

        for i in range(len(chunks) - 1):
            current_text = chunks[i].get('text', '')
            next_text = chunks[i + 1].get('text', '')

            if not current_text or not next_text:
                continue

            # Simple word overlap similarity (can be enhanced with embeddings)
            current_words = set(current_text.lower().split())
            next_words = set(next_text.lower().split())

            if not current_words or not next_words:
                continue

            # Jaccard similarity
            intersection = current_words.intersection(next_words)
            union = current_words.union(next_words)

            similarity = len(intersection) / len(union) if union else 0.0
            flow_scores.append(similarity)

        if not flow_scores:
            return 0.0

        return np.mean(flow_scores)

    @staticmethod
    def coherence_precision(retrieved_chunks: List[Dict],
                           expected_order: List[str]) -> Dict:
        """
        Calculate coherence precision - percentage of chunks in correct relative order.

        Args:
            retrieved_chunks: List of retrieved chunks
            expected_order: List of chunk IDs in expected order

        Returns:
            Dictionary containing coherence metrics
        """
        if not retrieved_chunks or not expected_order:
            return {'coherence_precision': 0.0, 'correct_pairs': 0, 'total_pairs': 0}

        # Create position mapping for expected order
        expected_positions = {chunk_id: i for i, chunk_id in enumerate(expected_order)}

        # Calculate number of correctly ordered pairs
        correct_pairs = 0
        total_pairs = 0

        retrieved_ids = [chunk.get('chunk_id', chunk.get('id', '')) for chunk in retrieved_chunks]

        for i in range(len(retrieved_ids)):
            for j in range(i + 1, len(retrieved_ids)):
                chunk_i = retrieved_ids[i]
                chunk_j = retrieved_ids[j]

                if chunk_i in expected_positions and chunk_j in expected_positions:
                    total_pairs += 1

                    # Check if relative order is correct
                    if expected_positions[chunk_i] < expected_positions[chunk_j]:
                        correct_pairs += 1

        coherence_precision = correct_pairs / total_pairs if total_pairs > 0 else 0.0

        return {
            'coherence_precision': coherence_precision,
            'correct_pairs': correct_pairs,
            'total_pairs': total_pairs
        }

    @staticmethod
    def temporal_consistency(retrieved_chunks: List[Dict]) -> Dict:
        """
        Calculate temporal consistency metrics for retrieved chunks.

        Args:
            retrieved_chunks: List of retrieved chunks with timestamp information

        Returns:
            Dictionary containing temporal consistency metrics
        """
        if not retrieved_chunks:
            return {
                'temporal_consistency': 0.0,
                'timestamp_gaps': [],
                'mean_gap': 0.0,
                'std_gap': 0.0
            }

        # Sort chunks by timestamp
        sorted_chunks = sorted(retrieved_chunks,
                             key=lambda x: x.get('timestamp_start', 0))

        # Calculate gaps between consecutive chunks
        timestamp_gaps = []
        for i in range(len(sorted_chunks) - 1):
            current_end = sorted_chunks[i].get('timestamp_end', 0)
            next_start = sorted_chunks[i + 1].get('timestamp_start', 0)

            gap = next_start - current_end
            timestamp_gaps.append(max(0, gap))  # Non-negative gaps

        if not timestamp_gaps:
            return {
                'temporal_consistency': 1.0,
                'timestamp_gaps': [],
                'mean_gap': 0.0,
                'std_gap': 0.0
            }

        mean_gap = np.mean(timestamp_gaps)
        std_gap = np.std(timestamp_gaps)

        # Temporal consistency: inverse of gap variability
        # Lower variability = higher consistency
        temporal_consistency = 1.0 / (1.0 + std_gap) if std_gap > 0 else 1.0

        return {
            'temporal_consistency': temporal_consistency,
            'timestamp_gaps': timestamp_gaps,
            'mean_gap': float(mean_gap),
            'std_gap': float(std_gap)
        }

    @staticmethod
    def graph_coherence_analysis(chunks: List[Dict], temporal_graph=None) -> Dict:
        """
        Analyze coherence using temporal dependency graph structure.

        Args:
            chunks: List of retrieved chunks
            temporal_graph: Optional temporal dependency graph

        Returns:
            Dictionary containing graph coherence metrics
        """
        if not chunks:
            return {
                'graph_coherence': 0.0,
                'connected_components': 0,
                'average_path_length': 0.0
            }

        # If no graph provided, create simple adjacency based on temporal proximity
        if temporal_graph is None:
            chunk_ids = [chunk.get('chunk_id', chunk.get('id', f'chunk_{i}'))
                        for i, chunk in enumerate(chunks)]

            # Build simple adjacency based on sequence
            temporal_graph = defaultdict(set)
            for i in range(len(chunk_ids) - 1):
                temporal_graph[chunk_ids[i]].add(chunk_ids[i + 1])
                temporal_graph[chunk_ids[i + 1]].add(chunk_ids[i])

        # Calculate graph metrics
        num_components = 0
        visited = set()

        def dfs(node, component):
            if node in visited:
                return
            visited.add(node)
            component.add(node)
            for neighbor in temporal_graph.get(node, set()):
                dfs(neighbor, component)

        components = []
        for node in temporal_graph.keys():
            if node not in visited:
                component = set()
                dfs(node, component)
                components.append(component)

        num_components = len(components)

        # Calculate average path length within components
        path_lengths = []
        for component in components:
            component_list = list(component)
            if len(component_list) > 1:
                # Simple approximation: component size represents connectedness
                path_lengths.append(len(component_list))

        avg_path_length = np.mean(path_lengths) if path_lengths else 0.0

        # Graph coherence: prefer single connected component with good path length
        graph_coherence = 1.0 / num_components if num_components > 0 else 0.0

        return {
            'graph_coherence': graph_coherence,
            'connected_components': num_components,
            'average_path_length': float(avg_path_length)
        }

    @staticmethod
    def calculate_all_metrics(retrieved_sequences: List[Dict],
                             expected_sequences: List[Dict]) -> Dict:
        """
        Calculate comprehensive coherence metrics.

        Args:
            retrieved_sequences: List of retrieved chunk sequences
            expected_sequences: List of expected chunk sequences in correct order

        Returns:
            Dictionary containing all coherence metrics
        """
        if len(retrieved_sequences) != len(expected_sequences):
            raise ValueError("Retrieved and expected sequences must have same length")

        metrics = {
            'temporal_ordering_accuracy': [],
            'flow_scores': [],
            'coherence_precision': [],
            'temporal_consistency': [],
            'graph_coherence': []
        }

        for retrieved, expected in zip(retrieved_sequences, expected_sequences):
            # Temporal ordering accuracy
            ordering_acc = CoherenceMetrics.temporal_ordering_accuracy(retrieved, expected)
            metrics['temporal_ordering_accuracy'].append(ordering_acc)

            # Flow score
            flow = CoherenceMetrics.flow_score(retrieved)
            metrics['flow_scores'].append(flow)

            # Coherence precision
            expected_order = [chunk.get('chunk_id', chunk.get('id', ''))
                            for chunk in expected]
            coherence_prec = CoherenceMetrics.coherence_precision(retrieved, expected_order)
            metrics['coherence_precision'].append(coherence_prec['coherence_precision'])

            # Temporal consistency
            temporal_cons = CoherenceMetrics.temporal_consistency(retrieved)
            metrics['temporal_consistency'].append(temporal_cons['temporal_consistency'])

            # Graph coherence
            graph_coh = CoherenceMetrics.graph_coherence_analysis(retrieved)
            metrics['graph_coherence'].append(graph_coh['graph_coherence'])

        # Calculate summary statistics
        summary = {}
        for metric_name, values in metrics.items():
            summary[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values)
            }

        return {
            'individual_metrics': metrics,
            'summary': summary
        }