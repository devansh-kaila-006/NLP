"""
Custom Multi-Modal RAG Metrics
Specialized metrics for evaluating multi-modal RAG systems with video, PDF, and web content
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MultiModalMetricsResult:
    """Container for multi-modal metrics results"""
    cross_modal_consistency: float
    temporal_coherence_quality: float
    multimodal_context_utilization: float
    source_diversity: float
    citation_accuracy: float

    # Additional metadata
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "cross_modal_consistency": self.cross_modal_consistency,
            "temporal_coherence_quality": self.temporal_coherence_quality,
            "multimodal_context_utilization": self.multimodal_context_utilization,
            "source_diversity": self.source_diversity,
            "citation_accuracy": self.citation_accuracy,
            "details": self.details
        }

    def get_average_score(self) -> float:
        """Calculate average multi-modal metric score"""
        scores = [
            self.cross_modal_consistency,
            self.temporal_coherence_quality,
            self.multimodal_context_utilization,
            self.source_diversity,
            self.citation_accuracy
        ]
        return sum(scores) / len(scores)


class MultiModalRAGEvaluator:
    """
    Evaluates multi-modal RAG systems with specialized metrics
    """

    def __init__(self):
        """Initialize multi-modal RAG evaluator"""
        logger.info("Multi-Modal RAG Evaluator initialized")

    def calculate_cross_modal_consistency(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        Measure consistency of information across different modalities in sources

        Args:
            answer: Generated answer
            sources: List of source dictionaries with metadata

        Returns:
            Consistency score (0-1)
        """
        if not sources:
            return 0.0

        # Extract modality types from sources
        modalities = []
        for source in sources:
            source_type = source.get("source_type", "unknown")
            modalities.append(source_type)

        # Count modality types
        modality_counts = Counter(modalities)
        unique_modalities = len(modality_counts)

        # If only one modality, consistency is neutral
        if unique_modalities <= 1:
            return 0.7  # Baseline for single modality

        # Check for cross-modal references in answer
        cross_modal_indicators = [
            r"according to",
            r"mentioned in",
            r"shown in",
            r"discussed in",
            r"from the [video|pdf|course|lecture]",
            r"in the [video|pdf|course|lecture]"
        ]

        cross_modal_score = 0.0
        for pattern in cross_modal_indicators:
            if re.search(pattern, answer, re.IGNORECASE):
                cross_modal_score += 0.2
                break

        # Check for diverse source usage
        diversity_bonus = min(unique_modalities / 3.0, 0.3)  # Max 0.3 bonus

        # Consistency based on balanced usage of modalities
        if len(modalities) >= 3:
            balanced_usage = max(modality_counts.values()) / len(modalities)
            balance_score = 1.0 - min(balanced_usage - 0.5, 0.5) * 2  # Penalize over-reliance on one modality
        else:
            balance_score = 0.7

        # Combine scores
        consistency = (cross_modal_score + diversity_bonus + balance_score) / 3.0
        return min(consistency, 1.0)

    def calculate_temporal_coherence_quality(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        Measure temporal logical flow in video-based answers

        Args:
            answer: Generated answer
            sources: List of source dictionaries with metadata

        Returns:
            Temporal coherence score (0-1)
        """
        video_sources = [s for s in sources if s.get("source_type") == "video"]

        if not video_sources:
            return 1.0  # No video sources, temporal coherence is not applicable

        # Check for temporal ordering indicators in answer
        temporal_indicators = [
            r"\bfirst\b",
            r"\bthen\b",
            r"\bnext\b",
            r"\bafter\b",
            r"\bbefore\b",
            r"\bsubsequently\b",
            r"\bfinally\b",
            r"\binitially\b",
            r"\bfollowed by\b"
        ]

        temporal_score = 0.0
        for pattern in temporal_indicators:
            if re.search(pattern, answer, re.IGNORECASE):
                temporal_score += 0.15
                break

        # Check for timestamp references
        timestamp_pattern = r"\d{1,2}:\d{2}"
        if re.search(timestamp_pattern, answer):
            temporal_score += 0.2

        # Check for sequential progression in video sources
        if len(video_sources) >= 2:
            # Check if sources are in temporal order
            timestamps = []
            for source in video_sources:
                timestamp_start = source.get("timestamp_start")
                if timestamp_start is not None:
                    timestamps.append(timestamp_start)

            if timestamps and len(timestamps) >= 2:
                # Check if timestamps are in ascending order
                is_ordered = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
                if is_ordered:
                    temporal_score += 0.3

        # Check for logical flow indicators
        logical_flow_patterns = [
            r"\btherefore\b",
            r"\bconsequently\b",
            r"\bas a result\b",
            r"\bthis leads to\b",
            r"\bwhich means\b"
        ]

        for pattern in logical_flow_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                temporal_score += 0.1
                break

        return min(temporal_score, 1.0)

    def calculate_multimodal_context_utilization(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        Measure how effectively the system uses different modalities

        Args:
            answer: Generated answer
            sources: List of source dictionaries with metadata

        Returns:
            Context utilization score (0-1)
        """
        if not sources:
            return 0.0

        # Count modalities in sources
        modality_counts = Counter([s.get("source_type", "unknown") for s in sources])
        total_modalities = len(modality_counts)

        # Calculate utilization based on modality diversity
        diversity_score = min(total_modalities / 3.0, 1.0)  # Normalize to max 3 modalities

        # Check for explicit modality references in answer
        modality_references = {
            "video": [r"video", r"lecture", r"course", r"instructor", r"professor"],
            "pdf": [r"pdf", r"textbook", r"notes", r"document", r"chapter"],
            "web": [r"website", r"aman.ai", r"online", r"documentation", r"tutorial"]
        }

        reference_score = 0.0
        for modality, patterns in modality_references.items():
            if modality in modality_counts:
                for pattern in patterns:
                    if re.search(pattern, answer, re.IGNORECASE):
                        reference_score += 0.1
                        break

        # Check for source integration quality
        source_integration_patterns = [
            r"according to",
            r"as mentioned",
            r"as shown",
            r"as explained",
            r"based on"
        ]

        integration_score = 0.0
        for pattern in source_integration_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                integration_score += 0.2
                break

        # Combine scores
        utilization = (diversity_score * 0.4) + (reference_score * 0.4) + (integration_score * 0.2)
        return min(utilization, 1.0)

    def calculate_source_diversity(
        self,
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        Measure diversity of information sources across modalities

        Args:
            sources: List of source dictionaries with metadata

        Returns:
            Source diversity score (0-1)
        """
        if not sources:
            return 0.0

        # Extract unique source identifiers
        unique_sources = set()

        for source in sources:
            # Create unique identifier based on source type and name
            source_type = source.get("source_type", "unknown")
            source_name = source.get("source_name", "unknown")

            # For video sources, include video_id
            if source_type == "video":
                video_id = source.get("metadata", {}).get("video_id", "unknown")
                identifier = f"{source_type}:{video_id}"
            else:
                identifier = f"{source_type}:{source_name}"

            unique_sources.add(identifier)

        # Calculate diversity based on unique sources
        unique_count = len(unique_sources)
        total_count = len(sources)

        # Penalize if many duplicate sources
        if total_count > 0:
            diversity = unique_count / total_count
        else:
            diversity = 0.0

        # Bonus for having multiple modalities
        modality_types = set(s.get("source_type", "unknown") for s in sources)
        modality_bonus = min(len(modality_types) / 3.0, 0.2)

        return min(diversity + modality_bonus, 1.0)

    def calculate_citation_accuracy(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        Measure accuracy of citations across different content types

        Args:
            answer: Generated answer
            sources: List of source dictionaries with metadata

        Returns:
            Citation accuracy score (0-1)
        """
        if not sources:
            return 0.0

        # Check for citation patterns in answer
        citation_patterns = [
            r"\[\d+\]",  # [1], [2], etc.
            r"\([^)]+\)",  # (Author, Year) or (Source)
            r"according to [^.]+\.",  # according to [source]
            r"as mentioned in [^.]+\.",  # as mentioned in [source]
            r"source: [^.]+\.",  # source: [name]
        ]

        citation_count = 0
        for pattern in citation_patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            citation_count += len(matches)

        # Calculate citation density (citations per 100 words)
        word_count = len(answer.split())
        if word_count > 0:
            citation_density = (citation_count / word_count) * 100
        else:
            citation_density = 0.0

        # Optimal citation density: 1-3 citations per 100 words
        optimal_density = 1.0 <= citation_density <= 3.0
        density_score = 1.0 if optimal_density else max(0.0, 1.0 - abs(citation_density - 2.0) / 2.0)

        # Check for source-specific citation quality
        source_specific_patterns = {
            "video": [r"at \d{1,2}:\d{2}", r"in the video", r"lecture"],
            "pdf": [r"chapter \d+", r"page \d+", r"section", r"figure"],
            "web": [r"aman.ai", r"documentation", r"tutorial"]
        }

        specificity_score = 0.0
        for source in sources:
            source_type = source.get("source_type", "unknown")
            if source_type in source_specific_patterns:
                for pattern in source_specific_patterns[source_type]:
                    if re.search(pattern, answer, re.IGNORECASE):
                        specificity_score += 0.2
                        break

        # Check for proper citation placement (not just at the end)
        placement_patterns = [
            r"according to [^.]+,",
            r"as [^.]+ states,",
            r"[^.]+ \[\d+\],",
            r"based on [^.]+,",
        ]

        placement_score = 0.0
        for pattern in placement_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                placement_score += 0.3
                break

        # Combine scores
        accuracy = (density_score * 0.4) + (min(specificity_score, 1.0) * 0.4) + (placement_score * 0.2)
        return min(accuracy, 1.0)

    def evaluate_all_metrics(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        query: Optional[str] = None
    ) -> MultiModalMetricsResult:
        """
        Calculate all multi-modal RAG metrics

        Args:
            answer: Generated answer
            sources: List of source dictionaries with metadata
            query: Optional query for context

        Returns:
            MultiModalMetricsResult with all metrics
        """
        details = {
            "num_sources": len(sources),
            "modalities_present": list(set(s.get("source_type", "unknown") for s in sources)),
            "answer_length": len(answer),
            "answer_word_count": len(answer.split())
        }

        return MultiModalMetricsResult(
            cross_modal_consistency=self.calculate_cross_modal_consistency(answer, sources),
            temporal_coherence_quality=self.calculate_temporal_coherence_quality(answer, sources),
            multimodal_context_utilization=self.calculate_multimodal_context_utilization(answer, sources),
            source_diversity=self.calculate_source_diversity(sources),
            citation_accuracy=self.calculate_citation_accuracy(answer, sources),
            details=details
        )

    def evaluate_batch(
        self,
        data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate multi-modal metrics for a batch of query-answer pairs

        Args:
            data: List of dictionaries containing answer, sources, and optional query

        Returns:
            Dictionary with aggregated metrics
        """
        results = []

        for item in data:
            answer = item["answer"]
            sources = item.get("sources", [])
            query = item.get("query")

            result = self.evaluate_all_metrics(answer, sources, query)
            results.append(result)

        # Calculate averages
        if not results:
            return {
                "average_cross_modal_consistency": 0.0,
                "average_temporal_coherence_quality": 0.0,
                "average_multimodal_context_utilization": 0.0,
                "average_source_diversity": 0.0,
                "average_citation_accuracy": 0.0,
                "average_overall_score": 0.0,
                "num_samples": 0
            }

        avg_cross_modal = np.mean([r.cross_modal_consistency for r in results])
        avg_temporal = np.mean([r.temporal_coherence_quality for r in results])
        avg_utilization = np.mean([r.multimodal_context_utilization for r in results])
        avg_diversity = np.mean([r.source_diversity for r in results])
        avg_citation = np.mean([r.citation_accuracy for r in results])
        avg_overall = np.mean([r.get_average_score() for r in results])

        return {
            "average_cross_modal_consistency": avg_cross_modal,
            "average_temporal_coherence_quality": avg_temporal,
            "average_multimodal_context_utilization": avg_utilization,
            "average_source_diversity": avg_diversity,
            "average_citation_accuracy": avg_citation,
            "average_overall_score": avg_overall,
            "num_samples": len(results),
            "individual_results": [r.to_dict() for r in results]
        }


def create_multimodal_rag_evaluator() -> MultiModalRAGEvaluator:
    """
    Factory function to create multi-modal RAG evaluator

    Returns:
        MultiModalRAGEvaluator instance
    """
    return MultiModalRAGEvaluator()


if __name__ == "__main__":
    # Test multi-modal evaluator
    logging.basicConfig(level=logging.INFO)

    evaluator = create_multimodal_rag_evaluator()

    # Test data
    test_answer = """
    Machine learning is a subset of artificial intelligence that enables systems to learn from data.
    According to the lecture at 15:30 in the Stanford ML course, supervised learning uses labeled data.
    As mentioned in the PDF textbook chapter 3, this differs from traditional programming.
    The aman.ai documentation also explains how modern ML approaches have evolved.
    First, we collect data, then we train models, and finally we deploy them.
    """

    test_sources = [
        {
            "source_type": "video",
            "source_name": "Stanford ML",
            "metadata": {"video_id": "abc123", "timestamp_start": 900, "timestamp_end": 1000}
        },
        {
            "source_type": "pdf",
            "source_name": "ML_Textbook",
            "metadata": {"chapter": "3", "page_start": 45}
        },
        {
            "source_type": "web",
            "source_name": "aman.ai",
            "metadata": {"url": "https://aman.ai/ml/intro"}
        }
    ]

    # Evaluate
    result = evaluator.evaluate_all_metrics(test_answer, test_sources)

    print("Multi-Modal RAG Metrics Results:")
    print(f"Cross-Modal Consistency: {result.cross_modal_consistency:.3f}")
    print(f"Temporal Coherence Quality: {result.temporal_coherence_quality:.3f}")
    print(f"Multi-Modal Context Utilization: {result.multimodal_context_utilization:.3f}")
    print(f"Source Diversity: {result.source_diversity:.3f}")
    print(f"Citation Accuracy: {result.citation_accuracy:.3f}")
    print(f"Overall Score: {result.get_average_score():.3f}")
    print(f"\nDetails: {result.details}")