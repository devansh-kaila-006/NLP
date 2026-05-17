"""
Video Chunker - Implements Novelty Features for Multi-Modal RAG

Three Novel Contributions:
1. Timestamp-Aware Video RAG: Semantic chunking using transcript analysis
2. Temporal Coherence: Flow-aware retrieval using temporal dependency graphs
3. Cross-Modal Reranking: Adaptive modality selection based on query analysis
"""

import re
from pathlib import Path
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.loaders.srt_loader import TranscriptSegment, SRTLoader


@dataclass
class VideoChunk:
    """Represents a semantically chunked video segment"""
    chunk_id: str
    video_source: str
    lecture_number: int
    start_time: float      # seconds
    end_time: float        # seconds
    text: str
    segments: List[TranscriptSegment]
    embedding: np.ndarray = None

    # Metadata for temporal coherence
    previous_chunks: List[str] = None
    next_chunks: List[str] = None
    temporal_score: float = 0.0

    # For video linking
    video_url: str = None
    timestamp_url: str = None

    def duration_minutes(self) -> float:
        return (self.end_time - self.start_time) / 60.0

    def __repr__(self):
        return f"VideoChunk({self.chunk_id}, {self.duration_minutes():.1f}min, {self.start_time/60:.0f}-{self.end_time/60:.0f}min)"


class VideoChunker:
    """
    Advanced video chunking with novelty features

    Implements:
    1. Semantic chunking using transcript analysis (vs fixed intervals)
    2. Temporal coherence using dependency graphs
    3. Cross-modal query analysis
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        min_chunk_duration: float = 120.0,   # 2 minutes
        max_chunk_duration: float = 300.0,   # 5 minutes
        semantic_threshold: float = 0.6,     # For topic boundary detection
        temporal_window: int = 3             # For temporal coherence
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.min_chunk_duration = min_chunk_duration
        self.max_chunk_duration = max_chunk_duration
        self.semantic_threshold = semantic_threshold
        self.temporal_window = temporal_window

        self.srt_loader = SRTLoader()
        self.logger = None

    # ========================================================================
    # NOVELTY 1: Timestamp-Aware Video RAG (Semantic Chunking)
    # ========================================================================

    def detect_topic_boundaries(
        self,
        segments: List[TranscriptSegment]
    ) -> List[int]:
        """
        Detect topic boundaries using transcript analysis

        NOVELTY: Instead of fixed 3-minute intervals, we detect actual topic shifts
        by analyzing semantic similarity between consecutive segments.

        Args:
            segments: List of transcript segments

        Returns:
            List of boundary indices (where topic changes occur)
        """
        # Extract embeddings for each segment
        texts = [seg.text for seg in segments]
        embeddings = self.embedding_model.encode(texts)

        # Calculate similarity between consecutive segments
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                [embeddings[i]],
                [embeddings[i + 1]]
            )[0][0]
            similarities.append(sim)

        # Find sharp drops in similarity (topic boundaries)
        boundaries = [0]  # Always start at beginning

        for i, sim in enumerate(similarities):
            if sim < self.semantic_threshold:
                # Check if we're far enough from previous boundary
                if i - boundaries[-1] >= 3:  # At least 3 segments
                    boundaries.append(i + 1)

        boundaries.append(len(segments))  # Always end at the end
        return boundaries

    def create_semantic_chunks(
        self,
        segments: List[TranscriptSegment],
        video_source: str,
        lecture_number: int
    ) -> List[VideoChunk]:
        """
        Create semantically coherent video chunks

        NOVELTY: Groups transcript segments by topic boundaries rather than
        fixed time intervals. Ensures each chunk covers a single concept.

        Args:
            segments: Transcript segments
            video_source: Source video name
            lecture_number: Lecture number

        Returns:
            List of VideoChunk objects
        """
        # Detect topic boundaries
        boundaries = self.detect_topic_boundaries(segments)

        chunks = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            # Get segments for this chunk
            chunk_segments = segments[start_idx:end_idx]

            # Calculate duration
            start_time = chunk_segments[0].start_time
            end_time = chunk_segments[-1].end_time
            duration = end_time - start_time

            # Skip if too short
            if duration < self.min_chunk_duration:
                # Merge with next chunk if too short
                if i + 1 < len(boundaries) - 1:
                    end_idx = boundaries[i + 2]
                    chunk_segments = segments[start_idx:end_idx]
                    end_time = chunk_segments[-1].end_time
                    duration = end_time - start_time

            # Skip if still too short
            if duration < self.min_chunk_duration:
                continue

            # Skip if too long (split further)
            if duration > self.max_chunk_duration:
                # Split into sub-chunks
                sub_chunks = self.split_long_chunk(
                    chunk_segments, video_source, lecture_number
                )
                chunks.extend(sub_chunks)
                continue

            # Combine text
            text = ' '.join(seg.text for seg in chunk_segments)

            # Create chunk
            chunk = VideoChunk(
                chunk_id=f"{video_source}_L{lecture_number:02d}_C{i+1:03d}",
                video_source=video_source,
                lecture_number=lecture_number,
                start_time=start_time,
                end_time=end_time,
                text=text,
                segments=chunk_segments
            )

            chunks.append(chunk)

        if self.logger:
            self.logger.info(f"Created {len(chunks)} semantic chunks from lecture {lecture_number}")

        return chunks

    def split_long_chunk(
        self,
        segments: List[TranscriptSegment],
        video_source: str,
        lecture_number: int
    ) -> List[VideoChunk]:
        """Split a long chunk into smaller semantic pieces"""
        # Find mid-point based on time
        mid_time = (segments[0].start_time + segments[-1].end_time) / 2

        # Find segment closest to mid-time
        for i, seg in enumerate(segments):
            if seg.start_time >= mid_time:
                split_idx = i
                break
        else:
            split_idx = len(segments) // 2

        # Split into two chunks
        chunk1_segments = segments[:split_idx]
        chunk2_segments = segments[split_idx:]

        chunks = []
        for idx, chunk_segments in enumerate([chunk1_segments, chunk2_segments], 1):
            text = ' '.join(seg.text for seg in chunk_segments)
            chunk = VideoChunk(
                chunk_id=f"{video_source}_L{lecture_number:02d}_C{idx:03d}",
                video_source=video_source,
                lecture_number=lecture_number,
                start_time=chunk_segments[0].start_time,
                end_time=chunk_segments[-1].end_time,
                text=text,
                segments=chunk_segments
            )
            chunks.append(chunk)

        return chunks

    # ========================================================================
    # NOVELTY 2: Temporal Coherence
    # ========================================================================

    def build_temporal_graph(
        self,
        chunks: List[VideoChunk]
    ) -> nx.DiGraph:
        """
        Build temporal dependency graph between chunks

        NOVELTY: Creates a graph of temporal dependencies to ensure coherent
        retrieval paths. Chunks that flow logically are connected.

        Args:
            chunks: List of video chunks

        Returns:
            Directed graph representing temporal dependencies
        """
        G = nx.DiGraph()

        # Add all chunks as nodes
        for chunk in chunks:
            G.add_node(chunk.chunk_id, chunk=chunk)

        # Add edges between consecutive chunks (strong connection)
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            # Calculate semantic similarity
            similarity = cosine_similarity(
                [self.embedding_model.encode(current_chunk.text)],
                [self.embedding_model.encode(next_chunk.text)]
            )[0][0]

            # Add edge with weight based on similarity
            G.add_edge(
                current_chunk.chunk_id,
                next_chunk.chunk_id,
                weight=similarity,
                type='consecutive'
            )

        # Add edges for related concepts (skip connections)
        for i in range(len(chunks)):
            for j in range(i + 1, min(i + self.temporal_window + 1, len(chunks))):
                chunk_i = chunks[i]
                chunk_j = chunks[j]

                # Calculate similarity
                similarity = cosine_similarity(
                    [self.embedding_model.encode(chunk_i.text)],
                    [self.embedding_model.encode(chunk_j.text)]
                )[0][0]

                # Add edge if highly related
                if similarity > 0.7:
                    G.add_edge(
                        chunk_i.chunk_id,
                        chunk_j.chunk_id,
                        weight=similarity,
                        type='related'
                    )

        if self.logger:
            self.logger.info(f"Built temporal graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        return G

    def find_coherent_path(
        self,
        graph: nx.DiGraph,
        start_chunk_id: str,
        end_chunk_id: str,
        num_chunks: int = 3
    ) -> List[str]:
        """
        Find temporally coherent path between chunks

        NOVELTY: Instead of returning random chunks, we find a path that
        maintains logical flow through the content.

        Args:
            graph: Temporal dependency graph
            start_chunk_id: Starting chunk
            end_chunk_id: Target chunk
            num_chunks: Number of chunks to return

        Returns:
            List of chunk IDs in coherent order
        """
        try:
            # Find all simple paths between start and end
            paths = list(nx.all_simple_paths(
                graph,
                start_chunk_id,
                end_chunk_id,
                cutoff=num_chunks
            ))

            if not paths:
                # Fallback: return direct path
                return [start_chunk_id, end_chunk_id]

            # Score paths by:
            # 1. Forward flow (prefer paths that move forward)
            # 2. Edge weights (prefer high similarity edges)
            # 3. Length (prefer paths close to num_chunks)

            best_path = None
            best_score = -float('inf')

            for path in paths:
                if len(path) != num_chunks:
                    continue

                # Calculate path score
                score = 0.0
                for i in range(len(path) - 1):
                    if graph.has_edge(path[i], path[i + 1]):
                        score += graph[path[i]][path[i + 1]]['weight']

                # Bonus for forward flow
                if path == sorted(path):
                    score += 0.5

                if score > best_score:
                    best_score = score
                    best_path = path

            return best_path if best_path else paths[0]

        except nx.NetworkXNoPath:
            # No path exists
            return [start_chunk_id, end_chunk_id]

    # ========================================================================
    # NOVELTY 3: Cross-Modal Reranking
    # ========================================================================

    def predict_modality(
        self,
        query: str,
        query_type: str = None
    ) -> Dict[str, float]:
        """
        Predict which modality works best for this query

        NOVELTY: Analyzes query features to predict optimal modality (video vs PDF vs docs).
        Boosts relevant modality during retrieval.

        Args:
            query: User query
            query_type: Pre-classified query type (optional)

        Returns:
            Dictionary with modality scores
        """
        # Initialize base scores
        modality_scores = {
            'video': 1.0,
            'pdf': 1.0,
            'docs': 1.0
        }

        query_lower = query.lower()

        # Feature extraction
        visual_keywords = ['how does', 'how do', 'visualize', 'look', 'see', 'show', 'demonstrate']
        math_keywords = ['formula', 'equation', 'derivation', 'proof', 'mathematics', 'calculate']
        code_keywords = ['implement', 'code', 'python', 'function', 'api', 'library']
        conceptual_keywords = ['what is', 'explain', 'intuition', 'understand', 'concept']

        # Analyze query features
        has_visual = any(keyword in query_lower for keyword in visual_keywords)
        has_math = any(keyword in query_lower for keyword in math_keywords)
        has_code = any(keyword in query_lower for keyword in code_keywords)
        has_concept = any(keyword in query_lower for keyword in conceptual_keywords)

        # Boost modalities based on query features
        if has_visual:
            modality_scores['video'] *= 1.3  # 30% boost for visual queries

        if has_math:
            modality_scores['pdf'] *= 1.4   # 40% boost for math queries

        if has_code:
            modality_scores['docs'] *= 1.5  # 50% boost for code queries

        if has_concept:
            modality_scores['video'] *= 1.2  # 20% boost for conceptual explanations
            modality_scores['pdf'] *= 1.1   # 10% boost for theoretical concepts

        # Normalize scores
        total = sum(modality_scores.values())
        modality_scores = {k: v / total for k, v in modality_scores.items()}

        if self.logger:
            self.logger.info(f"Query modality prediction: {modality_scores}")

        return modality_scores

    # ========================================================================
    # Main Processing Pipeline
    # ========================================================================

    def process_lecture(
        self,
        srt_path: str,
        video_url: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a single lecture: create chunks, build graph, enable retrieval

        Args:
            srt_path: Path to SRT file
            video_url: URL to full video
            metadata: Additional metadata

        Returns:
            Dictionary with chunks, graph, and metadata
        """
        # Extract lecture info from filename
        filename = Path(srt_path).stem
        lecture_match = re.search(r'L(\d+)', filename)
        lecture_num = int(lecture_match.group(1)) if lecture_match else 0

        source_match = re.search(r'(CS\d+)', filename)
        video_source = source_match.group(1) if source_match else "Unknown"

        # Load transcript
        segments = self.srt_loader.load_srt(srt_path)

        # Create semantic chunks (NOVELTY 1)
        chunks = self.create_semantic_chunks(segments, video_source, lecture_num)

        # Add video URLs
        if video_url:
            for chunk in chunks:
                chunk.video_url = video_url
                chunk.timestamp_url = f"{video_url}&t={int(chunk.start_time)}"

        # Generate embeddings
        for chunk in chunks:
            chunk.embedding = self.embedding_model.encode(chunk.text)

        # Build temporal graph (NOVELTY 2)
        temporal_graph = self.build_temporal_graph(chunks)

        result = {
            'chunks': chunks,
            'temporal_graph': temporal_graph,
            'metadata': {
                'lecture_number': lecture_num,
                'video_source': video_source,
                'video_url': video_url,
                'num_chunks': len(chunks),
                'total_duration': sum(seg.duration for seg in segments) / 3600.0
            }
        }

        if metadata:
            result['metadata'].update(metadata)

        if self.logger:
            self.logger.info(f"Processed lecture {lecture_num}: {len(chunks)} chunks, {temporal_graph.number_of_nodes()} nodes")

        return result


if __name__ == "__main__":
    # Test video chunker
    chunker = VideoChunker()

    # Process Lecture 1
    result = chunker.process_lecture(
        "data/transcripts/cs229/CS229_L01_I_Introduction_Lecture_1.srt",
        video_url="https://www.youtube.com/watch?v=CS229_Lecture1"
    )

    print(f"\n=== Lecture 1 Processing Results ===")
    print(f"Created {result['metadata']['num_chunks']} chunks")
    print(f"Total duration: {result['metadata']['total_duration']:.1f} hours")
    print(f"Temporal graph: {result['temporal_graph'].number_of_nodes()} nodes")

    print(f"\n=== First 3 Chunks ===")
    for chunk in result['chunks'][:3]:
        print(chunk)

    # Test modality prediction
    print(f"\n=== Modality Prediction Tests ===")
    test_queries = [
        "What is machine learning?",
        "How does convolution work?",
        "What's the formula for gradient descent?",
        "How do I implement backprop in PyTorch?"
    ]

    for query in test_queries:
        modality_scores = chunker.predict_modality(query)
        print(f"\nQuery: {query}")
        print(f"Modality scores: {modality_scores}")
