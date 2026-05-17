"""
Multi-Video Retriever - Combines Multiple Video Playlists
"""

from typing import List, Dict, Any
from src.retrieval.retriever import Retriever
from src.utils.logger import LoggerMixin


class MultiVideoRetriever(LoggerMixin):
    """
    Retriever that combines multiple video playlists into one unified search
    """

    def __init__(self, video_playlists: List[Dict[str, str]], config: Dict[str, Any] = None):
        """
        Initialize multi-video retriever

        Args:
            video_playlists: List of dicts with 'index_path' and 'chunks_path'
            config: Retrieval configuration
        """
        self.config = config or {}
        self.retrievers = []

        # Initialize individual retrievers for each playlist
        for playlist in video_playlists:
            try:
                retriever = Retriever(
                    index_path=playlist['index_path'],
                    chunks_path=playlist['chunks_path'],
                    config=self.config
                )
                self.retrievers.append(retriever)
                self.logger.info(f"Loaded playlist: {playlist.get('name', 'Unknown')}")

            except Exception as e:
                self.logger.warning(f"Failed to load playlist {playlist.get('name', 'Unknown')}: {e}")

        if not self.retrievers:
            raise ValueError("No video playlists could be loaded")

        self.logger.info(f"Multi-video retriever initialized with {len(self.retrievers)} playlists")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve from all video playlists combined

        Args:
            query: Query text (retrievers handle embedding generation)
            top_k: Total number of results to return

        Returns:
            Combined list of retrieved chunks from all playlists
        """
        all_results = []

        # Retrieve from each playlist
        for retriever in self.retrievers:
            try:
                results = retriever.retrieve(query, top_k=top_k)
                all_results.extend(results)
            except Exception as e:
                self.logger.warning(f"Retrieval failed for one playlist: {e}")

        # Sort by relevance score
        all_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        # Return top_k results
        return all_results[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics"""
        total_vectors = 0
        playlist_stats = []

        for i, retriever in enumerate(self.retrievers):
            stats = retriever.get_stats()
            total_vectors += stats.get('total_vectors', 0)
            playlist_stats.append(stats)

        return {
            'index_loaded': True,
            'total_playlists': len(self.retrievers),
            'total_vectors': total_vectors,
            'playlist_stats': playlist_stats,
            'embedding_model': self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        }
