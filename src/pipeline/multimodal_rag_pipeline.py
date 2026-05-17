"""
Multi-Modal RAG Pipeline - Integrates PDF and Video Sources
"""

import time
from typing import List, Dict, Any

from src.retrieval.retriever import Retriever
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.generation.gemini_generator import GeminiGenerator
from src.processors.video_chunker import VideoChunker
from src.utils.logger import LoggerMixin
from src.config import RETRIEVAL_CONFIG, RERANKING_CONFIG


class MultiModalRAGPipeline(LoggerMixin):
    """
    Multi-Modal RAG Pipeline with PDF and Video sources

    Features:
    1. Unified retrieval from PDF and video chunks
    2. Cross-modal reranking (NOVELTY 3)
    3. Temporal coherence for video chunks (NOVELTY 2)
    4. Video timestamp links in responses
    """

    def __init__(
        self,
        pdf_index_path: str = None,
        pdf_chunks_path: str = None,
        video_index_path: str = None,
        video_chunks_path: str = None,
        use_reranker: bool = True
    ):
        """Initialize multi-modal RAG pipeline"""
        self.use_reranker = use_reranker

        # Initialize PDF retriever
        self.pdf_retriever = None
        if pdf_index_path and pdf_chunks_path:
            self.pdf_retriever = Retriever(
                index_path=pdf_index_path,
                chunks_path=pdf_chunks_path,
                config=RETRIEVAL_CONFIG
            )
            self.logger.info(f"PDF retriever initialized: {self.pdf_retriever.get_stats()}")

        # Initialize video retriever
        self.video_retriever = None
        if video_index_path and video_chunks_path:
            self.video_retriever = Retriever(
                index_path=video_index_path,
                chunks_path=video_chunks_path,
                config=RETRIEVAL_CONFIG
            )
            self.logger.info(f"Video retriever initialized: {self.video_retriever.get_stats()}")

        # Initialize reranker
        self.reranker = None
        if use_reranker:
            self.reranker = CrossEncoderReranker(RERANKING_CONFIG)

        # Initialize generator with cross-modal prediction
        self.generator = GeminiGenerator()
        self.video_chunker = VideoChunker()  # For modality prediction

        self.logger.info("Multi-Modal RAG Pipeline initialized")

    def query(
        self,
        question: str,
        top_k: int = 5,
        rerank_top_n: int = 3,
        include_timing: bool = False,
        force_modality: str = None  # 'pdf', 'video', or None (auto)
    ) -> Dict[str, Any]:
        """
        Multi-modal query processing

        Args:
            question: User question
            top_k: Chunks to retrieve per modality
            rerank_top_n: Chunks after reranking
            include_timing: Include timing information
            force_modality: Force specific modality (for testing)

        Returns:
            Response with answer, sources, and video links
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Multi-Modal Query: {question}")
        self.logger.info(f"{'='*60}")

        timings = {}
        total_start = time.time()

        # Step 1: Predict optimal modality (NOVELTY 3)
        modality_start = time.time()

        if force_modality:
            modality_scores = {force_modality: 1.0}
            self.logger.info(f"Forced modality: {force_modality}")
        else:
            modality_scores = self.video_chunker.predict_modality(question)
            best_modality = max(modality_scores, key=modality_scores.get)
            self.logger.info(f"Predicted modality: {best_modality}")
            self.logger.info(f"  Modality scores: {modality_scores}")

        modality_time = time.time() - modality_start
        timings['modality_prediction'] = modality_time

        # Step 2: Retrieve from both modalities
        retrieval_start = time.time()

        pdf_chunks = []
        video_chunks = []

        if self.pdf_retriever and (force_modality != 'video'):
            # Adjust top_k based on modality score
            pdf_top_k = int(top_k * modality_scores.get('pdf', 1.0))
            pdf_top_k = max(pdf_top_k, 2)  # Minimum 2 chunks
            pdf_chunks = self.pdf_retriever.retrieve(question, top_k=pdf_top_k)

        if self.video_retriever and (force_modality != 'pdf'):
            # Adjust top_k based on modality score
            video_top_k = int(top_k * modality_scores.get('video', 1.0))
            video_top_k = max(video_top_k, 2)  # Minimum 2 chunks
            video_chunks = self.video_retriever.retrieve(question, top_k=video_top_k)

        retrieval_time = time.time() - retrieval_start
        timings['retrieval'] = retrieval_time

        self.logger.info(f"Retrieved {len(pdf_chunks)} PDF chunks, {len(video_chunks)} video chunks in {retrieval_time:.2f}s")

        if not pdf_chunks and not video_chunks:
            return {
                "answer": "I apologize, but I couldn't find relevant information in my knowledge base to answer your question.",
                "sources": [],
                "chunks_used": 0,
                "modality_used": "none",
                "error": "No relevant chunks found"
            }

        # Step 3: Rerank within each modality
        all_chunks = []
        rerank_time = 0

        if self.use_reranker and self.reranker:
            self.logger.info("Reranking chunks")

            # Rerank PDF chunks
            if pdf_chunks:
                rerank_start = time.time()
                pdf_chunks = self.reranker.rerank(question, pdf_chunks, top_n=rerank_top_n)
                rerank_time += time.time() - rerank_start

            # Rerank video chunks
            if video_chunks:
                rerank_start = time.time()
                video_chunks = self.reranker.rerank(question, video_chunks, top_n=rerank_top_n)
                rerank_time += time.time() - rerank_start

            timings['reranking'] = rerank_time

            # Combine chunks (interleave based on rerank scores)
            all_chunks = self._combine_chunks(pdf_chunks, video_chunks)
        else:
            # Simple concatenation without reranking
            all_chunks = pdf_chunks + video_chunks

        self.logger.info(f"Selected {len(all_chunks)} chunks after reranking")

        # Step 4: Apply temporal coherence to video chunks (NOVELTY 2)
        if video_chunks:
            video_chunks = self._apply_temporal_coherence(video_chunks, question)

        # Step 5: Generate response
        generation_start = time.time()

        response = self.generator.query(question, all_chunks)

        generation_time = time.time() - generation_start
        timings['generation'] = generation_time

        # Step 6: Format response with video links
        formatted_response = self._format_multimodal_response(
            response,
            pdf_chunks,
            video_chunks,
            modality_scores
        )

        # Add timings
        total_time = time.time() - total_start
        timings['total'] = total_time

        if include_timing:
            formatted_response['timings'] = timings

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Query completed in {total_time:.2f}s")
        self.logger.info(f"  Modality prediction: {timings.get('modality_prediction', 0):.2f}s")
        self.logger.info(f"  Retrieval: {timings.get('retrieval', 0):.2f}s")
        if 'reranking' in timings:
            self.logger.info(f"  Reranking: {timings['reranking']:.2f}s")
        self.logger.info(f"  Generation: {timings['generation']:.2f}s")
        self.logger.info(f"{'='*60}\n")

        return formatted_response

    def _combine_chunks(
        self,
        pdf_chunks: List[Dict],
        video_chunks: List[Dict]
    ) -> List[Dict]:
        """Combine PDF and video chunks, interleaving by relevance score"""

        combined = []

        # Add relevance scores to all chunks
        for chunk in pdf_chunks:
            chunk['modality'] = 'pdf'
            combined.append(chunk)

        for chunk in video_chunks:
            chunk['modality'] = 'video'
            combined.append(chunk)

        # Sort by rerank score (if available) or relevance score
        combined.sort(
            key=lambda x: x.get('rerank_score', x.get('relevance_score', 0)),
            reverse=True
        )

        # Take top chunks
        return combined[:5]  # Limit to 5 total chunks

    def _apply_temporal_coherence(
        self,
        video_chunks: List[Dict],
        question: str
    ) -> List[Dict]:
        """Apply temporal coherence to video chunks (NOVELTY 2)"""

        if len(video_chunks) <= 1:
            return video_chunks

        # Sort chunks by timestamp to ensure temporal flow
        video_chunks_sorted = sorted(video_chunks, key=lambda x: x.get('timestamp_start', 0))

        # Check if chunks are from same lecture
        lecture_numbers = [c.get('lecture_number') for c in video_chunks_sorted]

        if len(set(lecture_numbers)) == 1:
            # Same lecture - check temporal ordering
            timestamps = [c.get('timestamp_start', 0) for c in video_chunks_sorted]

            if timestamps == sorted(timestamps):
                self.logger.info("  Video chunks are temporally coherent")
            else:
                self.logger.info("  Reordered video chunks for temporal coherence")
                video_chunks = video_chunks_sorted

        return video_chunks

    def _format_multimodal_response(
        self,
        base_response: Dict,
        pdf_chunks: List[Dict],
        video_chunks: List[Dict],
        modality_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Format response with video links and modality information"""

        # Extract sources
        sources = []
        seen_sources = set()

        # Add PDF sources
        for chunk in pdf_chunks:
            source_name = chunk.get('source_name', 'Unknown')
            if source_name not in seen_sources:
                seen_sources.add(source_name)
                sources.append({
                    "name": source_name,
                    "type": "pdf",
                    "modality": "pdf",
                    "relevance": chunk.get('rerank_score', chunk.get('relevance_score', 0))
                })

        # Add video sources with timestamps
        for chunk in video_chunks:
            source_name = chunk.get('source_name', 'Unknown')
            if source_name not in seen_sources:
                seen_sources.add(source_name)
                source_info = {
                    "name": source_name,
                    "type": "video",
                    "modality": "video",
                    "relevance": chunk.get('rerank_score', chunk.get('relevance_score', 0)),
                    "timestamp_start": chunk.get('timestamp_start', 0),
                    "timestamp_end": chunk.get('timestamp_end', 0),
                    "video_url": chunk.get('video_url'),
                    "timestamp_url": chunk.get('timestamp_url')
                }
                sources.append(source_info)

        # Build response
        response = {
            "answer": base_response.get('answer', ''),
            "sources": sources,
            "chunks_used": len(pdf_chunks) + len(video_chunks),
            "modality_scores": modality_scores,
            "pdf_chunks_used": len(pdf_chunks),
            "video_chunks_used": len(video_chunks),
            "model": base_response.get('model', 'unknown')
        }

        # Add video links section if video chunks were used
        if video_chunks:
            video_links = []
            for chunk in video_chunks:
                if chunk.get('timestamp_url'):
                    video_links.append({
                        "source": chunk.get('source_name', 'Unknown'),
                        "timestamp": f"{chunk.get('timestamp_start', 0)/60:.0f}-{chunk.get('timestamp_end', 0)/60:.0f}min",
                        "url": chunk.get('timestamp_url')
                    })
            response['video_links'] = video_links

        return response

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = {
            "pdf_available": self.pdf_retriever is not None,
            "video_available": self.video_retriever is not None,
            "reranker_enabled": self.use_reranker
        }

        if self.pdf_retriever:
            stats["pdf_stats"] = self.pdf_retriever.get_stats()

        if self.video_retriever:
            stats["video_stats"] = self.video_retriever.get_stats()

        return stats


if __name__ == "__main__":
    # Test multi-modal RAG pipeline
    try:
        pipeline = MultiModalRAGPipeline(
            pdf_index_path="data/processed/indices/vector_index.faiss",
            pdf_chunks_path="data/processed/indices/chunks_metadata.pkl",
            video_index_path="data/processed/video_chunks/video_index.faiss",
            video_chunks_path="data/processed/video_chunks/video_metadata.pkl",
            use_reranker=True
        )

        print("\n" + "="*60)
        print("MULTI-MODAL RAG PIPELINE TEST")
        print("="*60)

        # Test queries
        test_queries = [
            "What is machine learning?",  # Should prefer video
            "What's the formula for gradient descent?",  # Should prefer PDF (math)
            "Explain how backpropagation works",  # Mixed
        ]

        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print('='*60)

            response = pipeline.query(query, include_timing=True)

            print(f"\nAnswer:\n{response['answer']}\n")

            print(f"Sources:")
            for source in response['sources']:
                if source['type'] == 'video':
                    timestamp = source['timestamp_start']
                    print(f"  [VIDEO] {source['name']} ({timestamp/60:.0f}min)")
                    if source.get('timestamp_url'):
                        print(f"          URL: {source['timestamp_url']}")
                else:
                    print(f"  [PDF] {source['name']}")

            print(f"\nModality Scores: {response['modality_scores']}")
            print(f"Chunks Used: {response['chunks_used']} (PDF: {response['pdf_chunks_used']}, Video: {response['video_chunks_used']})")

            if 'timings' in response:
                timings = response['timings']
                print(f"\nTiming: {timings['total']:.2f}s total")
                print(f"  Modality prediction: {timings['modality_prediction']:.2f}s")
                print(f"  Retrieval: {timings['retrieval']:.2f}s")
                print(f"  Generation: {timings['generation']:.2f}s")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
