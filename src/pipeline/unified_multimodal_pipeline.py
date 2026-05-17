"""
Unified Multi-Modal RAG Pipeline - All 5 Video Playlists Combined
"""

import time
from typing import List, Dict, Any
from src.retrieval.retriever import Retriever
from src.retrieval.multi_video_retriever import MultiVideoRetriever
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.reranking.enhanced_cross_encoder_reranker import EnhancedCrossEncoderReranker
from src.generation.gemini_generator import GeminiGenerator
from src.processors.video_chunker import VideoChunker
from src.utils.logger import LoggerMixin
from src.config import RETRIEVAL_CONFIG, RERANKING_CONFIG, ENHANCED_RERANKING_CONFIG


class UnifiedMultiModalRAGPipeline(LoggerMixin):
    """
    Unified Multi-Modal RAG Pipeline with ALL 5 video playlists

    Features:
    1. Combined retrieval from 5 video playlists + PDFs
    2. Cross-modal reranking
    3. Temporal coherence for video chunks
    4. Video timestamp links in responses
    """

    def __init__(self, use_reranker: bool = True, include_aman: bool = True):
        """Initialize unified multi-modal RAG pipeline"""
        self.use_reranker = use_reranker
        self.include_aman = include_aman

        # Initialize PDF retriever
        self.pdf_retriever = None
        try:
            self.pdf_retriever = Retriever(
                index_path="data/processed/indices/vector_index.faiss",
                chunks_path="data/processed/indices/chunks_metadata.pkl",
                config=RETRIEVAL_CONFIG
            )
            self.logger.info(f"PDF retriever initialized: {self.pdf_retriever.get_stats()}")
        except Exception as e:
            self.logger.warning(f"PDF retriever initialization failed: {e}")

        # Initialize Aman.ai retriever (modern web content)
        self.aman_retriever = None
        if include_aman:
            try:
                self.aman_retriever = Retriever(
                    index_path="data/processed/aman_primers/aman_index.faiss",
                    chunks_path="data/processed/aman_primers/aman_metadata.pkl",
                    config=RETRIEVAL_CONFIG
                )
                self.logger.info(f"Aman.ai retriever initialized: {self.aman_retriever.get_stats()}")
            except Exception as e:
                self.logger.warning(f"Aman.ai retriever initialization failed: {e}")

        # Initialize multi-video retriever with all 5 playlists
        video_playlists = [
            {
                "name": "CS229 Machine Learning",
                "index_path": "data/processed/video_chunks/video_index.faiss",
                "chunks_path": "data/processed/video_chunks/video_metadata.pkl"
            },
            {
                "name": "MIT DL Alternative",
                "index_path": "data/processed/video_chunks_mit_dl/mit_dl_video_index.faiss",
                "chunks_path": "data/processed/video_chunks_mit_dl/mit_dl_video_metadata.pkl"
            },
            {
                "name": "CS224n NLP",
                "index_path": "data/processed/video_chunks_cs224n/cs224n_video_index.faiss",
                "chunks_path": "data/processed/video_chunks_cs224n/cs224n_video_metadata.pkl"
            },
            {
                "name": "CS231n Computer Vision",
                "index_path": "data/processed/video_chunks_cs231n/cs231n_video_index.faiss",
                "chunks_path": "data/processed/video_chunks_cs231n/cs231n_video_metadata.pkl"
            },
            {
                "name": "MIT DL Main",
                "index_path": "data/processed/video_chunks_mit_dl_main/mit_dl_main_video_index.faiss",
                "chunks_path": "data/processed/video_chunks_mit_dl_main/mit_dl_main_video_metadata.pkl"
            }
        ]

        try:
            self.video_retriever = MultiVideoRetriever(video_playlists, config=RETRIEVAL_CONFIG)
            self.logger.info(f"Multi-video retriever initialized: {self.video_retriever.get_stats()}")
        except Exception as e:
            self.logger.error(f"Multi-video retriever initialization failed: {e}")
            self.video_retriever = None

        # Initialize enhanced reranker for better precision
        self.reranker = None
        if use_reranker:
            try:
                self.reranker = EnhancedCrossEncoderReranker(ENHANCED_RERANKING_CONFIG)
                self.logger.info("Enhanced cross-encoder reranker initialized")
            except Exception as e:
                self.logger.warning(f"Enhanced reranker initialization failed: {e}")
                self.logger.info("Falling back to standard reranker")
                self.reranker = CrossEncoderReranker(RERANKING_CONFIG)

        # Initialize generator
        self.generator = GeminiGenerator()
        self.video_chunker = VideoChunker()

        self.logger.info("Unified Multi-Modal RAG Pipeline initialized")

    def query(
        self,
        question: str,
        top_k: int = 5,
        rerank_top_n: int = 3,
        include_timing: bool = False,
        force_modality: str = None
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
        self.logger.info(f"Unified Multi-Modal Query: {question}")
        self.logger.info(f"{'='*60}")

        timings = {}
        total_start = time.time()

        # Step 1: Predict optimal modality
        modality_start = time.time()
        modality_scores = self._predict_modality(question)

        if force_modality:
            if force_modality == 'video':
                modality_scores = {'video': 1.0, 'pdf': 0.0, 'aman': 0.0}
            elif force_modality == 'pdf':
                modality_scores = {'video': 0.0, 'pdf': 1.0, 'aman': 0.0}
            elif force_modality == 'aman':
                modality_scores = {'video': 0.0, 'pdf': 0.0, 'aman': 1.0}

        modality_time = time.time() - modality_start
        timings['modality_prediction'] = modality_time

        predicted_modality = max(modality_scores, key=modality_scores.get)
        self.logger.info(f"Predicted modality: {predicted_modality}")
        self.logger.info(f"  Modality scores: {modality_scores}")

        # Step 2: Retrieve from both modalities
        retrieval_start = time.time()

        # Retrieve chunks (retrievers handle embedding generation internally)
        pdf_chunks = []
        video_chunks = []
        aman_chunks = []

        if self.pdf_retriever:
            pdf_chunks = self.pdf_retriever.retrieve(question, top_k=top_k)

        if self.video_retriever:
            video_chunks = self.video_retriever.retrieve(question, top_k=top_k)

        if self.aman_retriever:
            aman_chunks = self.aman_retriever.retrieve(question, top_k=top_k)

        retrieval_time = time.time() - retrieval_start
        timings['retrieval'] = retrieval_time

        self.logger.info(f"Retrieved {len(pdf_chunks)} PDF chunks, {len(video_chunks)} video chunks, {len(aman_chunks)} Aman.ai chunks in {retrieval_time:.2f}s")

        # Step 3: Rerank chunks
        if self.reranker and (pdf_chunks or video_chunks or aman_chunks):
            rerank_time = 0

            if pdf_chunks:
                rerank_start = time.time()
                pdf_chunks = self.reranker.rerank(question, pdf_chunks, top_n=rerank_top_n)
                rerank_time += time.time() - rerank_start

            if video_chunks:
                rerank_start = time.time()
                video_chunks = self.reranker.rerank(question, video_chunks, top_n=rerank_top_n)
                rerank_time += time.time() - rerank_start

            if aman_chunks:
                rerank_start = time.time()
                aman_chunks = self.reranker.rerank(question, aman_chunks, top_n=rerank_top_n)
                rerank_time += time.time() - rerank_start

            timings['reranking'] = rerank_time

        # Combine chunks
        all_chunks = pdf_chunks + video_chunks + aman_chunks

        # Step 4: Apply temporal coherence to video chunks
        if video_chunks:
            video_chunks = self._apply_temporal_coherence(video_chunks)

        # Step 5: Generate response
        generation_start = time.time()
        response = self.generator.query(question, all_chunks)
        generation_time = time.time() - generation_start
        timings['generation'] = generation_time

        # Step 6: Format response
        formatted_response = self._format_multimodal_response(
            response,
            pdf_chunks,
            video_chunks,
            aman_chunks,
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

    def _predict_modality(self, question: str) -> Dict[str, float]:
        """Predict optimal modality based on question content"""
        # Simple heuristic-based prediction for 3 modalities
        question_lower = question.lower()

        # Mathematical/formula indicators (PDF preference)
        math_keywords = ['formula', 'equation', 'derivation', 'prove', 'math', 'calculate',
                        'gradient', 'derivative', 'integral', 'matrix', 'vector']

        # Conceptual/explanation indicators (Video preference)
        conceptual_keywords = ['explain', 'what is', 'how does', 'describe', 'overview',
                              'introduction', 'summary', 'intuition', 'concept']

        # Modern AI/LLM indicators (Aman.ai preference - modern content)
        modern_ai_keywords = ['transformer', 'attention', 'bert', 'gpt', 'llama', 'llm',
                             'diffusion', 'gan', 'stable diffusion', 'chatgpt', 'prompt',
                             'rag', 'reinforcement learning', 'fine-tuning', 'embedding',
                             'tokenization', 'vision language model', 'multimodal']

        # Calculate scores
        conceptual_score = sum(1 for keyword in conceptual_keywords if keyword in question_lower)
        math_score = sum(1 for keyword in math_keywords if keyword in question_lower)
        modern_score = sum(1 for keyword in modern_ai_keywords if keyword in question_lower)

        # Base scores with small prior to avoid zero
        total_keywords = conceptual_score + math_score + modern_score + 1

        video_score = (conceptual_score + 0.3) / total_keywords
        pdf_score = (math_score + 0.3) / total_keywords
        aman_score = (modern_score + 0.4) / total_keywords  # Slight preference for modern content

        # Normalize
        total = video_score + pdf_score + aman_score
        video_score = video_score / total
        pdf_score = pdf_score / total
        aman_score = aman_score / total

        return {'video': video_score, 'pdf': pdf_score, 'aman': aman_score}

    def _apply_temporal_coherence(self, video_chunks: List[Dict]) -> List[Dict]:
        """Apply temporal coherence to video chunks"""
        if len(video_chunks) <= 1:
            return video_chunks

        # Sort chunks by timestamp
        video_chunks_sorted = sorted(video_chunks, key=lambda x: x.get('timestamp_start', 0))

        # Check if chunks are from same lecture
        lecture_numbers = [c.get('lecture_number') for c in video_chunks_sorted]

        if len(set(lecture_numbers)) == 1:
            timestamps = [c.get('timestamp_start', 0) for c in video_chunks_sorted]
            if timestamps == sorted(timestamps):
                self.logger.info("  Video chunks are temporally coherent")
                return video_chunks_sorted

        return video_chunks_sorted

    def _format_multimodal_response(self, base_response: Dict, pdf_chunks: List[Dict],
                                   video_chunks: List[Dict], aman_chunks: List[Dict],
                                   modality_scores: Dict[str, float]) -> Dict[str, Any]:
        """Format response with video links, aman.ai sources, and modality information"""
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

        # Add Aman.ai sources
        for chunk in aman_chunks:
            source_name = chunk.get('source_name', 'Unknown')
            if source_name not in seen_sources:
                seen_sources.add(source_name)
                sources.append({
                    "name": source_name,
                    "type": "aman_primer",
                    "modality": "aman",
                    "relevance": chunk.get('rerank_score', chunk.get('relevance_score', 0)),
                    "category": chunk.get('category', 'General'),
                    "url": chunk.get('url', '')
                })

        # Build response
        response = {
            "answer": base_response.get('answer', ''),
            "sources": sources,
            "chunks_used": len(pdf_chunks) + len(video_chunks) + len(aman_chunks),
            "modality_scores": modality_scores,
            "pdf_chunks_used": len(pdf_chunks),
            "video_chunks_used": len(video_chunks),
            "aman_chunks_used": len(aman_chunks),
            "model": base_response.get('model', 'unknown')
        }

        # Add video links section
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
            "aman_available": self.aman_retriever is not None,
            "reranker_enabled": self.use_reranker,
            "aman_enabled": self.include_aman
        }

        if self.pdf_retriever:
            stats["pdf_stats"] = self.pdf_retriever.get_stats()

        if self.video_retriever:
            stats["video_stats"] = self.video_retriever.get_stats()

        if self.aman_retriever:
            stats["aman_stats"] = self.aman_retriever.get_stats()

        return stats


if __name__ == "__main__":
    # Test unified pipeline with Aman.ai
    try:
        pipeline = UnifiedMultiModalRAGPipeline(use_reranker=True, include_aman=True)

        print("\n" + "="*60)
        print("UNIFIED MULTI-MODAL RAG PIPELINE TEST")
        print("PDF + Video + Aman.ai")
        print("="*60)

        # Test queries
        test_queries = [
            "What is machine learning?",  # Should prefer CS229
            "Explain transformers",  # Should prefer Aman.ai (modern content)
            "What are word embeddings?",  # Should prefer CS224n
            "How do CNNs work?",  # Should prefer CS231n
            "What is RAG?",  # Should prefer Aman.ai (modern topic)
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
                elif source['type'] == 'aman_primer':
                    print(f"  [AMAN.AI] {source['name']} ({source.get('category', 'General')})")
                else:
                    print(f"  [PDF] {source['name']}")

            print(f"\nModality Scores: {response['modality_scores']}")
            print(f"Chunks Used: {response['chunks_used']} (PDF: {response['pdf_chunks_used']}, Video: {response['video_chunks_used']}, Aman.ai: {response.get('aman_chunks_used', 0)})")

        print(f"\n[SUCCESS] Unified pipeline with all 5 playlists + Aman.ai working!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
