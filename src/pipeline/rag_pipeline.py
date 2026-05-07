"""
RAG Pipeline - Main orchestrator for Retrieval Augmented Generation
Combines retriever, reranker, and LLM generator
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.retrieval.retriever import Retriever
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.generation.gemini_generator import GeminiGenerator
from src.utils.logger import LoggerMixin
from src.utils.helpers import Timer
from src.config import RETRIEVAL_CONFIG, RERANKING_CONFIG


class RAGPipeline(LoggerMixin):
    """
    End-to-end RAG pipeline for question answering
    """

    def __init__(
        self,
        index_path: str | Path = None,
        chunks_path: str | Path = None,
        use_reranker: bool = True,
        config: Dict[str, Any] = None
    ):
        """
        Initialize RAG pipeline

        Args:
            index_path: Path to FAISS index
            chunks_path: Path to chunks metadata
            use_reranker: Enable/disable reranking
            config: Pipeline configuration
        """
        self.use_reranker = use_reranker
        self.config = config or {}

        # Initialize components
        self.retriever = Retriever(
            index_path=index_path,
            chunks_path=chunks_path,
            config=RETRIEVAL_CONFIG
        )

        self.reranker = None
        if use_reranker:
            self.reranker = CrossEncoderReranker(RERANKING_CONFIG)

        self.generator = GeminiGenerator()

        self.logger.info("RAG Pipeline initialized")
        self.logger.info(f"  Reranker: {'Enabled' if use_reranker else 'Disabled'}")

    def query(
        self,
        question: str,
        top_k: int = 5,
        rerank_top_n: int = 3,
        return_sources: bool = True,
        include_timing: bool = False
    ) -> Dict[str, Any]:
        """
        End-to-end RAG query processing

        Args:
            question: User question
            top_k: Number of chunks to retrieve
            rerank_top_n: Number of chunks after reranking
            return_sources: Include source information
            include_timing: Include timing information

        Returns:
            Response dictionary with answer and metadata
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Query: {question}")
        self.logger.info(f"{'='*60}")

        timings = {}
        total_start = time.time()

        # Step 1: Retrieval
        self.logger.info("Step 1: Retrieval")
        retrieval_start = time.time()

        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)

        retrieval_time = time.time() - retrieval_start
        timings['retrieval'] = retrieval_time

        self.logger.info(f"  Retrieved {len(retrieved_chunks)} chunks in {retrieval_time:.2f}s")

        if not retrieved_chunks:
            return {
                "answer": "I apologize, but I couldn't find relevant information in my knowledge base to answer your question.",
                "sources": [],
                "chunks_used": 0,
                "error": "No relevant chunks found"
            }

        # Step 2: Reranking (optional)
        reranked_chunks = retrieved_chunks
        if self.use_reranker and self.reranker:
            self.logger.info("Step 2: Reranking")
            rerank_start = time.time()

            reranked_chunks = self.reranker.rerank(question, retrieved_chunks, top_n=rerank_top_n)

            rerank_time = time.time() - rerank_start
            timings['reranking'] = rerank_time

            self.logger.info(f"  Reranked to {len(reranked_chunks)} chunks in {rerank_time:.2f}s")

            # Log rerank scores
            if reranked_chunks:
                scores = [c.get('rerank_score', 0) for c in reranked_chunks]
                self.logger.info(f"  Rerank scores: {[f'{s:.4f}' for s in scores]}")

        # Step 3: Generation
        self.logger.info("Step 3: Generation")
        generation_start = time.time()

        response = self.generator.query(question, reranked_chunks)

        generation_time = time.time() - generation_start
        timings['generation'] = generation_time

        self.logger.info(f"  Generated answer in {generation_time:.2f}s")

        # Finalize response
        total_time = time.time() - total_start
        timings['total'] = total_time

        # Add timings to response
        if include_timing:
            response['timings'] = timings

        # Add used chunks
        if return_sources:
            response['used_chunks'] = reranked_chunks

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Query completed in {total_time:.2f}s")
        self.logger.info(f"  Retrieval: {timings.get('retrieval', 0):.2f}s")
        if 'reranking' in timings:
            self.logger.info(f"  Reranking: {timings['reranking']:.2f}s")
        self.logger.info(f"  Generation: {timings['generation']:.2f}s")
        self.logger.info(f"{'='*60}\n")

        return response

    def batch_query(
        self,
        questions: List[str],
        top_k: int = 5,
        rerank_top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries

        Args:
            questions: List of questions
            top_k: Chunks to retrieve per query
            rerank_top_n: Chunks after reranking

        Returns:
            List of response dictionaries
        """
        self.logger.info(f"\nBatch processing {len(questions)} queries")

        responses = []
        for i, question in enumerate(questions):
            self.logger.info(f"\nProcessing query {i+1}/{len(questions)}")
            response = self.query(question, top_k=top_k, rerank_top_n=rerank_top_n)
            responses.append(response)

        return responses

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics

        Returns:
            Dictionary with pipeline info
        """
        stats = {
            "retriever": self.retriever.get_stats(),
            "reranker_enabled": self.use_reranker,
            "generator_model": self.generator.model_name
        }

        if self.reranker:
            stats["reranker"] = self.reranker.get_model_info()

        return stats

    def interactive_mode(self):
        """
        Run interactive query loop
        """
        print("\n" + "="*60)
        print("RAG Pipeline - Interactive Mode")
        print("="*60)
        print("\nCommands:")
        print("  /quit or /exit - Exit")
        print("  /help - Show help")
        print("  /stats - Show pipeline stats")
        print("\n")

        while True:
            try:
                question = input("Your question: ").strip()

                if not question:
                    continue

                if question.lower() in ['/quit', '/exit']:
                    print("\nGoodbye!")
                    break

                if question.lower() == '/help':
                    print("\nCommands:")
                    print("  /quit or /exit - Exit")
                    print("  /stats - Show pipeline stats")
                    print("  /help - Show this help")
                    continue

                if question.lower() == '/stats':
                    stats = self.get_stats()
                    print("\nPipeline Stats:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue

                # Process query
                response = self.query(question, include_timing=True)

                # Display answer
                print(f"\nAnswer:\n{response['answer']}\n")

                # Display sources
                if response.get('sources'):
                    print("Sources:")
                    for source in response['sources']:
                        print(f"  - {source['name']} (relevance: {source['relevance']:.4f})")

                # Display timing
                if 'timings' in response:
                    timings = response['timings']
                    print(f"\nTiming: {timings['total']:.2f}s "
                          f"(retrieval: {timings['retrieval']:.2f}s, "
                          f"generation: {timings['generation']:.2f}s)")

                print("\n" + "-"*60 + "\n")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                self.logger.error(f"Error processing query: {e}")
                print(f"\nError: {e}\n")


if __name__ == "__main__":
    # Test RAG pipeline
    try:
        pipeline = RAGPipeline(
            index_path="data/processed/indices/vector_index.faiss",
            chunks_path="data/processed/indices/chunks_metadata.pkl",
            use_reranker=True
        )

        # Test query
        question = "What is gradient descent?"
        response = pipeline.query(question, include_timing=True)

        print(f"\nQuestion: {question}")
        print(f"\nAnswer:\n{response['answer']}\n")

        if response.get('sources'):
            print("Sources:")
            for source in response['sources']:
                print(f"  - {source['name']}")

        if 'timings' in response:
            timings = response['timings']
            print(f"\nTimings: {timings['total']:.2f}s total")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you've run: python scripts/02_build_index.py")
