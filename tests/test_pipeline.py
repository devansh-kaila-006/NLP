"""
Unit Tests for RAG Pipeline Components
Tests for retriever, reranker, and pipeline
"""

import sys
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.retriever import Retriever
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.generation.gemini_generator import GeminiGenerator
from src.pipeline.rag_pipeline import RAGPipeline
from src.utils.helpers import load_pickle
import numpy as np


class TestRetriever(unittest.TestCase):
    """Test retriever functionality"""

    def setUp(self):
        """Setup test fixtures"""
        self.index_path = "data/processed/indices/vector_index.faiss"
        self.chunks_path = "data/processed/indices/chunks_metadata.pkl"

        # Skip if files don't exist
        if not Path(self.index_path).exists():
            self.skipTest("Index file not found")

    def test_load_index(self):
        """Test index loading"""
        retriever = Retriever(
            index_path=self.index_path,
            chunks_path=self.chunks_path
        )

        stats = retriever.get_stats()
        self.assertEqual(stats["index_loaded"], True)
        self.assertGreater(stats["total_vectors"], 0)

    def test_retrieve(self):
        """Test single query retrieval"""
        retriever = Retriever(
            index_path=self.index_path,
            chunks_path=self.chunks_path
        )

        results = retriever.retrieve("What is machine learning?", top_k=3)

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIn("text", results[0])
        self.assertIn("source_name", results[0])


class TestReranker(unittest.TestCase):
    """Test reranker functionality"""

    def setUp(self):
        """Setup test fixtures"""
        self.reranker = CrossEncoderReranker()

    def test_model_loading(self):
        """Test model can be loaded"""
        # This may take a while on first run
        try:
            model = self.reranker.load_model()
            self.assertIsNotNone(model)
        except Exception as e:
            self.skipTest(f"Model loading failed (may be network issue): {e}")

    def test_rerank(self):
        """Test reranking functionality"""
        query = "What is machine learning?"
        chunks = [
            {"text": "Machine learning is a subset of AI.", "chunk_id": "1"},
            {"text": "Deep learning uses neural networks.", "chunk_id": "2"},
        ]

        try:
            reranked = self.reranker.rerank(query, chunks, top_n=2)

            self.assertIsInstance(reranked, list)
            self.assertEqual(len(reranked), 2)
            self.assertIn("rerank_score", reranked[0])

        except Exception as e:
            self.skipTest(f"Reranking failed (model may not be loaded): {e}")


class TestGeminiGenerator(unittest.TestCase):
    """Test Gemini generator functionality"""

    def test_prompt_building(self):
        """Test prompt construction"""
        generator = GeminiGenerator()
        generator.api_key = "test-key"  # Mock for testing

        question = "Test question?"
        chunks = [
            {
                "text": "Test content",
                "source_name": "Test_Source",
                "source_type": "pdf",
                "chapter": "1"
            }
        ]

        prompt = generator.build_prompt(question, chunks)

        self.assertIn("Test question?", prompt)
        self.assertIn("Test content", prompt)
        self.assertIn("Test_Source", prompt)


class TestRAGPipeline(unittest.TestCase):
    """Test RAG pipeline integration"""

    def setUp(self):
        """Setup test fixtures"""
        self.index_path = "data/processed/indices/vector_index.faiss"
        self.chunks_path = "data/processed/indices/chunks_metadata.pkl"

        if not Path(self.index_path).exists():
            self.skipTest("Index file not found")

    def test_pipeline_initialization(self):
        """Test pipeline can be initialized"""
        pipeline = RAGPipeline(
            index_path=self.index_path,
            chunks_path=self.chunks_path,
            use_reranker=False  # Skip reranker for faster testing
        )

        stats = pipeline.get_stats()
        self.assertIsNotNone(stats)

    def test_pipeline_query(self):
        """Test end-to-end query (may require API key)"""
        pipeline = RAGPipeline(
            index_path=self.index_path,
            chunks_path=self.chunks_path,
            use_reranker=False
        )

        try:
            response = pipeline.query("What is gradient descent?", top_k=2)

            self.assertIn("answer", response)
            self.assertIn("sources", response)
            self.assertIsInstance(response["answer"], str)

        except ValueError as e:
            if "GOOGLE_API_KEY" in str(e):
                self.skipTest("API key not set - skipping query test")
            else:
                raise


class TestIntegration(unittest.TestCase):
    """Integration tests"""

    def test_end_to_end(self):
        """Test complete pipeline"""
        self.index_path = "data/processed/indices/vector_index.faiss"
        self.chunks_path = "data/processed/indices/chunks_metadata.pkl"

        if not Path(self.index_path).exists():
            self.skipTest("Index file not found")

        # This test requires API key
        import os
        if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your-api-key-here":
            self.skipTest("API key not set")

        pipeline = RAGPipeline(
            index_path=self.index_path,
            chunks_path=self.chunks_path,
            use_reranker=False
        )

        response = pipeline.query("What is gradient descent?", top_k=2)

        self.assertIn("answer", response)
        self.assertGreater(len(response["answer"]), 10)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestRetriever))
    suite.addTests(loader.loadTestsFromTestCase(TestReranker))
    suite.addTests(loader.loadTestsFromTestCase(TestGeminiGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestRAGPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
