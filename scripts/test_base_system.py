"""
Test script for base RAG system.

This script demonstrates the end-to-end functionality of the base RAG system.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.base_rag.pdf_processor import PDFProcessor
from src.base_rag.retriever import MultiModalRetriever
from src.base_rag.llm_generator import LLMGenerator
from src.utils.logger import setup_logging, get_logger
from src.utils.exceptions import RAGSystemError

# Setup logging
setup_logging(log_level="INFO", log_file="logs/test_system.log")
logger = get_logger(__name__)


def create_sample_data():
    """
    Create sample data for testing.

    Returns:
        List of sample chunks
    """
    logger.info("Creating sample data")

    sample_chunks = [
        {
            "text": "Convolutional Neural Networks (CNNs) are a type of deep learning architecture specifically designed for processing grid-like data such as images. They use convolutional layers to apply filters across the input, capturing spatial hierarchies and patterns.",
            "source": "cs231n_lecture1",
            "page": 5,
            "topic": "computer vision",
            "difficulty": "intermediate",
            "chunk_type": "pdf"
        },
        {
            "text": "The backpropagation algorithm is the key to training neural networks. It computes gradients of the loss function with respect to each weight by applying the chain rule, working backwards from the output layer to the input layer.",
            "source": "deep_learning_book_ch5",
            "page": 120,
            "topic": "deep learning",
            "difficulty": "intermediate",
            "chunk_type": "pdf"
        },
        {
            "text": "Recurrent Neural Networks (RNNs) are designed to handle sequential data by maintaining a hidden state that captures information about previous inputs. This makes them suitable for tasks like language modeling and time series prediction.",
            "source": "cs224n_lecture2",
            "page": 8,
            "topic": "natural language processing",
            "difficulty": "intermediate",
            "chunk_type": "pdf"
        },
        {
            "text": "In machine learning, overfitting occurs when a model learns the training data too well, including its noise and outliers, leading to poor generalization to new data. Techniques like regularization and cross-validation help prevent overfitting.",
            "source": "cs229_notes_lecture3",
            "page": 45,
            "topic": "machine learning",
            "difficulty": "beginner",
            "chunk_type": "pdf"
        },
        {
            "text": "Gradient descent is an optimization algorithm used to minimize the loss function by iteratively moving in the direction of steepest descent. The learning rate determines the size of each step and is a crucial hyperparameter to tune.",
            "source": "cs229_notes_lecture2",
            "page": 30,
            "topic": "machine learning",
            "difficulty": "beginner",
            "chunk_type": "pdf"
        }
    ]

    logger.info(f"Created {len(sample_chunks)} sample chunks")
    return sample_chunks


def test_retriever(retriever, queries):
    """
    Test the retriever with sample queries.

    Args:
        retriever: MultiModalRetriever instance
        queries: List of test queries
    """
    logger.info("Testing retriever")

    for query in queries:
        logger.info(f"\nQuery: {query}")

        try:
            results = retriever.retrieve(query, k=3)

            logger.info(f"Retrieved {len(results)} results:")

            for i, result in enumerate(results, start=1):
                logger.info(f"\nResult {i}:")
                logger.info(f"  Score: {result.get('score', 0):.4f}")
                logger.info(f"  Source: {result.get('source', 'Unknown')}")
                logger.info(f"  Text: {result.get('text', '')[:100]}...")

        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")


def test_llm_generator(generator, retriever, queries):
    """
    Test the LLM generator with sample queries.

    Args:
        generator: LLMGenerator instance
        retriever: MultiModalRetriever instance
        queries: List of test queries
    """
    logger.info("Testing LLM generator")

    for query in queries:
        logger.info(f"\nQuery: {query}")

        try:
            # Retrieve relevant chunks
            retrieved_chunks = retriever.retrieve(query, k=3)

            # Generate response
            response = generator.generate_with_sources(query, retrieved_chunks)

            logger.info(f"\nResponse:")
            logger.info(response)

        except Exception as e:
            logger.error(f"Generation failed for query '{query}': {e}")


def main():
    """
    Main test function.
    """
    logger.info("Starting base RAG system test")

    try:
        # Check for API key
        if not os.getenv("GOOGLE_API_KEY"):
            logger.error("GOOGLE_API_KEY environment variable not set")
            logger.info("Set it using: export GOOGLE_API_KEY=your_key_here")
            return

        # Create sample data
        sample_chunks = create_sample_data()

        # Initialize retriever
        logger.info("Initializing retriever")
        retriever = MultiModalRetriever(
            embedding_model="all-MiniLM-L6-v2",
            index_dir="indices/test_index"
        )

        # Index sample data
        logger.info("Indexing sample data")
        retriever.index_chunks(sample_chunks, save_index=True)

        # Get index stats
        stats = retriever.get_index_stats()
        logger.info(f"Index stats: {stats}")

        # Test queries
        queries = [
            "What is a CNN?",
            "How does backpropagation work?",
            "What is overfitting in machine learning?"
        ]

        # Test retriever
        test_retriever(retriever, queries)

        # Test LLM generator (only if API key is set)
        if os.getenv("GOOGLE_API_KEY"):
            logger.info("Initializing LLM generator")
            generator = LLMGenerator()

            # Test with one query (to save API calls)
            test_llm_generator(generator, retriever, [queries[0]])
        else:
            logger.info("Skipping LLM generator test (no API key)")

        logger.info("Test completed successfully")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
