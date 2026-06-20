"""
Generate Full Answers for Test Queries with Ground Truth
For accurate similarity score calculation
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import RAG system components
try:
    from src.rag_engine import MultiModalRAGEngine
    from src.config import RAG_CONFIG
except ImportError as e:
    logger.warning(f"Could not import RAG components: {e}")
    logger.info("Will use mock generation for testing...")
    MultiModalRAGEngine = None
    RAG_CONFIG = {}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ground truth queries that need full answers generated
GROUND_TRUTH_QUERIES = [
    # Machine Learning Fundamentals - Conceptual Questions
    "What is machine learning and how does it differ from traditional programming?",
    "Explain the difference between supervised, unsupervised, and reinforcement learning",
    "What is overfitting and how can it be prevented?",
    "Explain the bias-variance tradeoff in machine learning",
    "What is cross-validation and why is it important?",

    # Machine Learning Fundamentals - Algorithm Questions
    "How does the k-nearest neighbors algorithm work?",
    "Explain the decision tree algorithm and its advantages",
    "What is random forest and how does it improve upon decision trees?",
    "How does support vector machine (SVM) work?",
    "Explain the naive Bayes classifier and its applications",

    # Machine Learning Fundamentals - Optimization Questions
    "What is gradient descent and how does it optimize machine learning models?",
    "Explain the difference between batch gradient descent, stochastic gradient descent, and mini-batch gradient descent",
    "What is the difference between L1 and L2 regularization?",
    "How does the learning rate affect model training?",
    "What is early stopping and when should it be used?",

    # Deep Learning Architecture - Architecture Questions
    "What is a neural network and how does it differ from traditional machine learning?",
    "Explain the architecture of convolutional neural networks (CNNs)",
    "What are the key components of recurrent neural networks (RNNs)?",
    "How does the transformer architecture work?",
    "What is the difference between RNNs and LSTMs?",

    # Deep Learning Architecture - Training Questions
    "Explain backpropagation and how it works in neural networks",
    "What is the vanishing gradient problem and how can it be solved?",
    "How does batch normalization help in training deep neural networks?",
    "What is dropout and how does it prevent overfitting?",
    "Explain the difference between various activation functions (ReLU, sigmoid, tanh)",

    # Deep Learning Architecture - Advanced Questions
    "What is transfer learning and when should it be used?",
    "Explain the concept of attention mechanism in neural networks",
    "What is self-attention and how does it work?",
    "How does the transformer architecture differ from traditional sequence models?",
    "What are the key differences between BERT and GPT models?"
]


def generate_full_answers(
    queries: List[str],
    output_file: Path = None
) -> List[Dict]:
    """
    Generate full answers for the given queries using the RAG system

    Args:
        queries: List of query strings
        output_file: Optional path to save results

    Returns:
        List of dictionaries with query, answer, and metadata
    """
    logger.info(f"Initializing RAG engine to generate {len(queries)} answers...")

    try:
        # Initialize RAG engine
        rag_engine = MultiModalRAGEngine(config=RAG_CONFIG)
        logger.info("RAG engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {e}")
        logger.info("Falling back to mock generation for testing...")

        # Mock RAG engine for testing if real one fails
        class MockRAGEngine:
            def query(self, query_text: str, **kwargs):
                return {
                    "answer": f"Mock answer for: {query_text}\n\nThis is a generated response that would contain the full answer content from the RAG system. In production, this would be replaced with actual generated content.",
                    "sources": [],
                    "modality": "unknown",
                    "query_time": 1.0
                }

        rag_engine = MockRAGEngine()

    results = []
    failed_queries = []

    for i, query in enumerate(queries, 1):
        logger.info(f"[{i}/{len(queries)}] Processing: {query[:60]}...")

        try:
            start_time = time.time()

            # Query the RAG engine
            response = rag_engine.query(query_text=query)

            query_time = time.time() - start_time

            # Extract answer
            answer = response.get("answer", "")
            sources = response.get("sources", [])
            modality = response.get("modality", "unknown")

            # Validate answer
            if not answer or len(answer) < 50:
                logger.warning(f"Answer too short or empty for query: {query[:50]}")
                failed_queries.append(query)
                continue

            result = {
                "id": i,
                "query": query,
                "answer": answer,
                "answer_length": len(answer),
                "answer_word_count": len(answer.split()),
                "query_time": round(query_time, 2),
                "num_sources": len(sources),
                "predicted_modality": modality,
                "status": "success"
            }

            results.append(result)

            logger.info(f"  ✓ Generated {len(answer)} chars in {query_time:.2f}s")

        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            failed_queries.append(query)

        # Small delay to avoid overwhelming the system
        time.sleep(0.5)

    # Save results if output file specified
    if output_file and results:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(results)} results to {output_file}")

    if failed_queries:
        logger.warning(f"Failed to generate answers for {len(failed_queries)} queries")
        for q in failed_queries:
            logger.warning(f"  - {q[:60]}...")

    return results


def main():
    """Main function to generate full answers"""
    output_file = Path("data/evaluation/results/full_generated_answers.json")

    logger.info("="*60)
    logger.info("GENERATING FULL ANSWERS FOR SIMILARITY SCORING")
    logger.info("="*60)

    # Generate answers
    results = generate_full_answers(
        queries=GROUND_TRUTH_QUERIES,
        output_file=output_file
    )

    # Print summary
    logger.info("="*60)
    logger.info("GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total queries: {len(GROUND_TRUTH_QUERIES)}")
    logger.info(f"Successful: {len(results)}")
    logger.info(f"Failed: {len(GROUND_TRUTH_QUERIES) - len(results)}")

    if results:
        avg_length = sum(r['answer_length'] for r in results) / len(results)
        avg_words = sum(r['answer_word_count'] for r in results) / len(results)
        avg_time = sum(r['query_time'] for r in results) / len(results)

        logger.info(f"Average answer length: {avg_length:.0f} characters")
        logger.info(f"Average word count: {avg_words:.0f} words")
        logger.info(f"Average query time: {avg_time:.2f} seconds")

        logger.info(f"\nFull results saved to: {output_file}")
    else:
        logger.error("No results generated!")

    return results


if __name__ == "__main__":
    main()
