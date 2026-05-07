"""
Evaluation Script - Test the RAG system with sample questions
Evaluates retrieval quality and answer generation
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.rag_pipeline import RAGPipeline
from src.utils.logger import setup_logger
from src.utils.helpers import Timer


# Evaluation questions
EVALUATION_QUESTIONS = [
    {
        "question": "What is gradient descent?",
        "topic": "Optimization",
        "difficulty": "Beginner",
        "expected_keywords": ["optimization", "algorithm", "gradient", "descent"]
    },
    {
        "question": "Explain how backpropagation works",
        "topic": "Neural Networks",
        "difficulty": "Intermediate",
        "expected_keywords": ["gradient", "chain rule", "neural network", "backward"]
    },
    {
        "question": "What is overfitting in machine learning?",
        "topic": "ML Theory",
        "difficulty": "Beginner",
        "expected_keywords": ["overfitting", "training", "generalization", "variance"]
    },
    {
        "question": "How does a CNN differ from a regular neural network?",
        "topic": "Deep Learning",
        "difficulty": "Intermediate",
        "expected_keywords": ["convolution", "pooling", "spatial", "feature"]
    },
    {
        "question": "What is the difference between L1 and L2 regularization?",
        "topic": "ML Theory",
        "difficulty": "Intermediate",
        "expected_keywords": ["regularization", "L1", "L2", "penalty"]
    },
    {
        "question": "How do you use scikit-learn's LinearRegression?",
        "topic": "Implementation",
        "difficulty": "Beginner",
        "expected_keywords": ["fit", "predict", "scikit-learn", "LinearRegression"]
    },
    {
        "question": "What is the vanishing gradient problem?",
        "topic": "Deep Learning",
        "difficulty": "Advanced",
        "expected_keywords": ["gradient", "vanishing", "backpropagation", "deep"]
    },
    {
        "question": "Explain the bias-variance tradeoff",
        "topic": "ML Theory",
        "difficulty": "Intermediate",
        "expected_keywords": ["bias", "variance", "tradeoff", "model complexity"]
    }
]


def evaluate_retrieval(pipeline: RAGPipeline, question_data: dict) -> dict:
    """
    Evaluate retrieval quality

    Args:
        pipeline: RAG pipeline
        question_data: Question dictionary

    Returns:
        Evaluation metrics
    """
    question = question_data["question"]

    start = time.time()
    response = pipeline.query(question, top_k=5, rerank_top_n=3, include_timing=False)
    retrieval_time = time.time() - start

    # Check if keywords are present in chunks
    chunks_used = response.get("used_chunks", [])
    keyword_coverage = 0

    for chunk in chunks_used:
        chunk_text = chunk["text"].lower()
        for keyword in question_data["expected_keywords"]:
            if keyword.lower() in chunk_text:
                keyword_coverage += 1
                break

    coverage_pct = (keyword_coverage / len(question_data["expected_keywords"])) * 100

    return {
        "retrieval_time": retrieval_time,
        "chunks_retrieved": len(chunks_used),
        "keyword_coverage": coverage_pct,
        "sources": [c["source_name"] for c in response.get("sources", [])]
    }


def main():
    """Main evaluation function"""
    logger = setup_logger("evaluation")

    logger.info("=" * 60)
    logger.info("RAG System Evaluation")
    logger.info("=" * 60)

    # Initialize pipeline
    try:
        pipeline = RAGPipeline(
            index_path="data/processed/indices/vector_index.faiss",
            chunks_path="data/processed/indices/chunks_metadata.pkl",
            use_reranker=True
        )
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return 1

    # Evaluate each question
    results = []

    logger.info(f"\nEvaluating {len(EVALUATION_QUESTIONS)} questions...")
    logger.info("-" * 60)

    for i, question_data in enumerate(EVALUATION_QUESTIONS):
        logger.info(f"\n{i+1}. {question_data['question']}")
        logger.info(f"   Topic: {question_data['topic']}")
        logger.info(f"   Difficulty: {question_data['difficulty']}")

        try:
            metrics = evaluate_retrieval(pipeline, question_data)

            # Get full response
            response = pipeline.query(question_data["question"])
            answer = response.get("answer", "")[:200]

            logger.info(f"   Retrieval time: {metrics['retrieval_time']:.2f}s")
            logger.info(f"   Chunks retrieved: {metrics['chunks_retrieved']}")
            logger.info(f"   Keyword coverage: {metrics['keyword_coverage']:.1f}%")
            logger.info(f"   Sources: {', '.join(set(metrics['sources']))}")
            logger.info(f"   Answer preview: {answer}...")

            results.append({
                "question": question_data["question"],
                "retrieval_time": metrics["retrieval_time"],
                "keyword_coverage": metrics["keyword_coverage"],
                "chunks_used": metrics["chunks_retrieved"],
                "sources": metrics["sources"]
            })

        except Exception as e:
            logger.error(f"   Error evaluating question: {e}")
            results.append({
                "question": question_data["question"],
                "error": str(e)
            })

    # Calculate summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)

    successful_results = [r for r in results if "error" not in r]

    if successful_results:
        avg_time = sum(r["retrieval_time"] for r in successful_results) / len(successful_results)
        avg_coverage = sum(r["keyword_coverage"] for r in successful_results) / len(successful_results)

        logger.info(f"Questions evaluated: {len(success_results)}/{len(EVALUATION_QUESTIONS)}")
        logger.info(f"Average retrieval time: {avg_time:.2f}s")
        logger.info(f"Average keyword coverage: {avg_coverage:.1f}%")

        # Source distribution
        all_sources = []
        for r in successful_results:
            all_sources.extend(r["sources"])

        from collections import Counter
        source_counts = Counter(all_sources)
        logger.info(f"\nSource usage:")
        for source, count in source_counts.most_common():
            logger.info(f"  {source}: {count} times")

    else:
        logger.warning("No successful evaluations")

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
