"""
Comprehensive RAG Evaluation Script
Runs the complete 72-query test set with RAG quality metrics
"""

import sys
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src directory to path for proper imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))
# Also add the parent directory to import test_query_set
sys.path.insert(0, str(src_path.parent))

# Local imports
from src.pipeline.unified_multimodal_pipeline import UnifiedMultiModalRAGPipeline
from src.evaluation.evaluators.multimodal_rag_evaluator import create_comprehensive_rag_evaluator
from test_query_set import FLAT_TEST_QUERIES, ALL_TEST_QUERIES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveRAGTestRunner:
    """
    Test runner for comprehensive RAG evaluation
    """

    def __init__(self):
        """Initialize test runner"""
        logger.info("Initializing Comprehensive RAG Test Runner")

        # Initialize pipeline
        logger.info("Loading Multi-Modal RAG Pipeline...")
        self.pipeline = UnifiedMultiModalRAGPipeline(use_reranker=True, include_aman=True)

        # Initialize evaluator
        logger.info("Initializing Comprehensive RAG Evaluator...")
        self.evaluator = create_comprehensive_rag_evaluator(
            enable_ragas=True,
            enable_multimodal=True
        )

        # Show system stats
        stats = self.pipeline.get_stats()
        logger.info(f"System Stats: PDF={stats.get('pdf_stats', {}).get('total_vectors', 0)}, "
                   f"Video={stats.get('video_stats', {}).get('total_vectors', 0)}, "
                   f"Aman={stats.get('aman_stats', {}).get('total_vectors', 0)}")

    def process_query_with_metadata(self, query: str) -> Dict[str, Any]:
        """
        Process query and extract metadata for evaluation

        Args:
            query: User query

        Returns:
            Dictionary with answer, contexts, and sources metadata
        """
        try:
            # Run query through pipeline
            result = self.pipeline.query(query, top_k=5)

            # Extract answer
            answer = result.get('answer', '')

            # Extract contexts from retrieved chunks
            contexts = []
            sources = []

            # Process video chunks
            for chunk in result.get('retrieved_video_chunks', []):
                contexts.append(chunk.get('text', ''))
                sources.append({
                    'source_type': 'video',
                    'source_name': chunk.get('playlist_name', 'Unknown'),
                    'metadata': {
                        'video_id': chunk.get('video_id'),
                        'timestamp_start': chunk.get('timestamp_start'),
                        'timestamp_end': chunk.get('timestamp_end'),
                        'video_title': chunk.get('video_title')
                    }
                })

            # Process PDF chunks
            for chunk in result.get('retrieved_pdf_chunks', []):
                contexts.append(chunk.get('text', ''))
                sources.append({
                    'source_type': 'pdf',
                    'source_name': chunk.get('source_name', 'Unknown'),
                    'metadata': {
                        'chapter': chunk.get('metadata', {}).get('chapter'),
                        'page_start': chunk.get('metadata', {}).get('page_start')
                    }
                })

            # Process Aman.ai chunks
            for chunk in result.get('retrieved_aman_chunks', []):
                contexts.append(chunk.get('text', ''))
                sources.append({
                    'source_type': 'web',
                    'source_name': 'aman.ai',
                    'metadata': {
                        'url': chunk.get('url'),
                        'title': chunk.get('title')
                    }
                })

            return {
                'query': query,
                'answer': answer,
                'contexts': contexts,
                'sources': sources,
                'success': True,
                'error': None
            }

        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            return {
                'query': query,
                'answer': '',
                'contexts': [],
                'sources': [],
                'success': False,
                'error': str(e)
            }

    def run_full_evaluation(
        self,
        save_results: bool = True,
        output_dir: Path = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive RAG evaluation on all 72 test queries

        Args:
            save_results: Whether to save results to file
            output_dir: Directory to save results

        Returns:
            Dictionary with comprehensive evaluation results
        """
        if output_dir is None:
            output_dir = Path("data/evaluation_results")

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("COMPREHENSIVE RAG EVALUATION - 72 QUERY TEST SET")
        logger.info("=" * 80)
        logger.info(f"Started at: {datetime.now().isoformat()}")

        start_time = time.time()

        # Process all queries
        evaluation_data = []

        for i, item in enumerate(FLAT_TEST_QUERIES):
            query = item["query"]
            category = item["category"]

            logger.info(f"Processing query {i+1}/72 [{category}]: {query[:60]}...")

            # Process query
            query_result = self.process_query_with_metadata(query)

            # Add category
            query_result['category'] = category

            # Add to evaluation data
            evaluation_data.append(query_result)

            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/72 queries processed")

        # Filter successful queries for RAG evaluation
        successful_queries = [q for q in evaluation_data if q['success']]
        failed_queries = [q for q in evaluation_data if not q['success']]

        logger.info(f"Successfully processed: {len(successful_queries)}/72 queries")
        logger.info(f"Failed queries: {len(failed_queries)}/72")

        # Run comprehensive RAG evaluation
        logger.info("Running comprehensive RAG metrics evaluation...")

        # Prepare data for evaluator
        rag_eval_data = []
        for query_result in successful_queries:
            rag_eval_data.append({
                'query': query_result['query'],
                'contexts': query_result['contexts'],
                'answer': query_result['answer'],
                'sources': query_result['sources']
            })

        # Run evaluation
        eval_start_time = time.time()
        evaluation_results = self.evaluator.evaluate_batch(
            evaluation_data=rag_eval_data,
            save_path=output_dir / "rag_evaluation_results.json" if save_results else None
        )
        eval_time = time.time() - eval_start_time

        total_time = time.time() - start_time

        # Compile comprehensive results
        comprehensive_results = {
            'summary': {
                'total_queries': len(FLAT_TEST_QUERIES),
                'successful_queries': len(successful_queries),
                'failed_queries': len(failed_queries),
                'success_rate': len(successful_queries) / len(FLAT_TEST_QUERIES),
                'total_evaluation_time': total_time,
                'rag_metrics_evaluation_time': eval_time,
                'timestamp': datetime.now().isoformat()
            },
            'rag_metrics': evaluation_results,
            'category_breakdown': self._analyze_category_performance(evaluation_data, evaluation_results),
            'failed_queries': failed_queries
        }

        # Save comprehensive results
        if save_results:
            results_file = output_dir / f"comprehensive_rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Comprehensive results saved to: {results_file}")

        # Print summary
        self._print_evaluation_summary(comprehensive_results)

        return comprehensive_results

    def _analyze_category_performance(
        self,
        evaluation_data: List[Dict[str, Any]],
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze performance by query category

        Args:
            evaluation_data: List of query results
            evaluation_results: RAG evaluation results

        Returns:
            Category performance breakdown
        """
        category_stats = {}

        for i, item in enumerate(evaluation_data):
            category = item['category']

            if category not in category_stats:
                category_stats[category] = {
                    'total_queries': 0,
                    'successful_queries': 0,
                    'queries': []
                }

            category_stats[category]['total_queries'] += 1
            if item['success']:
                category_stats[category]['successful_queries'] += 1
                category_stats[category]['queries'].append(item['query'])

        # Calculate success rates
        for category, stats in category_stats.items():
            stats['success_rate'] = stats['successful_queries'] / stats['total_queries']

        return category_stats

    def _print_evaluation_summary(self, results: Dict[str, Any]) -> None:
        """
        Print human-readable evaluation summary

        Args:
            results: Comprehensive evaluation results
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE RAG EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info("")

        # Overall statistics
        summary = results['summary']
        logger.info(f"Total Queries: {summary['total_queries']}")
        logger.info(f"Successful Queries: {summary['successful_queries']}")
        logger.info(f"Failed Queries: {summary['failed_queries']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1%}")
        logger.info(f"Total Time: {summary['total_evaluation_time']:.2f}s")
        logger.info("")

        # RAG metrics
        rag_metrics = results.get('rag_metrics', {})
        if 'overall_average_quality' in rag_metrics:
            logger.info(f"Overall RAG Quality Score: {rag_metrics['overall_average_quality']:.3f}")
            logger.info("")

        # RAGAS metrics
        if 'ragas_metrics' in rag_metrics:
            logger.info("RAGAS Framework Metrics:")
            for metric, value in rag_metrics['ragas_metrics'].items():
                logger.info(f"  {metric}: {value:.3f}")
            logger.info("")

        # Multi-modal metrics
        if 'multimodal_metrics' in rag_metrics:
            logger.info("Multi-Modal RAG Metrics:")
            for metric, value in rag_metrics['multimodal_metrics'].items():
                logger.info(f"  {metric}: {value:.3f}")
            logger.info("")

        # Category breakdown
        logger.info("Category Breakdown:")
        for category, stats in results['category_breakdown'].items():
            logger.info(f"  {category}: {stats['successful_queries']}/{stats['total_queries']} "
                       f"({stats['success_rate']:.1%})")
        logger.info("")

        # Success criteria assessment
        self._assess_success_criteria(rag_metrics)

        logger.info("=" * 80)

    def _assess_success_criteria(self, rag_metrics: Dict[str, Any]) -> None:
        """
        Assess whether evaluation meets success criteria

        Args:
            rag_metrics: RAG metrics results
        """
        logger.info("Success Criteria Assessment:")

        overall_quality = rag_metrics.get('overall_average_quality', 0.0)

        # Check against success criteria from plan
        if overall_quality >= 0.8:
            logger.info("  ✅ Overall RAG Quality: EXCELLENT (≥0.8)")
        elif overall_quality >= 0.7:
            logger.info("  ✓ Overall RAG Quality: GOOD (≥0.7)")
        elif overall_quality >= 0.6:
            logger.info("  ⚠️  Overall RAG Quality: ACCEPTABLE (≥0.6)")
        else:
            logger.info("  ❌ Overall RAG Quality: NEEDS IMPROVEMENT (<0.6)")

        # RAGAS metrics
        if 'ragas_metrics' in rag_metrics:
            ragas = rag_metrics['ragas_metrics']
            faithfulness = ragas.get('ragas_average_faithfulness', 0.0)
            answer_relevance = ragas.get('ragas_average_answer_relevancy', 0.0)
            context_relevance = ragas.get('ragas_average_context_relevancy', 0.0)

            if faithfulness >= 0.8:
                logger.info("  ✅ Faithfulness: EXCELLENT (≥0.8)")
            elif faithfulness >= 0.7:
                logger.info("  ✓ Faithfulness: GOOD (≥0.7)")
            else:
                logger.info(f"  ⚠️  Faithfulness: {faithfulness:.3f} (target: ≥0.8)")

            if answer_relevance >= 0.85:
                logger.info("  ✅ Answer Relevance: EXCELLENT (≥0.85)")
            elif answer_relevance >= 0.75:
                logger.info("  ✓ Answer Relevance: GOOD (≥0.75)")
            else:
                logger.info(f"  ⚠️  Answer Relevance: {answer_relevance:.3f} (target: ≥0.85)")

        # Multi-modal metrics
        if 'multimodal_metrics' in rag_metrics:
            multimodal = rag_metrics['multimodal_metrics']
            cross_modal = multimodal.get('multimodal_average_cross_modal_consistency', 0.0)

            if cross_modal >= 0.85:
                logger.info("  ✅ Cross-Modal Consistency: EXCELLENT (≥0.85)")
            elif cross_modal >= 0.75:
                logger.info("  ✓ Cross-Modal Consistency: GOOD (≥0.75)")
            else:
                logger.info(f"  ⚠️  Cross-Modal Consistency: {cross_modal:.3f} (target: ≥0.85)")


def main():
    """Main execution function"""
    try:
        # Initialize test runner
        test_runner = ComprehensiveRAGTestRunner()

        # Run comprehensive evaluation
        results = test_runner.run_full_evaluation(
            save_results=True,
            output_dir=Path("data/evaluation_results")
        )

        logger.info("✅ Comprehensive RAG Evaluation completed successfully!")

        # Check if results meet quality thresholds
        overall_quality = results.get('rag_metrics', {}).get('overall_average_quality', 0.0)

        if overall_quality >= 0.8:
            logger.info("🎯 EXCELLENT: System meets high quality standards!")
            return 0
        elif overall_quality >= 0.7:
            logger.info("✓ GOOD: System meets quality standards!")
            return 0
        else:
            logger.warning("⚠️ System needs improvement to meet quality standards")
            return 1

    except Exception as e:
        logger.error(f"❌ Comprehensive evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)