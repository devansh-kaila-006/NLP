"""
Simplified RAG Evaluation Script
Evaluates the multi-modal RAG system with custom metrics (avoiding RAGAS dependency issues)
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
sys.path.insert(0, str(src_path.parent))

from src.pipeline.unified_multimodal_pipeline import UnifiedMultiModalRAGPipeline
from src.evaluation.metrics.multimodal_metrics import MultiModalRAGEvaluator
from test_query_set import FLAT_TEST_QUERIES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimplifiedRAGTestRunner:
    """
    Simplified test runner for RAG evaluation focusing on custom multi-modal metrics
    """

    def __init__(self):
        """Initialize test runner"""
        logger.info("Initializing Simplified RAG Test Runner")

        # Initialize pipeline
        logger.info("Loading Multi-Modal RAG Pipeline...")
        self.pipeline = UnifiedMultiModalRAGPipeline(use_reranker=True, include_aman=True)

        # Initialize custom multi-modal evaluator
        logger.info("Initializing Multi-Modal RAG Evaluator...")
        self.multimodal_evaluator = MultiModalRAGEvaluator()

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

            # Extract sources with metadata
            sources = []

            # Process video chunks
            for chunk in result.get('retrieved_video_chunks', []):
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
                'sources': sources,
                'success': True,
                'error': None,
                'video_chunks_used': result.get('video_chunks_used', 0),
                'pdf_chunks_used': result.get('pdf_chunks_used', 0),
                'aman_chunks_used': result.get('aman_chunks_used', 0)
            }

        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            return {
                'query': query,
                'answer': '',
                'sources': [],
                'success': False,
                'error': str(e),
                'video_chunks_used': 0,
                'pdf_chunks_used': 0,
                'aman_chunks_used': 0
            }

    def run_evaluation(self, num_queries: int = 20, save_results: bool = True) -> Dict[str, Any]:
        """
        Run RAG evaluation on a subset of queries

        Args:
            num_queries: Number of queries to evaluate (default: 20 for faster testing)
            save_results: Whether to save results to file

        Returns:
            Dictionary with evaluation results
        """
        logger.info("=" * 80)
        logger.info(f"SIMPLIFIED RAG EVALUATION - {num_queries} QUERY TEST")
        logger.info("=" * 80)
        logger.info(f"Started at: {datetime.now().isoformat()}")

        start_time = time.time()

        # Limit queries for faster evaluation
        test_queries = FLAT_TEST_QUERIES[:num_queries]

        # Process queries
        evaluation_data = []
        multimodal_results = []

        for i, item in enumerate(test_queries):
            query = item["query"]
            category = item["category"]

            logger.info(f"Processing query {i+1}/{len(test_queries)} [{category}]: {query[:60]}...")

            # Process query
            query_result = self.process_query_with_metadata(query)

            # Add category
            query_result['category'] = category

            # Evaluate with multi-modal metrics
            if query_result['success']:
                try:
                    multimodal_result = self.multimodal_evaluator.evaluate_all_metrics(
                        answer=query_result['answer'],
                        sources=query_result['sources'],
                        query=query_result['query']
                    )
                    multimodal_results.append(multimodal_result)
                    query_result['multimodal_metrics'] = multimodal_result.to_dict()
                except Exception as e:
                    logger.warning(f"Multi-modal evaluation failed: {e}")
                    query_result['multimodal_metrics'] = None

            evaluation_data.append(query_result)

            # Log progress
            if (i + 1) % 5 == 0:
                logger.info(f"Progress: {i + 1}/{len(test_queries)} queries processed")

        total_time = time.time() - start_time

        # Calculate aggregate metrics
        successful_queries = [q for q in evaluation_data if q['success']]
        failed_queries = [q for q in evaluation_data if not q['success']]

        # Multi-modal metrics aggregation
        if multimodal_results:
            aggregate_metrics = {
                'cross_modal_consistency': sum(r.cross_modal_consistency for r in multimodal_results) / len(multimodal_results),
                'temporal_coherence_quality': sum(r.temporal_coherence_quality for r in multimodal_results) / len(multimodal_results),
                'multimodal_context_utilization': sum(r.multimodal_context_utilization for r in multimodal_results) / len(multimodal_results),
                'source_diversity': sum(r.source_diversity for r in multimodal_results) / len(multimodal_results),
                'citation_accuracy': sum(r.citation_accuracy for r in multimodal_results) / len(multimodal_results),
                'overall_score': sum(r.get_average_score() for r in multimodal_results) / len(multimodal_results)
            }
        else:
            aggregate_metrics = {
                'cross_modal_consistency': 0.0,
                'temporal_coherence_quality': 0.0,
                'multimodal_context_utilization': 0.0,
                'source_diversity': 0.0,
                'citation_accuracy': 0.0,
                'overall_score': 0.0
            }

        # Compile results
        results = {
            'summary': {
                'total_queries': len(test_queries),
                'successful_queries': len(successful_queries),
                'failed_queries': len(failed_queries),
                'success_rate': len(successful_queries) / len(test_queries),
                'total_evaluation_time': total_time,
                'timestamp': datetime.now().isoformat()
            },
            'multimodal_metrics': aggregate_metrics,
            'category_breakdown': self._analyze_category_performance(evaluation_data),
            'failed_queries': [q['query'] for q in failed_queries]
        }

        # Save results
        if save_results:
            output_dir = Path("data/evaluation_results")
            output_dir.mkdir(parents=True, exist_ok=True)

            results_file = output_dir / f"simplified_rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'results': results,
                    'individual_queries': evaluation_data
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {results_file}")

        # Print summary
        self._print_evaluation_summary(results)

        return results

    def _analyze_category_performance(self, evaluation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by query category"""
        category_stats = {}

        for item in evaluation_data:
            category = item['category']

            if category not in category_stats:
                category_stats[category] = {
                    'total_queries': 0,
                    'successful_queries': 0,
                    'total_video_chunks': 0,
                    'total_pdf_chunks': 0,
                    'total_aman_chunks': 0
                }

            category_stats[category]['total_queries'] += 1
            if item['success']:
                category_stats[category]['successful_queries'] += 1
                category_stats[category]['total_video_chunks'] += item.get('video_chunks_used', 0)
                category_stats[category]['total_pdf_chunks'] += item.get('pdf_chunks_used', 0)
                category_stats[category]['total_aman_chunks'] += item.get('aman_chunks_used', 0)

        # Calculate success rates and averages
        for category, stats in category_stats.items():
            stats['success_rate'] = stats['successful_queries'] / stats['total_queries']
            stats['avg_video_chunks'] = stats['total_video_chunks'] / stats['successful_queries'] if stats['successful_queries'] > 0 else 0
            stats['avg_pdf_chunks'] = stats['total_pdf_chunks'] / stats['successful_queries'] if stats['successful_queries'] > 0 else 0
            stats['avg_aman_chunks'] = stats['total_aman_chunks'] / stats['successful_queries'] if stats['successful_queries'] > 0 else 0

        return category_stats

    def _print_evaluation_summary(self, results: Dict[str, Any]) -> None:
        """Print human-readable evaluation summary"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("RAG EVALUATION SUMMARY")
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

        # Multi-modal metrics
        metrics = results.get('multimodal_metrics', {})
        logger.info("Multi-Modal RAG Metrics:")
        logger.info(f"  Cross-Modal Consistency: {metrics['cross_modal_consistency']:.3f}")
        logger.info(f"  Temporal Coherence Quality: {metrics['temporal_coherence_quality']:.3f}")
        logger.info(f"  Multi-Modal Context Utilization: {metrics['multimodal_context_utilization']:.3f}")
        logger.info(f"  Source Diversity: {metrics['source_diversity']:.3f}")
        logger.info(f"  Citation Accuracy: {metrics['citation_accuracy']:.3f}")
        logger.info(f"  Overall Score: {metrics['overall_score']:.3f}")
        logger.info("")

        # Category breakdown
        logger.info("Category Breakdown:")
        for category, stats in results['category_breakdown'].items():
            logger.info(f"  {category}: {stats['successful_queries']}/{stats['total_queries']} "
                       f"({stats['success_rate']:.1%}) | "
                       f"V:{stats['avg_video_chunks']:.1f} P:{stats['avg_pdf_chunks']:.1f} A:{stats['avg_aman_chunks']:.1f}")
        logger.info("")

        # Success criteria assessment
        self._assess_success_criteria(metrics)

        logger.info("=" * 80)

    def _assess_success_criteria(self, metrics: Dict[str, Any]) -> None:
        """Assess whether evaluation meets success criteria"""
        logger.info("Success Criteria Assessment:")

        overall_quality = metrics.get('overall_score', 0.0)

        if overall_quality >= 0.8:
            logger.info("  ✅ Overall Multi-Modal Quality: EXCELLENT (≥0.8)")
        elif overall_quality >= 0.7:
            logger.info("  ✓ Overall Multi-Modal Quality: GOOD (≥0.7)")
        elif overall_quality >= 0.6:
            logger.info("  ⚠️  Overall Multi-Modal Quality: ACCEPTABLE (≥0.6)")
        else:
            logger.info("  ❌ Overall Multi-Modal Quality: NEEDS IMPROVEMENT (<0.6)")

        # Individual metrics
        cross_modal = metrics.get('cross_modal_consistency', 0.0)
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
        test_runner = SimplifiedRAGTestRunner()

        # Run evaluation with 20 queries (adjustable)
        results = test_runner.run_evaluation(
            num_queries=20,  # Start with 20 for faster testing
            save_results=True
        )

        logger.info("✅ RAG Evaluation completed successfully!")

        # Check if results meet quality thresholds
        overall_quality = results.get('multimodal_metrics', {}).get('overall_score', 0.0)

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
        logger.error(f"❌ Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)