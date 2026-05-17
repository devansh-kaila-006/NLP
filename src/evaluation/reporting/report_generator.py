"""
Report Generator for Multi-Modal RAG System

Generates comprehensive evaluation reports with detailed analysis,
visualizations, and actionable insights.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

from src.utils.logger import LoggerMixin


class ReportGenerator(LoggerMixin):
    """
    Generate comprehensive evaluation reports.

    Creates detailed HTML reports with statistical analysis,
    visualizations, and actionable recommendations.
    """

    def __init__(self, output_dir: Path = None):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports (default: data/evaluation/reports)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("data/evaluation/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(self, evaluation_results: Dict,
                           report_name: str = None) -> Path:
        """
        Generate comprehensive evaluation report.

        Args:
            evaluation_results: Dictionary containing all evaluation results
            report_name: Optional custom report name

        Returns:
            Path to generated report file
        """
        if report_name is None:
            report_name = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        report_path = self.output_dir / f"{report_name}.html"

        self.logger.info(f"Generating comprehensive evaluation report: {report_path}")

        # Generate HTML report
        html_content = self._generate_html_report(evaluation_results)

        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.logger.info(f"Evaluation report generated: {report_path}")

        return report_path

    def _generate_html_report(self, results: Dict) -> str:
        """
        Generate HTML report content.

        Args:
            results: Evaluation results dictionary

        Returns:
            HTML content as string
        """
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Modal RAG System - Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            min-width: 200px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .pass {{
            color: #27ae60;
            font-weight: bold;
        }}
        .fail {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.8em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Multi-Modal RAG System - Evaluation Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary">
            <h2>Executive Summary</h2>
            {self._generate_summary_section(results)}
        </div>

        {self._generate_detailed_sections(results)}

        <div class="section">
            <h2>Conclusion & Recommendations</h2>
            {self._generate_conclusions(results)}
        </div>
    </div>
</body>
</html>
"""
        return html

    def _generate_summary_section(self, results: Dict) -> str:
        """
        Generate executive summary section.

        Args:
            results: Evaluation results

        Returns:
            HTML content for summary section
        """
        html = "<h3>System Performance Overview</h3>"

        # Add key metrics
        if 'modality_prediction' in results:
            modality_results = results['modality_prediction']
            accuracy = modality_results.get('overall_accuracy', 0.0)
            validation = modality_results.get('validation', {})

            status = "PASS" if validation.get('passed', False) else "FAIL"
            status_class = "pass" if validation.get('passed', False) else "fail"

            html += f"""
            <div class="metric">
                <div class="metric-value">{accuracy:.1%}</div>
                <div class="metric-label">Modality Prediction Accuracy</div>
                <div class="{status_class}">Status: {status}</div>
            </div>
            """

        if 'retrieval_quality' in results:
            retrieval_results = results['retrieval_quality']
            precision_5 = retrieval_results.get('retrieval_metrics', {}).get('precision_at_k', {}).get('P@5', 0.0)
            validation = retrieval_results.get('validation', {})

            status = "PASS" if validation.get('passed', False) else "FAIL"
            status_class = "pass" if validation.get('passed', False) else "fail"

            html += f"""
            <div class="metric">
                <div class="metric-value">{precision_5:.1%}</div>
                <div class="metric-label">Retrieval Precision@5</div>
                <div class="{status_class}">Status: {status}</div>
            </div>
            """

        if 'temporal_coherence' in results:
            coherence_results = results['temporal_coherence']
            coherence_precision = coherence_results.get('coherence_metrics', {}).get('summary', {}).get('coherence_precision', {}).get('mean', 0.0)
            validation = coherence_results.get('validation', {})

            status = "PASS" if validation.get('passed', False) else "FAIL"
            status_class = "pass" if validation.get('passed', False) else "fail"

            html += f"""
            <div class="metric">
                <div class="metric-value">{coherence_precision:.1%}</div>
                <div class="metric-label">Temporal Coherence</div>
                <div class="{status_class}">Status: {status}</div>
            </div>
            """

        if 'performance' in results:
            perf_results = results['performance']
            query_time = perf_results.get('query_time_stats', {}).get('mean', 0.0)
            validation = perf_results.get('validation', {})

            status = "PASS" if validation.get('passed', False) else "FAIL"
            status_class = "pass" if validation.get('passed', False) else "fail"

            html += f"""
            <div class="metric">
                <div class="metric-value">{query_time:.2f}s</div>
                <div class="metric-label">Average Query Time</div>
                <div class="{status_class}">Status: {status}</div>
            </div>
            """

        return html

    def _generate_detailed_sections(self, results: Dict) -> str:
        """
        Generate detailed evaluation sections.

        Args:
            results: Evaluation results

        Returns:
            HTML content for detailed sections
        """
        html = ""

        # Modality Prediction Section
        if 'modality_prediction' in results:
            html += self._generate_modality_section(results['modality_prediction'])

        # Retrieval Quality Section
        if 'retrieval_quality' in results:
            html += self._generate_retrieval_section(results['retrieval_quality'])

        # Temporal Coherence Section
        if 'temporal_coherence' in results:
            html += self._generate_coherence_section(results['temporal_coherence'])

        # Performance Section
        if 'performance' in results:
            html += self._generate_performance_section(results['performance'])

        return html

    def _generate_modality_section(self, results: Dict) -> str:
        """
        Generate modality prediction section.

        Args:
            results: Modality prediction results

        Returns:
            HTML content for modality section
        """
        accuracy_metrics = results.get('accuracy_metrics', {})

        html = """
        <div class="section">
            <h2>Modality Prediction Evaluation</h2>
            <h3>Cross-Modal Prediction Accuracy</h3>
            <p>Evaluates the system's ability to predict which content modality (video/PDF/Aman.ai)
            will provide the best answers for different query types.</p>

            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """

        # Add accuracy metrics
        overall_accuracy = results.get('overall_accuracy', 0.0)
        html += f"<tr><td>Overall Accuracy</td><td>{overall_accuracy:.2%}</td></tr>"

        # Add per-class metrics
        per_class = accuracy_metrics.get('per_class_metrics', {})
        for modality in ['video', 'pdf', 'aman']:
            if modality in per_class:
                metrics = per_class[modality]
                precision = metrics.get('precision', 0.0)
                recall = metrics.get('recall', 0.0)
                f1 = metrics.get('f1_score', 0.0)
                html += f"""
                <tr>
                    <td>{modality.capitalize()} F1-Score</td>
                    <td>{f1:.3f} (P: {precision:.3f}, R: {recall:.3f})</td>
                </tr>
                """

        html += "</table></div>"

        return html

    def _generate_retrieval_section(self, results: Dict) -> str:
        """
        Generate retrieval quality section.

        Args:
            results: Retrieval quality results

        Returns:
            HTML content for retrieval section
        """
        retrieval_metrics = results.get('retrieval_metrics', {})

        html = """
        <div class="section">
            <h2>Retrieval Quality Evaluation</h2>
            <h3>Multi-Modal Retrieval Effectiveness</h3>
            <p>Evaluates the system's ability to retrieve relevant content across PDFs,
            videos, and web content using standard information retrieval metrics.</p>

            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """

        # Add retrieval metrics
        precision_at_k = retrieval_metrics.get('precision_at_k', {})
        for k, value in precision_at_k.items():
            html += f"<tr><td>{k}</td><td>{value:.4f}</td></tr>"

        recall_at_k = retrieval_metrics.get('recall_at_k', {})
        for k, value in recall_at_k.items():
            html += f"<tr><td>{k}</td><td>{value:.4f}</td></tr>"

        ndcg_at_k = retrieval_metrics.get('ndcg_at_k', {})
        for k, value in ndcg_at_k.items():
            html += f"<tr><td>{k}</td><td>{value:.4f}</td></tr>"

        html += f"<tr><td>MAP</td><td>{retrieval_metrics.get('map', 0.0):.4f}</td></tr>"
        html += f"<tr><td>MRR</td><td>{retrieval_metrics.get('mrr', 0.0):.4f}</td></tr>"

        html += "</table></div>"

        return html

    def _generate_coherence_section(self, results: Dict) -> str:
        """
        Generate temporal coherence section.

        Args:
            results: Temporal coherence results

        Returns:
            HTML content for coherence section
        """
        coherence_metrics = results.get('coherence_metrics', {}).get('summary', {})

        html = """
        <div class="section">
            <h2>Temporal Coherence Evaluation</h2>
            <h3>Video Chunk Temporal Consistency</h3>
            <p>Evaluates the system's ability to maintain temporal consistency and
            logical progression across retrieved video chunks.</p>

            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """

        # Add coherence metrics
        for metric_name, metric_data in coherence_metrics.items():
            if isinstance(metric_data, dict) and 'mean' in metric_data:
                mean_val = metric_data['mean']
                std_val = metric_data['std']
                html += f"<tr><td>{metric_name.replace('_', ' ').title()}</td><td>{mean_val:.4f} (+/- {std_val:.4f})</td></tr>"

        html += "</table></div>"

        return html

    def _generate_performance_section(self, results: Dict) -> str:
        """
        Generate performance section.

        Args:
            results: Performance evaluation results

        Returns:
            HTML content for performance section
        """
        query_time_stats = results.get('query_time_stats', {})
        percentiles = results.get('query_time_percentiles', {})

        html = """
        <div class="section">
            <h2>Performance Evaluation</h2>
            <h3>System Performance Profiling</h3>
            <p>Evaluates query latency, throughput, and resource utilization to ensure
            production-ready performance.</p>

            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """

        # Add query time statistics
        html += f"<tr><td>Mean Query Time</td><td>{query_time_stats.get('mean', 0.0):.3f}s</td></tr>"
        html += f"<tr><td>Std Query Time</td><td>{query_time_stats.get('std', 0.0):.3f}s</td></tr>"
        html += f"<tr><td>Min Query Time</td><td>{query_time_stats.get('min', 0.0):.3f}s</td></tr>"
        html += f"<tr><td>Max Query Time</td><td>{query_time_stats.get('max', 0.0):.3f}s</td></tr>"

        # Add percentiles
        html += f"<tr><td>P50 (Median)</td><td>{percentiles.get('p50', 0.0):.3f}s</td></tr>"
        html += f"<tr><td>P95</td><td>{percentiles.get('p95', 0.0):.3f}s</td></tr>"
        html += f"<tr><td>P99</td><td>{percentiles.get('p99', 0.0):.3f}s</td></tr>"

        html += "</table></div>"

        return html

    def _generate_conclusions(self, results: Dict) -> str:
        """
        Generate conclusions and recommendations.

        Args:
            results: Evaluation results

        Returns:
            HTML content for conclusions section
        """
        html = "<h3>Summary and Recommendations</h3>"
        html += "<ul>"

        # Analyze overall system performance
        passed_count = 0
        total_count = 0

        for evaluator_name, evaluator_results in results.items():
            if 'validation' in evaluator_results:
                total_count += 1
                if evaluator_results['validation'].get('passed', False):
                    passed_count += 1

        html += f"<li><strong>Overall Performance:</strong> {passed_count}/{total_count} evaluation criteria passed</li>"

        # Add specific recommendations
        if 'modality_prediction' in results:
            modality_results = results['modality_prediction']
            if not modality_results.get('validation', {}).get('passed', False):
                html += "<li><strong>Modality Prediction:</strong> Consider improving prediction algorithm or expanding training data</li>"

        if 'retrieval_quality' in results:
            retrieval_results = results['retrieval_quality']
            if not retrieval_results.get('validation', {}).get('passed', False):
                html += "<li><strong>Retrieval Quality:</strong> Consider improving embedding models or reranking effectiveness</li>"

        if 'temporal_coherence' in results:
            coherence_results = results['temporal_coherence']
            if not coherence_results.get('validation', {}).get('passed', False):
                html += "<li><strong>Temporal Coherence:</strong> Consider improving temporal graph construction or path finding</li>"

        if 'performance' in results:
            perf_results = results['performance']
            if not perf_results.get('validation', {}).get('passed', False):
                html += "<li><strong>Performance:</strong> Consider optimizing query pipeline or implementing caching</li>"

        html += "</ul>"

        return html