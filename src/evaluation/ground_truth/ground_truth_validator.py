"""
Ground Truth Validator for Synthetic Reference Answers
Validates quality and consistency of generated ground truth samples
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json

# Similarity metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False

# Local imports
from .generate_ground_truth import GroundTruthSample

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results"""
    passed: bool
    overall_score: float
    quality_metrics: Dict[str, Any]
    similarity_metrics: Dict[str, Any]
    issues: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "passed": self.passed,
            "overall_score": self.overall_score,
            "quality_metrics": self.quality_metrics,
            "similarity_metrics": self.similarity_metrics,
            "issues": self.issues
        }


class GroundTruthValidator:
    """
    Validate synthetic ground truth samples for quality and consistency
    """

    def __init__(self, quality_threshold: float = 0.8):
        """
        Initialize ground truth validator

        Args:
            quality_threshold: Minimum quality threshold (0-1)
        """
        self.quality_threshold = quality_threshold

        # Initialize scorers
        self.rouge_scorer = None
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        logger.info(f"Ground Truth Validator initialized with threshold: {quality_threshold}")

    def validate_sample_quality(self, sample: GroundTruthSample) -> List[str]:
        """
        Validate quality of a single ground truth sample

        Args:
            sample: GroundTruthSample to validate

        Returns:
            List of quality issues (empty if no issues)
        """
        issues = []

        # Check for empty answers
        if not sample.reference_answer or len(sample.reference_answer.strip()) == 0:
            issues.append("Empty reference answer")

        # Check for minimum length
        word_count = sample.metadata.get("answer_word_count", 0)
        if word_count < 50:
            issues.append(f"Answer too short: {word_count} words (minimum: 50)")

        if word_count > 1000:
            issues.append(f"Answer too long: {word_count} words (maximum: 1000)")

        # Check for failed generation
        if sample.metadata.get("failed", False):
            issues.append("Generation failed")

        # Check for error in metadata
        if "error" in sample.metadata:
            issues.append(f"Generation error: {sample.metadata['error']}")

        return issues

    def calculate_similarity_scores(
        self,
        candidate: str,
        reference: str
    ) -> Dict[str, float]:
        """
        Calculate similarity scores between candidate and reference texts

        Args:
            candidate: Candidate text
            reference: Reference text

        Returns:
            Dictionary with similarity scores
        """
        scores = {}

        # ROUGE scores
        if self.rouge_scorer:
            try:
                rouge_scores = self.rouge_scorer.score(reference, candidate)
                scores["rouge1"] = rouge_scores["rouge1"].fmeasure
                scores["rouge2"] = rouge_scores["rouge2"].fmeasure
                scores["rougeL"] = rouge_scores["rougeL"].fmeasure
            except Exception as e:
                logger.warning(f"Error calculating ROUGE scores: {e}")

        return scores

    def validate_batch(
        self,
        samples: List[GroundTruthSample],
        reference_answers: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate a batch of ground truth samples

        Args:
            samples: List of GroundTruthSample objects
            reference_answers: Optional list of reference answers for similarity checking

        Returns:
            ValidationResult with comprehensive metrics
        """
        all_issues = []
        quality_metrics = {}
        similarity_metrics = {}

        total_samples = len(samples)
        successful_samples = [s for s in samples if not s.metadata.get("failed", False)]

        # Quality metrics
        word_counts = [s.metadata.get("answer_word_count", 0) for s in successful_samples]
        generation_times = [s.metadata.get("generation_time", 0) for s in successful_samples]

        quality_metrics = {
            "total_samples": total_samples,
            "successful_samples": len(successful_samples),
            "failed_samples": total_samples - len(successful_samples),
            "success_rate": len(successful_samples) / total_samples if total_samples > 0 else 0,
            "avg_word_count": sum(word_counts) / len(word_counts) if word_counts else 0,
            "avg_generation_time": sum(generation_times) / len(generation_times) if generation_times else 0,
            "min_word_count": min(word_counts) if word_counts else 0,
            "max_word_count": max(word_counts) if word_counts else 0
        }

        # Collect quality issues
        for sample in samples:
            issues = self.validate_sample_quality(sample)
            if issues:
                all_issues.extend([f"Query '{sample.query[:50]}...': {issue}" for issue in issues])

        # Similarity metrics (if reference answers provided)
        if reference_answers and len(reference_answers) == len(samples):
            rouge_scores = []

            for sample, reference in zip(samples, reference_answers):
                if not sample.metadata.get("failed", False):
                    scores = self.calculate_similarity_scores(sample.reference_answer, reference)
                    if "rougeL" in scores:
                        rouge_scores.append(scores["rougeL"])

            if rouge_scores:
                similarity_metrics["avg_rougeL"] = sum(rouge_scores) / len(rouge_scores)
                similarity_metrics["min_rougeL"] = min(rouge_scores)
                similarity_metrics["max_rougeL"] = max(rouge_scores)

        # Calculate overall score
        success_rate = quality_metrics["success_rate"]
        avg_similarity = similarity_metrics.get("avg_rougeL", 0.0)

        # Overall score combines success rate and similarity
        overall_score = (success_rate * 0.6) + (avg_similarity * 0.4)

        # Determine if validation passed
        passed = (
            success_rate >= self.quality_threshold and
            quality_metrics["avg_word_count"] >= 50 and
            len(all_issues) == 0
        )

        return ValidationResult(
            passed=passed,
            overall_score=overall_score,
            quality_metrics=quality_metrics,
            similarity_metrics=similarity_metrics,
            issues=all_issues
        )

    def validate_against_human_review(
        self,
        samples: List[GroundTruthSample],
        human_review_file: Path
    ) -> ValidationResult:
        """
        Validate synthetic ground truth against human review samples

        Args:
            samples: List of synthetic GroundTruthSample objects
            human_review_file: Path to JSON file with human-reviewed samples

        Returns:
            ValidationResult with comparison metrics
        """
        # Load human review data
        with open(human_review_file, 'r', encoding='utf-8') as f:
            human_data = json.load(f)

        human_samples = human_data.get("samples", [])

        # Map queries to human answers
        human_answers = {}
        for sample in human_samples:
            human_answers[sample["query"]] = sample["reference_answer"]

        # Get reference answers for matching queries
        reference_answers = []
        matching_samples = []

        for sample in samples:
            if sample.query in human_answers:
                reference_answers.append(human_answers[sample.query])
                matching_samples.append(sample)

        logger.info(f"Found {len(matching_samples)} matching samples for human review")

        return self.validate_batch(matching_samples, reference_answers)

    def generate_validation_report(
        self,
        validation_result: ValidationResult,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate human-readable validation report

        Args:
            validation_result: ValidationResult object
            output_path: Optional path to save report

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("GROUND TRUTH VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")

        # Overall status
        status = "✅ PASSED" if validation_result.passed else "❌ FAILED"
        report.append(f"Overall Status: {status}")
        report.append(f"Overall Score: {validation_result.overall_score:.3f}")
        report.append(f"Quality Threshold: {self.quality_threshold}")
        report.append("")

        # Quality metrics
        report.append("Quality Metrics:")
        for key, value in validation_result.quality_metrics.items():
            if isinstance(value, float):
                report.append(f"  {key}: {value:.3f}")
            else:
                report.append(f"  {key}: {value}")
        report.append("")

        # Similarity metrics
        if validation_result.similarity_metrics:
            report.append("Similarity Metrics:")
            for key, value in validation_result.similarity_metrics.items():
                report.append(f"  {key}: {value:.3f}")
            report.append("")

        # Issues
        if validation_result.issues:
            report.append(f"Issues Found ({len(validation_result.issues)}):")
            for issue in validation_result.issues:
                report.append(f"  - {issue}")
            report.append("")
        else:
            report.append("No issues detected ✅")
            report.append("")

        report.append("=" * 60)

        report_text = "\n".join(report)

        # Save report if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Validation report saved to {output_path}")

        return report_text


def create_ground_truth_validator(quality_threshold: float = 0.8) -> GroundTruthValidator:
    """
    Factory function to create ground truth validator

    Args:
        quality_threshold: Minimum quality threshold

    Returns:
        GroundTruthValidator instance
    """
    return GroundTruthValidator(quality_threshold=quality_threshold)


if __name__ == "__main__":
    # Test validator
    logging.basicConfig(level=logging.INFO)

    try:
        validator = create_ground_truth_validator()

        # Create test samples
        test_samples = [
            GroundTruthSample(
                query="What is machine learning?",
                reference_answer="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and learn from it to make predictions or decisions.",
                metadata={"answer_word_count": 45, "generation_time": 1.2}
            ),
            GroundTruthSample(
                query="Explain neural networks",
                reference_answer="Neural networks are computing systems inspired by biological neural networks that make up animal brains. They are based on a collection of connected units called artificial neurons, which loosely model the neurons in a biological brain.",
                metadata={"answer_word_count": 38, "generation_time": 1.0}
            )
        ]

        # Validate batch
        result = validator.validate_batch(test_samples)

        # Generate report
        report = validator.generate_validation_report(result)
        print(report)

    except Exception as e:
        print(f"Error testing validator: {e}")