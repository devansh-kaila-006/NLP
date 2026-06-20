"""
Calculate Similarity Scores Between Generated and Ground Truth Answers
Supports BLEU, ROUGE, BERT Score, and other text similarity metrics
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from collections import defaultdict

# Import available metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ground truth answers provided by user
GROUND_TRUTH_ANSWERS = {
    "Machine Learning Fundamentals": {
        "Conceptual Questions": {
            "What is machine learning and how does it differ from traditional programming?":
                "Traditional programming requires explicit, hard-coded rules and inputs to produce outputs. Machine learning reverses this: it takes data and known outputs to train a model, which automatically discovers the underlying rules and patterns.",

            "Explain the difference between supervised, unsupervised, and reinforcement learning":
                "Supervised learning trains on labeled data (input-output pairs) to predict outcomes. Unsupervised learning analyzes unlabeled data to find hidden patterns or groupings. Reinforcement learning trains an agent to make decisions via trial-and-error using a system of rewards and penalties.",

            "What is overfitting and how can it be prevented?":
                "Overfitting occurs when a model learns the training data's noise and details too well, causing poor performance on unseen data. It can be prevented using regularization (L1/L2), cross-validation, pruning (for trees), dropout (for neural networks), gathering more data, or reducing model complexity.",

            "Explain the bias-variance tradeoff in machine learning":
                "Bias is the error from overly simplistic assumptions (underfitting). Variance is the error from extreme sensitivity to training data fluctuations (overfitting). The tradeoff represents the challenge of minimizing both simultaneously to find an optimal model that generalizes well.",

            "What is cross-validation and why is it important?":
                "Cross-validation (e.g., K-fold) splits the dataset into multiple subsets, iteratively training the model on some folds and validating it on the remaining fold. It is important because it provides a reliable, robust estimate of model performance and ensures the model isn't just memorizing a single train-test split."
        },
        "Algorithm Questions": {
            "How does the k-nearest neighbors algorithm work?":
                "KNN is a non-parametric, lazy learning algorithm that classifies a data point or predicts its value based on the majority vote or average of the 'k' closest data points in the feature space, typically measured by Euclidean distance.",

            "Explain the decision tree algorithm and its advantages":
                "Decision trees recursively split data based on feature values that maximize information gain or minimize impurity (like Gini or Entropy). Advantages include high interpretability, minimal data preprocessing required (no scaling needed), and the ability to handle both numerical and categorical data.",

            "What is random forest and how does it improve upon decision trees?":
                "Random Forest is an ensemble method that builds a 'forest' of independent decision trees trained on random subsets of data (bagging) and random subsets of features. It improves upon single decision trees by drastically reducing variance and preventing overfitting without increasing bias.",

            "How does support vector machine (SVM) work?":
                "SVM finds the optimal hyperplane in an n-dimensional space that maximizes the margin (distance) between different classes of data points. For non-linear data, it uses the 'kernel trick' to project features into higher-dimensional spaces where they become linearly separable.",

            "Explain the naive Bayes classifier and its applications":
                "Naive Bayes is a probabilistic classifier based on Bayes' Theorem, operating under the 'naive' assumption that all features are completely independent given the class label. Common applications include spam filtering, text classification, and sentiment analysis due to its speed and efficiency with high-dimensional data."
        },
        "Optimization Questions": {
            "What is gradient descent and how does it optimize machine learning models?":
                "Gradient descent is an iterative optimization algorithm used to minimize a model's loss function. It calculates the partial derivatives (gradients) of the loss function with respect to the model parameters and updates the weights in the opposite direction of the gradient to step toward the global minimum.",

            "Explain the difference between batch gradient descent, stochastic gradient descent, and mini-batch gradient descent":
                "Batch Gradient Descent computes gradients using the entire dataset at once (slow, stable). Stochastic Gradient Descent (SGD) updates parameters using one random sample at a time (fast, noisy). Mini-Batch Gradient Descent updates parameters using small, random subsets of data (balances the speed of SGD and stability of Batch).",

            "What is the difference between L1 and L2 regularization?":
                "L1 regularization (Lasso) adds the absolute values of the weights as a penalty to the loss function, which can drive weights completely to zero, performing automated feature selection. L2 regularization (Ridge) adds the squared values of the weights as a penalty, which shrinks weights close to zero but never exactly to zero, keeping all features.",

            "How does the learning rate affect model training?":
                "The learning rate determines the step size taken toward the minimum during optimization. If it is too large, training may overshoot the minimum and diverge. If it is too small, training will be incredibly slow and risks getting trapped in local minima or saddle points.",

            "What is early stopping and when should it be used?":
                "Early stopping is a regularization technique that monitors a model's performance on a validation set during training and halts the training process as soon as the validation loss begins to increase, indicating the onset of overfitting."
        }
    },
    "Deep Learning Architecture": {
        "Architecture Questions": {
            "What is a neural network and how does it differ from traditional machine learning?":
                "A neural network is a computational model inspired by biological brains, consisting of layers of interconnected nodes (neurons) that pass signals to one another. It differs from traditional ML by performing automated feature engineering, learning complex hierarchical representations directly from raw data without manual feature extraction.",

            "Explain the architecture of convolutional neural networks (CNNs)":
                "CNNs are specialized for grid-like data like images. Their architecture consists of convolutional layers (using learnable filters to extract local spatial features), activation layers (usually ReLU), pooling layers (downsampling to reduce spatial dimensions and achieve translation invariance), and fully connected layers at the end for classification.",

            "What are the key components of recurrent neural networks (RNNs)?":
                "The key components of an RNN are its recurrent hidden states and a feedback loop. This architecture allows sequential information to persist, mapping an input sequence to a hidden state vector that updates at each time step using both current input and previous hidden memory.",

            "How does the transformer architecture work?":
                "The Transformer processes sequential data entirely in parallel rather than recurrently. It relies heavily on the self-attention mechanism to model global dependencies between inputs, utilizing an encoder-decoder stack, positional encodings (to retain sequence order), and multi-head attention to dynamically weigh the importance of different tokens.",

            "What is the difference between RNNs and LSTMs?":
                "Standard RNNs have short-term memory due to vanishing gradients over long sequences. LSTMs (Long Short-Term Memory) introduce a complex internal cell state regulated by three gating mechanisms—forget, input, and output gates—allowing them to actively maintain and track long-term dependencies across long sequences."
        },
        "Training Questions": {
            "Explain backpropagation and how it works in neural networks":
                "Backpropagation is the algorithm used to train neural networks. It computes the gradient of the loss function with respect to each weight by applying the calculus chain rule, propagating errors backward from the output layer through the hidden layers to update the weights via an optimizer.",

            "What is the vanishing gradient problem and how can it be solved?":
                "The vanishing gradient problem occurs when gradients shrink exponentially as they backpropagate through deep layers, leaving early layers virtually untrained. It can be solved by using non-saturating activation functions (like ReLU), proper weight initialization (He/Xavier), residual connections (ResNets), or Batch Normalization.",

            "How does batch normalization help in training deep neural networks?":
                "Batch Normalization normalizes the inputs of each layer across a mini-batch to have a mean of zero and a variance of one. This stabilizes training by mitigating internal covariate shift, allowing for higher learning rates, smoother gradient flow, and acting as a mild regularizer.",

            "What is dropout and how does it prevent overfitting?":
                "Dropout randomly deactivates a specified percentage of neurons during each training step. This forces the network to learn redundant representations and prevents co-adaptation of features, ensuring no single neuron carries the sole weight of a specific feature, which improves generalization.",

            "Explain the difference between various activation functions (ReLU, sigmoid, tanh)":
                "Sigmoid maps inputs to a (0, 1) range, causing vanishing gradients at extremes. Tanh maps inputs to a (-1, 1) range, zero-centered but still prone to vanishing gradients. ReLU outputs 0 for negative inputs and returns the input for positive values; it computes incredibly fast and resolves the vanishing gradient problem for positive values."
        },
        "Advanced Questions": {
            "What is transfer learning and when should it be used?":
                "Transfer learning leverages a model pretrained on a massive source dataset (e.g., ImageNet) and applies its learned features to a new, related target task. It should be used when target data is limited, training time is constrained, or the source and target tasks share low-level feature similarities.",

            "Explain the concept of attention mechanism in neural networks":
                "The attention mechanism allows a network to dynamically focus on specific, highly relevant parts of an input sequence when generating an output, rather than compressing an entire input sequence into a single, fixed-length context vector.",

            "What is self-attention and how does it work?":
                "Self-attention relates different positions of a single sequence to compute a representation of the same sequence. It creates three vectors for each token—Query, Key, and Value—and computes attention scores by taking the dot product of the Queries and Keys to determine how much focus to put on every other token when processing the Value vector.",

            "How does the transformer architecture differ from traditional sequence models?":
                "Traditional sequence models (RNNs/LSTMs) process data sequentially step-by-step, making them slow and prone to forgetting distant context. Transformers process entire sequences simultaneously (in parallel), achieving far superior scaling capabilities and capturing long-range context flawlessly using self-attention rather than recurrence.",

            "What are the key differences between BERT and GPT models?":
                "BERT (Bidirectional Encoder Representations from Transformers) uses an encoder-only architecture trained bidirectionally to understand context from both left and right, making it ideal for classification, NER, and QA. GPT (Generative Pre-trained Transformer) uses a decoder-only architecture trained autoregressively (left-to-right) to predict the next token, making it optimized for generative tasks."
        }
    },
    "Natural Language Processing": {
        "Fundamentals": {
            "What is word embedding and why is it important for NLP?":
                "Word embedding represents words as continuous vectors, capturing semantic meanings so similar words sit close together.",

            "Explain the difference between Word2Vec and GloVe embeddings":
                "Word2Vec predicts words based on local context windows; GloVe relies on global co-occurrence statistics across the entire text corpus.",

            "What is named entity recognition (NER) and how does it work?":
                "Named entity recognition locates and classifies mentions in text into categories like names, places, and dates using sequence labeling."
        },
        "Models & Techniques": {
            "How does recurrent neural network language modeling work?":
                "Recurrent neural network language models process text sequentially, updating an internal hidden state memory to predict the next token.",

            "Explain the transformer architecture for NLP tasks":
                "The transformer architecture processes text simultaneously using a self-attention mechanism to track relationships between all words in a sequence.",

            "What is BERT and how does it improve upon previous NLP models?":
                "BERT is a bidirectional transformer model trained by masking random words, allowing it to read context from both sides at once.",

            "How does GPT generate coherent text?":
                "GPT generates text autoregressively by predicting the next token in a sequence based on the context of all previous tokens."
        },
        "Applications": {
            "What is sentiment analysis and what are its applications?":
                "Sentiment analysis identifies emotional tones in text to classify user opinions for brand monitoring and customer feedback.",

            "Explain the machine translation pipeline and its challenges":
                "The machine translation pipeline encodes source text and decodes it to a target language, struggling with idioms and low-resource data.",

            "What is question answering and how do modern QA systems work?":
                "Question answering systems extract answers from reference text or use generative retrieval models to synthesize custom responses.",

            "What are the main challenges in natural language understanding?":
                "Main challenges in natural language understanding include word ambiguity, sarcasm, context dependencies, and a lack of common-sense knowledge.",

            "How do modern chatbots handle context and maintain conversation?":
                "Modern chatbots maintain conversation context by feeding a rolling history window of the dialogue back into the large language model."
        }
    },
    "Computer Vision": {
        "Fundamentals": {
            "What is computer vision and what are its main applications?":
                "Computer vision enables software to extract meaningful insights from digital images and videos for automation, robotics, and diagnostics.",

            "Explain how image classification works with CNNs":
                "Image classification with CNNs passes pixels through convolutional layers to extract visual features, which are then used to predict a single label.",

            "What is object detection and how does it differ from image classification?":
                "Object detection identifies individual objects within an image and outputs their specific locations using spatial bounding boxes."
        },
        "Techniques": {
            "How does semantic segmentation work in computer vision?":
                "Semantic segmentation performs pixel-level classification, assigning every single pixel in an image to a corresponding category label.",

            "Explain the concept of transfer learning in computer vision":
                "Transfer learning adapts a neural network pre-trained on a massive dataset to a new, smaller task to save training time.",

            "What is image segmentation and how does it differ from object detection?":
                "Object detection draws rough bounding boxes around items, whereas image segmentation maps the exact pixel boundaries of objects."
        },
        "Applications": {
            "What is face recognition and how does it work?":
                "Face recognition detects a face, converts its features into a unique vector embedding, and matches it against a database.",

            "How do modern vision transformers compare to CNNs for image tasks?":
                "Vision transformers split images into sequential patches, scaling better on massive datasets than CNNs by capturing global context."
        }
    },
    "Advanced Topics": {
        "Modern AI Topics": {
            "What is prompt engineering and why is it important for large language models?":
                "Prompt engineering is the process of structuring input text to guide large language models into generating accurate, specific outputs.",

            "Explain the concept of few-shot learning and its advantages":
                "Few-shot learning allows a model to generalize and perform a new task using only a tiny handful of training examples.",

            "What is reinforcement learning and how does it work?":
                "Reinforcement learning is a framework where an autonomous agent learns to make decisions by taking actions to maximize environmental rewards."
        },
        "Cross-Domain": {
            "How does multi-modal learning combine different types of data?":
                "Multi-modal learning projects different data types like text and images into a shared vector space to learn joint representations.",

            "What are generative adversarial networks (GANs) and how do they work?":
                "Generative adversarial networks pit a generator making fake data against a discriminator trying to detect it, improving overall quality.",

            "Explain the difference between discriminative and generative models":
                "Discriminative models learn boundary lines to classify data inputs, while generative models learn the actual distribution to create new data."
        },
        "Applications": {
            "What are the main applications of deep learning in healthcare?":
                "Deep learning in healthcare powers automated medical imaging analysis, genetic sequencing, risk prediction, and accelerated drug discovery.",

            "How is machine learning used in autonomous vehicles?":
                "Machine learning in autonomous vehicles tracks surrounding objects, maps environments, predicts trajectories, and plans driving paths.",

            "What are the ethical considerations in AI development?":
                "Ethical considerations in AI include algorithmic bias, data privacy, transparency, intellectual property, and automation job displacement.",

            "Explain the concept of AI safety and why it's important":
                "AI safety focuses on aligning model behavior with human values to prevent unintended harms or uncontrolled autonomous actions."
        }
    },
    "Evaluation Queries": {
        "Simple factual": {
            "What is linear regression?":
                "Linear regression is an algorithm that models a straight-line relationship between independent input variables and a continuous output.",

            "What is a neural network?":
                "A neural network is a computational pattern-recognition architecture composed of interconnected layers of nodes that process data using weights.",

            "What is NLP?":
                "NLP is a subfield of artificial intelligence focused on enabling computer systems to analyze, understand, and generate human languages."
        },
        "Complex analytical": {
            "Compare and contrast different optimization algorithms used in machine learning":
                "Optimization algorithms differ in step updates: SGD computes quickly using single points; Momentum accelerates past flat areas; Adam combines adaptive learning rates with momentum for fast convergence.",

            "Analyze the trade-offs between bias and variance in model performance":
                "The bias-variance trade-off balances underfitting and overfitting; high bias models simplify assumptions too much, while high variance models are overly sensitive to training noise.",

            "Evaluate the strengths and weaknesses of transformer models":
                "Transformers excel at capturing long-range dependencies in parallel, but suffer from high quadratic computational complexity relative to input text length."
        },
        "Technical mathematical": {
            "Derive the gradient descent update rule for linear regression":
                "The gradient descent update rule for linear regression modifies parameters by subtracting the scaled average error of predictions multiplied by the inputs.",

            "Explain the mathematical formulation of the attention mechanism":
                "The attention mechanism scales the dot-product of queries and keys, passes them through a softmax function, and multiplies the weights by the values.",

            "What is the computational complexity of support vector machines?":
                "Support vector machines feature a training computational complexity that scales quadratically or cubically with sample size, making them slow for large datasets."
        },
        "Applied/practical": {
            "How would you apply machine learning to solve a real-world classification problem?":
                "Solving a classification problem involves parsing data, separating test splits, engineering features, selecting a baseline model, and tuning hyperparameters.",

            "What are the best practices for training deep neural networks?":
                "Best practices for training deep neural networks include input normalization, proper weight initialization, dropout regularization, and learning rate decay.",

            "How do you handle imbalanced datasets in machine learning?":
                "Handling imbalanced datasets requires data techniques like SMOTE oversampling, loss adjustments like class weights, or focusing on F1 metrics over accuracy."
        }
    }
}


@dataclass
class SimilarityScoreResult:
    """Container for similarity score results"""
    query: str
    category: str
    bleu_1: float
    bleu_2: float
    bleu_3: float
    bleu_4: float
    rouge_1: float
    rouge_2: float
    rouge_l: float
    meteor: float = 0.0
    bert_score: float = 0.0
    length_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "category": self.category,
            "bleu_1": self.bleu_1,
            "bleu_2": self.bleu_2,
            "bleu_3": self.bleu_3,
            "bleu_4": self.bleu_4,
            "rouge_1": self.rouge_1,
            "rouge_2": self.rouge_2,
            "rouge_l": self.rouge_l,
            "meteor": self.meteor,
            "bert_score": self.bert_score,
            "length_ratio": self.length_ratio
        }


class SimilarityScoreCalculator:
    """Calculate various text similarity scores between generated and ground truth answers"""

    def __init__(self):
        """Initialize the calculator with required scorers"""
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        self.smoothing = SmoothingFunction().method4
        logger.info("Similarity Score Calculator initialized")

    def flatten_ground_truth(self) -> Dict[str, str]:
        """Flatten the nested ground truth dictionary to query -> answer mapping"""
        flat_gt = {}
        for category, subcategories in GROUND_TRUTH_ANSWERS.items():
            for subcategory, qa_pairs in subcategories.items():
                for query, answer in qa_pairs.items():
                    flat_gt[query] = answer
        return flat_gt

    def calculate_bleu_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate BLEU scores with different n-gram orders"""
        # Tokenize
        ref_tokens = reference.split()
        cand_tokens = candidate.split()

        # Calculate BLEU scores
        bleu_1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=self.smoothing)
        bleu_2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothing)
        bleu_3 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smoothing)
        bleu_4 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing)

        return {
            "bleu_1": bleu_1,
            "bleu_2": bleu_2,
            "bleu_3": bleu_3,
            "bleu_4": bleu_4
        }

    def calculate_rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        scores = self.rouge_scorer.score(reference, candidate)

        return {
            "rouge_1": scores["rouge1"].fmeasure,
            "rouge_2": scores["rouge2"].fmeasure,
            "rouge_l": scores["rougeL"].fmeasure
        }

    def calculate_length_ratio(self, reference: str, candidate: str) -> float:
        """Calculate the ratio of candidate length to reference length"""
        ref_len = len(reference.split())
        cand_len = len(candidate.split())

        if ref_len == 0:
            return 0.0

        return min(cand_len / ref_len, 2.0)  # Cap at 2.0

    def calculate_all_scores(
        self,
        generated_answers: List[Dict[str, Any]]
    ) -> List[SimilarityScoreResult]:
        """
        Calculate all similarity scores for generated answers

        Args:
            generated_answers: List of dicts with 'query', 'category', and 'answer' fields

        Returns:
            List of SimilarityScoreResult objects
        """
        # Flatten ground truth
        ground_truth = self.flatten_ground_truth()

        results = []
        matched_count = 0
        unmatched_queries = []

        for item in generated_answers:
            query = item.get("query", "")
            category = item.get("category", "Unknown")
            answer = item.get("answer", "")

            # Find matching ground truth
            if query in ground_truth:
                reference = ground_truth[query]
                matched_count += 1

                # Calculate all scores
                bleu_scores = self.calculate_bleu_scores(reference, answer)
                rouge_scores = self.calculate_rouge_scores(reference, answer)
                length_ratio = self.calculate_length_ratio(reference, answer)

                result = SimilarityScoreResult(
                    query=query,
                    category=category,
                    bleu_1=bleu_scores["bleu_1"],
                    bleu_2=bleu_scores["bleu_2"],
                    bleu_3=bleu_scores["bleu_3"],
                    bleu_4=bleu_scores["bleu_4"],
                    rouge_1=rouge_scores["rouge_1"],
                    rouge_2=rouge_scores["rouge_2"],
                    rouge_l=rouge_scores["rouge_l"],
                    length_ratio=length_ratio
                )
                results.append(result)
            else:
                unmatched_queries.append(query)

        logger.info(f"Matched {matched_count} queries with ground truth")
        if unmatched_queries:
            logger.warning(f"Could not find ground truth for {len(unmatched_queries)} queries:")
            for q in unmatched_queries[:5]:  # Show first 5
                logger.warning(f"  - {q[:60]}...")

        return results

    def generate_summary_report(
        self,
        results: List[SimilarityScoreResult]
    ) -> Dict[str, Any]:
        """Generate summary statistics from results"""
        if not results:
            return {}

        # Calculate averages
        avg_bleu_1 = sum(r.bleu_1 for r in results) / len(results)
        avg_bleu_2 = sum(r.bleu_2 for r in results) / len(results)
        avg_bleu_3 = sum(r.bleu_3 for r in results) / len(results)
        avg_bleu_4 = sum(r.bleu_4 for r in results) / len(results)
        avg_rouge_1 = sum(r.rouge_1 for r in results) / len(results)
        avg_rouge_2 = sum(r.rouge_2 for r in results) / len(results)
        avg_rouge_l = sum(r.rouge_l for r in results) / len(results)
        avg_length_ratio = sum(r.length_ratio for r in results) / len(results)

        # Group by category
        by_category = defaultdict(list)
        for r in results:
            by_category[r.category].append(r)

        category_stats = {}
        for cat, cat_results in by_category.items():
            category_stats[cat] = {
                "count": len(cat_results),
                "avg_bleu_4": sum(r.bleu_4 for r in cat_results) / len(cat_results),
                "avg_rouge_l": sum(r.rouge_l for r in cat_results) / len(cat_results)
            }

        return {
            "total_evaluated": len(results),
            "averages": {
                "bleu_1": avg_bleu_1,
                "bleu_2": avg_bleu_2,
                "bleu_3": avg_bleu_3,
                "bleu_4": avg_bleu_4,
                "rouge_1": avg_rouge_1,
                "rouge_2": avg_rouge_2,
                "rouge_l": avg_rouge_l,
                "length_ratio": avg_length_ratio
            },
            "by_category": category_stats
        }


def main():
    """Main function to calculate similarity scores"""
    # Load generated answers
    full_answers_file = Path("data/evaluation/results/full_generated_answers.json")

    if not full_answers_file.exists():
        logger.error(f"Full answers file not found: {full_answers_file}")
        return

    with open(full_answers_file, 'r', encoding='utf-8') as f:
        generated_data = json.load(f)

    logger.info(f"Loaded {len(generated_data)} full generated answers")

    # Initialize calculator
    calculator = SimilarityScoreCalculator()

    # Calculate scores
    results = calculator.calculate_all_scores(generated_data)

    # Generate summary
    summary = calculator.generate_summary_report(results)

    # Print results
    print("\n" + "="*60)
    print("SIMILARITY SCORE EVALUATION RESULTS")
    print("="*60)

    print(f"\nTotal queries evaluated: {summary['total_evaluated']}")

    print("\nAverage Scores:")
    print(f"  BLEU-1:  {summary['averages']['bleu_1']:.4f}")
    print(f"  BLEU-2:  {summary['averages']['bleu_2']:.4f}")
    print(f"  BLEU-3:  {summary['averages']['bleu_3']:.4f}")
    print(f"  BLEU-4:  {summary['averages']['bleu_4']:.4f}")
    print(f"  ROUGE-1: {summary['averages']['rouge_1']:.4f}")
    print(f"  ROUGE-2: {summary['averages']['rouge_2']:.4f}")
    print(f"  ROUGE-L: {summary['averages']['rouge_l']:.4f}")
    print(f"  Length Ratio: {summary['averages']['length_ratio']:.4f}")

    print("\nBy Category:")
    for cat, stats in summary['by_category'].items():
        print(f"  {cat}: {stats['count']} queries, BLEU-4: {stats['avg_bleu_4']:.4f}, ROUGE-L: {stats['avg_rouge_l']:.4f}")

    # Save detailed results
    output_file = Path("data/evaluation/results/similarity_scores.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": summary,
            "detailed_results": [r.to_dict() for r in results]
        }, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
