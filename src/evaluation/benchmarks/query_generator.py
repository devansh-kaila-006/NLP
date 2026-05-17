"""
Query Generator for Multi-Modal RAG System

Automatically generates diverse test queries covering different domains,
query types, and difficulty levels for comprehensive evaluation.
"""

from typing import Dict, List, Set
import random


class QueryGenerator:
    """
    Generate diverse test queries for multi-modal RAG evaluation.

    Creates queries across different domains (ML, DL, NLP, CV), query types
    (conceptual, mathematical, implementation), and difficulty levels.
    """

    # Domain-specific query templates
    QUERY_TEMPLATES = {
        'machine_learning': {
            'conceptual': [
                "What is {concept} in machine learning?",
                "Explain the concept of {concept}",
                "How does {concept} work in ML?",
                "What are the main applications of {concept}?"
            ],
            'mathematical': [
                "What is the mathematical formula for {concept}?",
                "Derive the {concept} equation",
                "Show the mathematical derivation of {concept}",
                "What are the key mathematical properties of {concept}?"
            ],
            'implementation': [
                "How do you implement {concept}?",
                "What are the practical steps for {concept}?",
                "Show me an example of {concept} implementation",
                "What are common challenges when implementing {concept}?"
            ]
        },
        'deep_learning': {
            'conceptual': [
                "Explain how {concept} works in deep learning",
                "What is the role of {concept} in neural networks?",
                "How does {concept} improve deep learning models?",
                "What are the advantages of using {concept}?"
            ],
            'mathematical': [
                "What is the gradient descent formula for {concept}?",
                "Derive the backpropagation equations for {concept}",
                "Show the mathematical formulation of {concept}",
                "What are the mathematical properties of {concept}?"
            ],
            'implementation': [
                "How do you implement {concept} in PyTorch?",
                "What is the code for {concept}?",
                "Show me a practical example of {concept}",
                "What are the best practices for implementing {concept}?"
            ]
        },
        'nlp': {
            'conceptual': [
                "What is {concept} in natural language processing?",
                "Explain the concept of {concept} for NLP",
                "How does {concept} improve language models?",
                "What are the applications of {concept} in NLP?"
            ],
            'mathematical': [
                "What is the attention mechanism formula for {concept}?",
                "Derive the transformer equations for {concept}",
                "Show the mathematical basis of {concept}",
                "What are the mathematical properties of {concept}?"
            ],
            'implementation': [
                "How do you implement {concept} for text processing?",
                "What is the code for {concept} in NLP?",
                "Show me a practical NLP example with {concept}",
                "What are common issues when implementing {concept}?"
            ]
        },
        'computer_vision': {
            'conceptual': [
                "What is {concept} in computer vision?",
                "Explain how {concept} works for image processing",
                "How does {concept} improve visual recognition?",
                "What are the applications of {concept} in CV?"
            ],
            'mathematical': [
                "What is the convolution formula for {concept}?",
                "Derive the mathematical foundation of {concept}",
                "Show the equations for {concept} in vision",
                "What are the mathematical properties of {concept}?"
            ],
            'implementation': [
                "How do you implement {concept} for image processing?",
                "What is the code for {concept} in CV?",
                "Show me a practical vision example with {concept}",
                "What are the best practices for implementing {concept}?"
            ]
        }
    }

    # Concept keywords for different domains
    CONCEPTS = {
        'machine_learning': [
            'gradient descent', 'linear regression', 'logistic regression',
            'decision trees', 'random forests', 'support vector machines',
            'k-means clustering', 'principal component analysis', 'bias-variance tradeoff'
        ],
        'deep_learning': [
            'neural networks', 'backpropagation', 'activation functions',
            'batch normalization', 'dropout', 'weight initialization',
            'loss functions', 'optimization algorithms', 'regularization'
        ],
        'nlp': [
            'word embeddings', 'attention mechanism', 'transformer architecture',
            'language models', 'named entity recognition', 'sentiment analysis',
            'machine translation', 'text classification', 'BERT'
        ],
        'computer_vision': [
            'convolutional neural networks', 'image segmentation',
            'object detection', 'feature extraction', 'image classification',
            'transfer learning', 'data augmentation', 'CNN architectures'
        ]
    }

    # Query types mapped to expected modalities
    MODALITY_MAPPING = {
        'conceptual': 'video',      # Explanations work best in video
        'mathematical': 'pdf',       # Formulas and derivations in PDFs
        'implementation': 'aman'     # Modern practices in Aman.ai primers
    }

    def __init__(self, seed: int = 42):
        """
        Initialize query generator.

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)

    def generate_queries(self,
                        domains: List[str] = None,
                        query_types: List[str] = None,
                        difficulty_levels: List[str] = None,
                        num_queries_per_category: int = 10) -> List[Dict]:
        """
        Generate diverse test queries.

        Args:
            domains: List of domains to include (default: all)
            query_types: List of query types to include (default: all)
            difficulty_levels: List of difficulty levels (default: all)
            num_queries_per_category: Number of queries to generate per category

        Returns:
            List of query dictionaries with metadata
        """
        if domains is None:
            domains = list(self.QUERY_TEMPLATES.keys())
        if query_types is None:
            query_types = ['conceptual', 'mathematical', 'implementation']
        if difficulty_levels is None:
            difficulty_levels = ['easy', 'medium', 'hard']

        queries = []

        for domain in domains:
            if domain not in self.QUERY_TEMPLATES:
                continue

            for query_type in query_types:
                if query_type not in self.QUERY_TEMPLATES[domain]:
                    continue

                # Generate queries for this category
                category_queries = self._generate_category_queries(
                    domain, query_type, num_queries_per_category
                )
                queries.extend(category_queries)

        # Add difficulty and metadata
        for i, query in enumerate(queries):
            query['difficulty'] = random.choice(difficulty_levels)
            query['query_id'] = f"query_{i:04d}"
            query['expected_modality'] = self.MODALITY_MAPPING.get(query['query_type'], 'video')

        return queries

    def _generate_category_queries(self, domain: str, query_type: str,
                                  num_queries: int) -> List[Dict]:
        """
        Generate queries for a specific category.

        Args:
            domain: Domain name
            query_type: Type of query
            num_queries: Number of queries to generate

        Returns:
            List of query dictionaries
        """
        queries = []
        concepts = self.CONCEPTS.get(domain, [])
        templates = self.QUERY_TEMPLATES[domain][query_type]

        for i in range(num_queries):
            concept = random.choice(concepts)
            template = random.choice(templates)

            # Generate query by filling template
            query_text = template.format(concept=concept)

            query = {
                'text': query_text,
                'domain': domain,
                'query_type': query_type,
                'concept': concept,
                'template': template
            }

            queries.append(query)

        return queries

    def generate_sequential_queries(self, num_sequences: int = 10,
                                   sequence_length: int = 5) -> List[List[Dict]]:
        """
        Generate sequential query sets for temporal coherence testing.

        Args:
            num_sequences: Number of sequential query sets to generate
            sequence_length: Number of queries in each sequence

        Returns:
            List of sequential query sets
        """
        sequences = []

        for i in range(num_sequences):
            # Choose a domain and concept for this sequence
            domain = random.choice(list(self.QUERY_TEMPLATES.keys()))
            concepts = self.CONCEPTS.get(domain, [])
            concept = random.choice(concepts)

            # Generate sequential queries that follow a logical progression
            sequence = []

            # Query progression: What -> How -> Why -> Implementation -> Applications
            progression_templates = [
                f"What is {concept}?",
                f"How does {concept} work?",
                f"Why is {concept} important?",
                f"How do you implement {concept}?",
                f"What are the applications of {concept}?"
            ]

            for j in range(min(sequence_length, len(progression_templates))):
                query_text = progression_templates[j]

                query = {
                    'text': query_text,
                    'domain': domain,
                    'query_type': ['conceptual', 'conceptual', 'conceptual',
                                  'implementation', 'conceptual'][j],
                    'concept': concept,
                    'sequence_position': j,
                    'sequence_id': i
                }

                sequence.append(query)

            sequences.append(sequence)

        return sequences

    def generate_performance_queries(self, num_queries: int = 50) -> List[Dict]:
        """
        Generate queries specifically for performance testing.

        Args:
            num_queries: Number of performance test queries

        Returns:
            List of performance test queries
        """
        # Generate diverse queries for performance testing
        queries = self.generate_queries(
            domains=None,  # All domains
            query_types=None,  # All query types
            difficulty_levels=['medium'],  # Focus on medium difficulty
            num_queries_per_category=num_queries // 12  # Distribute across categories
        )

        # Ensure we have exactly num_queries
        return queries[:num_queries]