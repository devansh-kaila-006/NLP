    """
    Comprehensive Query Test Set for Multi-Modal RAG System
    50+ diverse queries across ML/DL/NLP/CV domains for quality evaluation
    """

    # Machine Learning Fundamentals (15 queries)
    ML_QUERIES = [
        # Conceptual Questions
        "What is machine learning and how does it differ from traditional programming?",
        "Explain the difference between supervised, unsupervised, and reinforcement learning",
        "What is overfitting and how can it be prevented?",
        "Explain the bias-variance tradeoff in machine learning",
        "What is cross-validation and why is it important?",

        # Algorithm Questions
        "How does the k-nearest neighbors algorithm work?",
        "Explain the decision tree algorithm and its advantages",
        "What is random forest and how does it improve upon decision trees?",
        "How does support vector machine (SVM) work?",
        "Explain the naive Bayes classifier and its applications",

        # Optimization Questions
        "What is gradient descent and how does it optimize machine learning models?",
        "Explain the difference between batch gradient descent, stochastic gradient descent, and mini-batch gradient descent",
        "What is the difference between L1 and L2 regularization?",
        "How does the learning rate affect model training?",
        "What is early stopping and when should it be used?"
    ]

    # Deep Learning Architecture (15 queries)
    DL_QUERIES = [
        # Architecture Questions
        "What is a neural network and how does it differ from traditional machine learning?",
        "Explain the architecture of convolutional neural networks (CNNs)",
        "What are the key components of recurrent neural networks (RNNs)?",
        "How does the transformer architecture work?",
        "What is the difference between RNNs and LSTMs?",

        # Training Questions
        "Explain backpropagation and how it works in neural networks",
        "What is the vanishing gradient problem and how can it be solved?",
        "How does batch normalization help in training deep neural networks?",
        "What is dropout and how does it prevent overfitting?",
        "Explain the difference between various activation functions (ReLU, sigmoid, tanh)",

        # Advanced Questions
        "What is transfer learning and when should it be used?",
        "Explain the concept of attention mechanism in neural networks",
        "What is self-attention and how does it work?",
        "How does the transformer architecture differ from traditional sequence models?",
        "What are the key differences between BERT and GPT models?"
    ]

    # Natural Language Processing (12 queries)
    NLP_QUERIES = [
        # Fundamentals
        "What is word embedding and why is it important for NLP?",
        "Explain the difference between Word2Vec and GloVe embeddings",
        "What is named entity recognition (NER) and how does it work?",

        # Models & Techniques
        "How does recurrent neural network language modeling work?",
        "Explain the transformer architecture for NLP tasks",
        "What is BERT and how does it improve upon previous NLP models?",
        "How does GPT generate coherent text?",

        # Applications
        "What is sentiment analysis and what are its applications?",
        "Explain the machine translation pipeline and its challenges",
        "What is question answering and how do modern QA systems work?",
        "What are the main challenges in natural language understanding?",
        "How do modern chatbots handle context and maintain conversation?"
    ]

    # Computer Vision (8 queries)
    CV_QUERIES = [
        # Fundamentals
        "What is computer vision and what are its main applications?",
        "Explain how image classification works with CNNs",
        "What is object detection and how does it differ from image classification?",

        # Techniques
        "How does semantic segmentation work in computer vision?",
        "Explain the concept of transfer learning in computer vision",
        "What is image segmentation and how does it differ from object detection?",

        # Applications
        "What is face recognition and how does it work?",
        "How do modern vision transformers compare to CNNs for image tasks?"
    ]

    # Advanced & Cross-Domain Questions (10 queries)
    ADVANCED_QUERIES = [
        # Modern AI Topics
        "What is prompt engineering and why is it important for large language models?",
        "Explain the concept of few-shot learning and its advantages",
        "What is reinforcement learning and how does it work?",

        # Cross-Domain
        "How does multi-modal learning combine different types of data?",
        "What are generative adversarial networks (GANs) and how do they work?",
        "Explain the difference between discriminative and generative models",

        # Applications
        "What are the main applications of deep learning in healthcare?",
        "How is machine learning used in autonomous vehicles?",
        "What are the ethical considerations in AI development?",
        "Explain the concept of AI safety and why it's important"
    ]

    # Test Queries with Expected Modality
    TEST_QUERIES = [
        # Should favor PDF/textbook sources
        ("Derive the backpropagation formula", "pdf"),
        ("Explain the mathematical formulation of gradient descent", "pdf"),
        ("What is the formal definition of overfitting?", "pdf"),

        # Should favor video sources
        ("Show me an example of how convolutional layers work", "video"),
        ("Demonstrate how to implement a neural network from scratch", "video"),
        ("What does the CS229 course say about logistic regression?", "video"),

        # Should favor modern AI sources (Aman.ai)
        ("What is prompt engineering in the context of large language models?", "aman"),
        ("Explain the latest developments in AI agents", "aman"),
        ("What are modern approaches to AI safety?", "aman"),

        # Should be multi-modal
        ("What is machine learning and how does it work?", "multi"),
        ("Explain deep learning and its applications", "multi"),
        ("Compare different approaches to image classification", "multi")
    ]

    # Evaluation Queries (Quality Assessment)
    EVALUATION_QUERIES = [
        # Simple factual
        "What is linear regression?",
        "What is a neural network?",
        "What is NLP?",

        # Complex analytical
        "Compare and contrast different optimization algorithms used in machine learning",
        "Analyze the trade-offs between bias and variance in model performance",
        "Evaluate the strengths and weaknesses of transformer models",

        # Technical mathematical
        "Derive the gradient descent update rule for linear regression",
        "Explain the mathematical formulation of the attention mechanism",
        "What is the computational complexity of support vector machines?",

        # Applied/practical
        "How would you apply machine learning to solve a real-world classification problem?",
        "What are the best practices for training deep neural networks?",
        "How do you handle imbalanced datasets in machine learning?"
    ]

    # Combine all queries
    ALL_TEST_QUERIES = {
        "ML": ML_QUERIES,
        "DL": DL_QUERIES,
        "NLP": NLP_QUERIES,
        "CV": CV_QUERIES,
        "Advanced": ADVANCED_QUERIES,
        "Evaluation": EVALUATION_QUERIES
    }

    # Flatten all queries for sequential testing
    FLAT_TEST_QUERIES = []
    for category, queries in ALL_TEST_QUERIES.items():
        for query in queries:
            FLAT_TEST_QUERIES.append({
                "category": category,
                "query": query
            })

    print(f"Total queries: {len(FLAT_TEST_QUERIES)}")
    print(f"Categories: {list(ALL_TEST_QUERIES.keys())}")
    print(f"ML queries: {len(ML_QUERIES)}")
    print(f"DL queries: {len(DL_QUERIES)}")
    print(f"NLP queries: {len(NLP_QUERIES)}")
    print(f"CV queries: {len(CV_QUERIES)}")
    print(f"Advanced queries: {len(ADVANCED_QUERIES)}")
    print(f"Evaluation queries: {len(EVALUATION_QUERIES)}")
