# Multi-Modal RAG System

**Project Duration:** 14 weeks
**Team Size:** 4 people
**Goal:** Build and publish a multi-modal educational RAG system with 3 novel contributions

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that works with both PDF documents and video lectures. The system includes three novel contributions:

1. **Timestamp-Aware Video RAG** - Smart video chunking at topic boundaries
2. **Temporal Coherence** - Maintaining logical flow in multi-segment retrieval
3. **Cross-Modal Reranking** - Adaptive modality selection based on query features

## Project Structure

```
multi-modal-rag/
├── README.md                          # Project overview
├── tasks.md                           # Task checklist
├── rules.md                           # Team rules & guidelines
├── requirements.txt                   # All dependencies
├── setup/                             # Installation scripts
├── config/                            # Configuration files
├── src/                               # Main source code
│   ├── base_rag/                      # Base RAG system
│   ├── contributions/                 # Novel contributions
│   ├── evaluation/                    # Evaluation infrastructure
│   └── utils/                         # Shared utilities
├── video-processing/                  # Kaggle video processing
├── data/                              # Data storage
├── models/                            # Trained models
├── indices/                           # FAISS indices
├── evaluation/                        # Evaluation data
├── notebooks/                         # Jupyter notebooks
├── tests/                             # Unit tests
├── scripts/                           # Utility scripts
└── docs/                              # Documentation
```

## Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Conda (recommended) or virtualenv

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd multi-modal-rag
```

2. Create virtual environment
```bash
conda create -n rag python=3.10
conda activate rag
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
```bash
cp .env.template .env
# Edit .env and add your API keys
```

5. Test installation
```bash
python -c "import src; print('Installation successful')"
```

## Usage

### Basic Query

```python
from src.base_rag.retriever import MultiModalRetriever
from src.base_rag.llm_generator import LLMGenerator

# Initialize retriever
retriever = MultiModalRetriever()

# Query the system
query = "What is a convolutional neural network?"
results = retriever.retrieve(query, k=5)

# Generate response
generator = LLMGenerator()
response = generator.generate(query, results)
print(response)
```

### Processing PDFs

```python
from src.base_rag.pdf_processor import PDFProcessor

processor = PDFProcessor()
documents = processor.process_pdf("path/to/pdf.pdf")
chunks = processor.chunk_documents(documents)
```

### Processing Videos

Video processing is done on Kaggle. See `video-processing/README.md` for details.

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Code Style

```bash
# Format code
black src/

# Check linting
flake8 src/
```

## Team Guidelines

See `rules.md` for complete team rules and guidelines. Key points:

- NO emojis in code or documentation
- Follow PEP 8 style guide
- Write tests for all modules
- Document all functions with docstrings
- Use type hints for function signatures
- No hardcoded values (use config files)

## Phase 1 Status

Current focus: Base System Setup (Weeks 1-2)

- Environment setup
- PDF processing pipeline
- Video processing pipeline
- Embedding generation
- Vector database setup
- Basic retrieval system
- LLM integration

## Contributing

This is a team project. All contributions should:

1. Be on a feature branch
2. Include tests
3. Be documented
4. Follow code style guidelines
5. Pass all tests before merge

## License

[To be determined]

## Questions?

See `tasks.md` for detailed task breakdown or create a GitHub issue.
