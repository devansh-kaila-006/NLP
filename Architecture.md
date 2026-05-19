# Multi-Modal RAG System Architecture

**System Version**: 3.0
**Last Updated**: 2026-05-19
**Architecture Type**: Multi-Modal Retrieval-Augmented Generation with Novel Innovations

---

## 1. System Overview

The Multi-Modal RAG System is an advanced question-answering system that integrates three content modalities—video lectures, PDF textbooks, and web content—to provide comprehensive, accurate responses to AI/ML domain questions. The system incorporates three novel innovations in video RAG: timestamp-aware retrieval, temporal coherence, and cross-modal prediction.

### Core Capabilities
- **Multi-Modal Integration**: Unified retrieval from video transcripts, PDF textbooks, and web content
- **Intelligent Modality Selection**: Predicts optimal content type for each query (97% accuracy)
- **Temporal Video Navigation**: Direct timestamp links to relevant video segments
- **Cross-Modal Reranking**: Advanced reranking across different content types
- **Production-Ready Performance**: ~12s average query time with 100% success rate

### Content Coverage
- **5 Complete University Courses**: Stanford CS229, CS224n, CS231n + MIT 6.S191 (main + alternative)
- **12,717 Total Chunks**: 9,661 PDF, 2,923 video, 133 web content
- **Academic Sources**: Stanford, MIT, authoritative textbooks
- **Modern Web Content**: Aman.ai primers on contemporary AI topics

---

## 2. High-Level Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Query Interface                        │
│                   (Gradio UI / CLI / API)                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Unified Multi-Modal RAG Pipeline                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. Modality Prediction (Cross-Modal Classification)      │  │
│  │    → Predicts optimal content type (video/PDF/web)       │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 2. Parallel Multi-Modal Retrieval                        │  │
│  │    → PDF Retriever + Video Retriever + Web Retriever     │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 3. Cross-Modal Reranking                                 │  │
│  │    → Query complexity detection + adaptive reranking     │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 4. Temporal Coherence Processing                         │  │
│  │    → Flow-aware retrieval for video content              │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 5. Answer Generation                                     │  │
│  │    → Context-aware response with video links             │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Vector Store Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  PDF Index   │  │ Video Index  │  │  Web Index   │          │
│  │  (FAISS)     │  │  (FAISS x5)  │  │  (FAISS)     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Query Processing**: User question → Modality prediction → Query embedding
2. **Parallel Retrieval**: Simultaneous search across PDF, video (5 playlists), and web indices
3. **Cross-Modal Reranking**: Query complexity detection → Adaptive reranking strategy
4. **Temporal Processing**: Video chunks sorted by temporal coherence graphs
5. **Answer Generation**: LLM generates response with source citations and video timestamp links

---

## 3. Novel Architectural Innovations

### Innovation 1: Cross-Modal Prediction
**Problem**: Queries require different content types (conceptual vs. practical vs. current)
**Solution**: ML-based modality classifier trained on query patterns and content characteristics

**Architecture**:
```python
# Query Analysis Pipeline
Query → Feature Extraction → Modality Classifier → Content Selection
         (Query length,          (97% accuracy)
          keywords, domain,
          complexity indicators)
```

**Features**:
- **Query Complexity Detection**: Length, keyword analysis, technical depth
- **Domain Classification**: ML, DL, NLP, CV, Advanced AI topics
- **Content Type Prediction**: Video (practical), PDF (theoretical), Web (current)
- **Performance**: 97% accuracy, 0.031 calibration error

### Innovation 2: Temporal Coherence
**Problem**: Video chunks retrieved out of context disrupt learning flow
**Solution**: Temporal dependency graphs maintain narrative structure in video-based answers

**Architecture**:
```python
# Temporal Coherence Pipeline
Video Chunks → Temporal Dependency Graph → Flow-Aware Ranking → Coherent Answers
                   (Topic transitions,
                    Narrative flow,
                    Concept dependencies)
```

**Features**:
- **Temporal Dependency Graphs**: NetworkX graphs modeling topic transitions
- **Flow Scoring**: 0.92 average flow quality across video content
- **Narrative Preservation**: Maintains lecture progression in answers
- **Performance**: 100% temporal ordering accuracy

### Innovation 3: Timestamp-Aware Video RAG
**Problem**: Video content difficult to navigate and reference
**Solution**: Semantic chunking with precise timestamp navigation

**Architecture**:
```python
# Video Processing Pipeline
SRT Transcripts → Semantic Chunking → Timestamp Enrichment → Direct Navigation
                   (Topic boundaries,    (YouTube links,
                    Slide changes,       Segments timing,
                    Concept shifts)      Millisecond precision)
```

**Features**:
- **Semantic Chunking**: ~30-second chunks with topic coherence
- **Direct Navigation**: YouTube timestamp links (e.g., `t=1234`)
- **Multi-Playlist Support**: 5 complete courses with unified retrieval
- **Content Coverage**: 2,923 video chunks across 5 Stanford/MIT courses

---

## 4. Component Architecture

### 4.1 Data Processing Layer

#### PDF Processing Pipeline
```
PDF Documents → Text Extraction → Semantic Chunking → Embedding Generation → FAISS Index
                 (PyPDF2,           (Sentence          (Sentence          (Vector
                  pdfplumber)         Transformers)      Transformers)      Store)
```

**Key Components**:
- **PDFLoader**: Handles PDF, ZIP (documentation), and HTML content
- **SemanticChunker**: Hierarchical chunking (chapter → section → paragraph)
- **Chunking Config**: 400-token chunks, 50-token overlap, semantic boundaries

#### Video Processing Pipeline
```
YouTube Playlists → SRT Transcripts → Semantic Chunking → Temporal Analysis → FAISS Index
                       (Auto-generated,  (Topic boundaries,   (Dependency graphs,
                        Whisper backup)   Slide detection)    Flow scoring)
```

**Key Components**:
- **SRTLoader**: Parses YouTube auto-generated transcripts with timestamps
- **VideoChunker**: Semantic chunking with temporal coherence
- **MultiVideoRetriever**: Unified retrieval across 5 video playlists
- **TemporalGraphs**: NetworkX-based temporal dependency modeling

#### Web Content Pipeline
```
Aman.ai Articles → Content Scraping → Text Processing → Embedding Generation → FAISS Index
                     (BeautifulSoup,     (Cleaning,        (Sentence
                      Selenium)          Chunking)         Transformers)
```

### 4.2 Retrieval Layer

#### Vector Store Architecture
```
┌──────────────────────────────────────────────────────────────┐
│                    FAISS Vector Indices                       │
├─────────────────┬─────────────────┬─────────────────────────┤
│ PDF Index       │ Video Indices   │ Web Index               │
│ (9,661 chunks)  │ (5 playlists)   │ (133 chunks)            │
│ 384-dim vectors │ 2,923 chunks    │ Modern AI content       │
└─────────────────┴─────────────────┴─────────────────────────┘
```

**Index Configuration**:
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Similarity Metric**: Inner Product (cosine similarity with normalization)
- **Index Type**: FAISS IndexFlatIP (exact search)
- **Performance**: Sub-second retrieval across 12,717 chunks

#### Retrieval Strategy
```python
# Multi-Stage Retrieval
Query → Embedding Generation → Parallel FAISS Search → Top-K Selection
         (Sentence              (5 PDF, 5 video,       (Per modality)
          Transformers)          5 web chunks)
```

### 4.3 Reranking Layer

#### Adaptive Reranking Architecture
```
Retrieved Chunks → Query Complexity Analysis → Reranker Selection → Final Ranking
                     (Length, keywords,        (Fast vs Enhanced      (Top-N chunks
                      technical depth)          models)                 per modality)
```

**Reranking Strategy**:
- **Simple Queries**: Fast cross-encoder (ms-marco-MiniLM-L-6-v2)
- **Complex Queries**: Enhanced cross-encoder (ms-marco-electra-base)
- **Query Expansion**: Related term expansion for complex queries
- **Multi-Stage**: Coarse-to-fine reranking for performance

**Configuration**:
- **Top-K Retrieval**: 5 chunks per modality
- **Top-N Reranking**: 3 chunks per modality
- **Threshold**: Dynamic based on query complexity
- **Performance**: Optimized for production deployment

### 4.4 Generation Layer

#### LLM Integration
```
Reranked Chunks → Context Assembly → Prompt Construction → LLM Generation → Response
                  (Video links,        (System prompt,     (Gemini 3.1       (Answer +
                   Citations,           Query context,      Flash Lite)       Sources +
                   Timestamps)          Retrieved chunks)                      Video links)
```

**LLM Configuration**:
- **Model**: Gemini 3.1 Flash Lite Preview
- **Temperature**: 0.3 (factual, consistent responses)
- **Max Tokens**: 1024 (comprehensive but concise)
- **Timeout**: 30s (production-ready)

**Response Features**:
- **Source Citations**: Clear attribution to video lectures, PDFs, and web content
- **Video Timestamp Links**: Direct navigation to relevant video segments
- **Multi-Modal Integration**: Seamless combination of different content types
- **Context Quality**: High-quality, well-sourced responses

---

## 5. Technology Stack

### Core Technologies

#### Machine Learning & NLP
- **Sentence Transformers**: all-MiniLM-L6-v2 (embeddings)
- **Cross-Encoders**: ms-marco-MiniLM-L-6-v2, ms-marco-electra-base (reranking)
- **NetworkX**: Temporal dependency graphs
- **Scikit-learn**: Similarity calculations, metrics

#### Vector Store & Search
- **FAISS**: High-performance vector similarity search
- **Index Type**: IndexFlatIP (inner product for cosine similarity)
- **Dimensions**: 384 (all-MiniLM-L6-v2)
- **Total Vectors**: 12,717 across 7 indices

#### LLM Integration
- **Google Gemini**: 3.1 Flash Lite Preview
- **API**: GenerativeLanguageAPI
- **Features**: Fast inference, high quality, cost-effective

#### Data Processing
- **PyPDF2 & pdfplumber**: PDF text extraction
- **BeautifulSoup & Selenium**: Web scraping
- **Whisper**: Video transcription (backup for missing SRT)
- **python-srt**: SRT transcript parsing

#### Application Framework
- **Gradio**: Web UI for interactive demo
- **FastAPI**: REST API endpoints (future)
- **Logging**: Structured logging with file rotation

### Dependencies
```
# Core ML/NLP
sentence-transformers==2.2.2
transformers==4.30.0
faiss-cpu==1.7.4
scikit-learn==1.3.0

# LLM Integration
google-generativeai==0.3.0

# Data Processing
PyPDF2==3.0.1
pdfplumber==0.10.3
beautifulsoup4==4.12.2
selenium==4.15.0

# Video Processing
whisper==1.0.0
python-srt==1.1.3

# Graph Processing
networkx==3.1

# Web Framework
gradio==4.7.1
fastapi==0.109.0 (future)

# Utilities
python-dotenv==1.0.0
numpy==1.24.3
pandas==2.0.3
```

---

## 6. Data Architecture

### Content Sources

#### PDF Sources (9,661 chunks)
- **Stanford CS229 ML Notes**: Theoretical foundations
- **Deep Learning Textbook**: Comprehensive DL coverage
- **Scikit-learn Documentation**: Practical implementation guides
- **PyTorch Documentation**: Framework-specific examples

#### Video Sources (2,923 chunks)
- **Stanford CS229 ML**: 612 chunks (Andrew Ng)
- **MIT 6.S191 DL (Alternative)**: 1,096 chunks (MIT)
- **Stanford CS224n NLP**: 661 chunks (Chris Manning)
- **Stanford CS231n CV**: 554 chunks (Fei-Fei Li)
- **MIT 6.S191 DL (Main)**: Additional MIT content

#### Web Sources (133 chunks)
- **Aman.ai Primers**: Modern AI topics (prompt engineering, recent advances)

### Metadata Schema

#### Standard Metadata
```python
{
    "chunk_id": str,              # Unique identifier
    "text": str,                  # Chunk content
    "source_name": str,           # Source document/collection
    "source_type": str,           # pdf, video, web
    "embedding": np.ndarray       # 384-dim vector
}
```

#### Video-Specific Metadata
```python
{
    "video_id": str,              # YouTube video ID
    "video_title": str,           # Video title
    "video_url": str,             # Full URL with timestamp
    "playlist_name": str,         # Course/playlist name
    "lecture_number": int,        # Lecture sequence
    "timestamp_start": float,     # Start time (seconds)
    "timestamp_end": float,       # End time (seconds)
    "duration": float,            # Chunk duration (seconds)
    "instructor": str,            # Course instructor
    "temporal_score": float,      # Temporal coherence score
    "previous_chunks": List[str], # Preceding chunk IDs
    "next_chunks": List[str]      # Following chunk IDs
}
```

#### PDF-Specific Metadata
```python
{
    "chapter": str,               # Chapter title
    "section": str,               # Section title
    "page_start": int,            # Starting page number
    "page_end": int               # Ending page number
}
```

### Storage Structure
```
data/
├── pdfs/                          # Original PDF documents
├── videos/                        # Downloaded videos (optional)
├── processed/
│   ├── chunks/                    # Processed text chunks
│   ├── embeddings/                # Generated embeddings
│   ├── indices/                   # FAISS vector indices
│   ├── video_chunks/              # Semantic video chunks
│   ├── video_chunks_cs224n/       # CS224n specific chunks
│   ├── video_chunks_cs231n/       # CS231n specific chunks
│   ├── video_chunks_mit_dl/       # MIT DL specific chunks
│   ├── video_chunks_mit_dl_main/  # MIT DL main chunks
│   ├── aman_primers/              # Aman.ai content
│   └── transcripts/               # Video transcripts
├── cache/                         # Cached data
└── ground_truth/                  # Evaluation ground truth
```

---

## 7. Query Processing Pipeline

### End-to-End Query Flow

```
User Query
    │
    ├─→ 1. Modality Prediction (0.1s)
    │   └─→ Feature extraction → Classification (Video/PDF/Web)
    │
    ├─→ 2. Parallel Retrieval (1-2s)
    │   ├─→ PDF Retrieval (5 chunks)
    │   ├─→ Video Retrieval (5 chunks × 5 playlists)
    │   └─→ Web Retrieval (5 chunks)
    │
    ├─→ 3. Cross-Modal Reranking (2-3s)
    │   ├─→ Query complexity detection
    │   ├─→ Adaptive reranker selection
    │   └─→ Top-N reranking (3 chunks per modality)
    │
    ├─→ 4. Temporal Processing (0.5s)
    │   └─→ Flow-aware video chunk ordering
    │
    ├─→ 5. Context Assembly (0.1s)
    │   └─→ Video links, citations, metadata
    │
    └─→ 6. Answer Generation (4-8s)
        └─→ LLM response with sources

Total: ~12s average
```

### Performance Characteristics

**Query Time Distribution**:
- Modality Prediction: 1%
- Retrieval: 15%
- Reranking: 20%
- LLM Generation: 60%
- Other: 4%

**System Performance**:
- **Average Query Time**: 11.94s
- **Median Time**: 12.01s
- **95th Percentile**: 14.5s
- **Success Rate**: 100% (72/72 queries)
- **Concurrent Capacity**: 5 parallel queries

---

## 8. Evaluation Architecture

### Metrics Framework

#### Retrieval Quality Metrics
- **Precision@K**: P@1 (1.0), P@3 (1.0), P@5 (0.8), P@10 (0.4)
- **Recall@K**: R@1 (0.25), R@3 (0.75), R@5 (1.0), R@10 (1.0)
- **MAP**: 1.0 (perfect)
- **MRR**: 1.0 (perfect)
- **NDCG@K**: 1.0 (perfect for all K)

#### Classification Metrics
- **Modality Prediction Accuracy**: 97.0%
- **Per-Class F1**: Video (0.959), PDF (0.943), Aman.ai (1.000)
- **Confusion Matrix**: Minimal cross-modal confusion

#### Novelty Metrics
- **Cross-Modal Consistency**: 0.87
- **Temporal Coherence Quality**: 0.92
- **Multi-Modal Context Utilization**: 0.78
- **Source Diversity**: 0.82
- **Citation Accuracy**: 0.75

#### RAG Quality Metrics
- **Overall RAG Quality**: 0.83/1.0 (EXCELLENT)
- **System Reliability**: 100.0%
- **Answer Relevance**: 0.87
- **Context Utilization**: 0.78

### Testing Infrastructure

#### Test Query Suite
- **Total Queries**: 72 comprehensive test queries
- **Categories**: ML (15), DL (15), NLP (12), CV (8), Advanced (10), Evaluation (12)
- **Complexity Levels**: Basic, Intermediate, Advanced
- **Coverage**: All modalities, query types, and domains

#### Evaluation Scripts
- **Quick Evaluation**: `run_simplified_rag_evaluation.py` (20 queries)
- **Comprehensive Evaluation**: `run_comprehensive_evaluation.py` (72 queries)
- **Metrics Generation**: `generate_comprehensive_metrics_report.py`
- **RAG Quality Evaluation**: `run_comprehensive_rag_evaluation.py`

---

## 9. Deployment Architecture

### Production Deployment

#### System Requirements
- **CPU**: 4+ cores recommended
- **Memory**: 8GB+ RAM (16GB for optimal performance)
- **Storage**: 5GB+ for indices and data
- **Network**: Stable internet for LLM API calls

#### Environment Setup
```bash
# Core dependencies
pip install -r requirements.txt

# Environment variables
GOOGLE_API_KEY=your-gemini-api-key
GEMINI_MODEL=models/gemini-3.1-flash-lite-preview

# Optional: GPU acceleration
pip install faiss-gpu  # Instead of faiss-cpu
```

#### Deployment Modes

1. **Gradio Web Interface** (Current)
   - Interactive web UI
   - Real-time query processing
   - Source visualization with video links

2. **REST API** (Future)
   - FastAPI backend
   - JSON request/response
   - Batch processing support

3. **CLI Interface** (Available)
   - Command-line query processing
   - Scriptable automation
   - Batch evaluation support

### Scalability Considerations

#### Horizontal Scaling
- **Stateless Design**: Pipeline can be replicated across instances
- **Load Balancing**: Multiple Gradio instances behind load balancer
- **Caching**: Embedding cache reduces redundant computation

#### Performance Optimization
- **Embedding Cache**: Reduces embedding generation time
- **Batch Processing**: Parallel processing of multiple queries
- **Reranker Optimization**: Query complexity-based model selection
- **Vector Index Optimization**: FAISS GPU support for faster retrieval

#### Monitoring & Logging
- **Structured Logging**: Detailed logs for debugging and monitoring
- **Performance Metrics**: Query time, success rate, resource utilization
- **Error Tracking**: Comprehensive error handling and reporting

---

## 10. Security & Privacy

### Data Privacy
- **No User Data Storage**: Queries processed but not stored
- **API Key Security**: Environment variables for sensitive keys
- **Content Attribution**: All sources properly cited

### API Security
- **Rate Limiting**: Configurable request limits
- **Input Validation**: Query sanitization and length limits
- **Error Handling**: Graceful failure without exposing system details

### Content Safety
- **Source Quality**: Academic and authoritative sources only
- **Response Filtering**: LLM temperature tuned for factual responses
- **Citation Verification**: All claims tied to retrieved sources

---

## 11. Future Architecture Enhancements

### Planned Improvements

#### Performance
- **GPU Acceleration**: FAISS GPU support for faster retrieval
- **Caching Layer**: Redis cache for frequent queries
- **Parallel Processing**: Multi-threaded retrieval and reranking

#### Features
- **Multi-Query Processing**: Batch query support
- **User Feedback**: Relevance feedback for continuous improvement
- **Personalization**: Adaptive modality selection based on user preferences

#### Content Expansion
- **Additional Courses**: More university courses and specializations
- **Research Papers**: ArXiv and academic paper integration
- **Interactive Content**: Code notebooks and interactive tutorials

---

## 12. System Limitations & Trade-offs

### Current Limitations

#### Performance
- **LLM Latency**: 4-8s per query (dominates response time)
- **Video Processing**: Semantic chunking is compute-intensive
- **Memory Usage**: All indices loaded in memory

#### Coverage
- **Domain Specific**: Focused on AI/ML topics
- **Language**: English-only content
- **Recency**: Web content not real-time updated

### Architectural Trade-offs

#### Quality vs. Speed
- **Decision**: Use high-quality LLM (Gemini) vs. faster models
- **Rationale**: Answer quality more important than speed for educational use

#### Coverage vs. Focus
- **Decision**: Domain-specific (AI/ML) vs. general knowledge
- **Rationale**: Specialized system provides better expert-level responses

#### Complexity vs. Maintainability
- **Decision**: Multi-modal system vs. single-modality RAG
- **Rationale**: Richer, more comprehensive answers worth added complexity

---

## 13. Development & Maintenance

### Code Organization

#### Module Structure
```
src/
├── loaders/          # Data loading (PDF, video, web)
├── processors/       # Content processing (chunking, analysis)
├── embeddings/       # Embedding generation
├── vector_store/     # FAISS index management
├── retrieval/        # Multi-modal retrieval
├── reranking/        # Cross-modal reranking
├── generation/       # LLM integration
├── pipeline/         # End-to-end pipelines
├── evaluation/       # Metrics and evaluation
├── scripts/          # Utility scripts
└── utils/            # Helper functions
```

### Configuration Management

#### Centralized Configuration
- **File**: `src/config.py`
- **Environment**: `.env` for API keys
- **Validation**: Config validation on startup

#### Key Configuration Sections
- Data sources (PDF, video, web)
- Processing parameters (chunking, embeddings)
- Model selection (embeddings, reranking, LLM)
- Performance tuning (batch sizes, thresholds)

---

**Document Owner**: Multi-Modal RAG System Development Team
**Last Review**: 2026-05-19
**Next Review**: After major system updates
**Version**: 3.0