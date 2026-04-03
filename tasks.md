# Multi-Modal RAG Project Tasks

**Project Duration:** 14 weeks
**Team Size:** 4 people
**Goal:** Build and publish a multi-modal educational RAG system with 3 novel contributions

---

## Phase 1: Base System Setup (Weeks 1-2)

### Environment & Dependencies
- [x] Set up project repository (Git, branching strategy)
- [x] Create project directory structure
- [x] Set up logging and error handling framework
- [ ] Configure Python virtual environment
- [ ] Install core dependencies:
  - [ ] LangChain + LangChain Community
  - [ ] Google Generative AI (Gemini)
  - [ ] Sentence Transformers
  - [ ] FAISS
  - [ ] PyPDF
  - [ ] NumPy, PyTorch
- [ ] Install video processing dependencies:
  - [ ] yt-dlp (YouTube download)
  - [ ] Whisper (transcription)
  - [ ] OpenCV (frame extraction)
  - [ ] Tesseract/paddleocr (OCR)
  - [ ] MoviePy (video processing)
- [ ] Set up Gemini API key and environment variables
- [ ] Test API connection

### PDF Processing Pipeline
- [ ] Download full PDF corpus:
  - [ ] CS229 Machine Learning notes (complete)
  - [ ] Deep Learning Book (Goodfellow) - full book
  - [ ] PyTorch documentation (tutorials and API reference)
  - [ ] Scikit-learn documentation (user guide and API)
  - [ ] Additional ML/DL textbooks and resources
- [ ] Implement PDF text extraction (PyPDF)
- [ ] Implement PDF chunking:
  - [ ] Fixed-size chunks (400-800 tokens)
  - [ ] 100 token overlap
  - [ ] Metadata extraction (page, chapter, source)
- [ ] Extract metadata from PDFs:
  - [ ] Source name
  - [ ] Topic classification
  - [ ] Difficulty level
  - [ ] Page numbers
- [ ] Process all PDFs and create production chunk database
- [ ] Create comprehensive PDF chunks (1000+ chunks covering all material)

### Video Processing Pipeline
- [ ] Download complete Stanford course playlists:
  - [ ] CS231n (CNNs) - complete course
  - [ ] CS224n (NLP) - complete course
  - [ ] CS230 (Deep Learning) - complete course
  - [ ] Additional relevant courses (MIT 6.S191, Fast.ai, etc.)
- [ ] Implement audio extraction from videos
- [ ] Set up Whisper transcription:
  - [ ] Generate timestamped transcripts for all videos
  - [ ] Save transcripts with metadata
  - [ ] Handle transcription errors and edge cases
- [ ] Implement frame extraction:
  - [ ] Extract keyframes every 5 seconds
  - [ ] Save frames with timestamps
- [ ] Implement slide detection:
  - [ ] Frame comparison algorithm
  - [ ] Detect slide changes
  - [ ] Extract slide metadata
- [ ] Implement OCR on keyframes:
  - [ ] Extract text from slides
  - [ ] Store OCR results with timestamps
- [ ] Process all video lectures end-to-end
- [ ] Create comprehensive video chunks (500+ chunks covering all lectures)

### Embedding Generation
- [ ] Set up sentence-transformers model (all-MiniLM-L6-v2)
- [ ] Generate embeddings for all PDF chunks
- [ ] Generate embeddings for all video transcripts
- [ ] Validate embedding quality (similarity checks)
- [ ] Save embeddings with metadata
- [ ] Create embedding backup and versioning

### Vector Database Setup
- [ ] Set up FAISS index:
  - [ ] Choose index type (IndexFlatIP for cosine similarity)
  - [ ] Configure embedding dimension (384)
  - [ ] Implement L2 normalization for cosine similarity
- [ ] Store all PDF embeddings in FAISS
- [ ] Store all video embeddings in FAISS (separate index or combined)
- [ ] Implement metadata storage alongside vectors
- [ ] Create index persistence and loading mechanism

### Basic Retrieval System
- [ ] Implement query embedding generation
- [ ] Implement vector similarity search (FAISS)
- [ ] Implement top-K retrieval (K=5)
- [ ] Implement metadata filtering:
  - [ ] Filter by source type (pdf/video)
  - [ ] Filter by difficulty
  - [ ] Filter by topic
- [ ] Validate retrieval with diverse query types
- [ ] Create retrieval performance benchmarks

### Prompt Construction & LLM Integration
- [ ] Design base prompt template:
  - [ ] System prompt (teaching assistant persona)
  - [ ] Context inclusion (retrieved chunks)
  - [ ] Citation format instructions
- [ ] Implement prompt construction with:
  - [ ] Retrieved chunks
  - [ ] Source metadata
  - [ ] Video timestamps/links
- [ ] Integrate Gemini 1.5 Flash API
- [ ] Implement response generation
- [ ] Validate end-to-end query responses

### Phase 1 Deliverables
- [ ] Production-ready multi-modal RAG system (PDF + video)
- [ ] Full corpus processed and indexed
- [ ] End-to-end query responses working
- [ ] Retrieval benchmarks established
- [ ] Complete documentation of setup and architecture

---

## Phase 2: Novel Contributions Development (Weeks 3-7)

### Contribution 1: Timestamp-Aware Video RAG (Weeks 3-4)

#### Topic Boundary Detection
- [ ] Research text segmentation algorithms:
  - [ ] TextTiling
  - [ ] Topic segmentation methods
  - [ ] Slide change detection
- [ ] Implement topic segmentation on all transcripts:
  - [ ] Sentence tokenization
  - [ ] Similarity computation between sentences
  - [ ] Boundary detection algorithm
- [ ] Implement slide change detection:
  - [ ] Frame comparison (SSIM, MSE)
  - [ ] Threshold tuning for slide changes
- [ ] Combine transcript and slide boundaries:
  - [ ] Align transcript with slide changes
  - [ ] Identify coherent topic segments
- [ ] Create smart video chunks for all lectures:
  - [ ] Chunk at topic boundaries (not fixed intervals)
  - [ ] Ensure 2-5 minute segments
  - [ ] Store with topic labels and timestamps
- [ ] Evaluate chunk quality across entire corpus:
  - [ ] Systematic quality assessment
  - [ ] Compare with fixed-interval chunking
- [ ] Generate embeddings for all smart chunks
- [ ] Update FAISS index with smart chunks

#### Testing & Optimization
- [ ] Test timestamp-aware retrieval systematically:
  - [ ] Query for specific topics across all courses
  - [ ] Verify retrieved segments are relevant
  - [ ] Check timestamps are accurate
- [ ] Compare with baseline (fixed-interval chunking):
  - [ ] Retrieval precision comparison on full dataset
  - [ ] Comprehensive evaluation queries
- [ ] Optimize boundary detection parameters
- [ ] Document improvements over baseline

**Deliverables:**
- [ ] Production timestamp-aware video chunking for all lectures
- [ ] Comprehensive evaluation vs fixed-interval chunking
- [ ] Documentation of algorithm and results

---

### Contribution 2: Temporal Coherence (Weeks 4-6)

#### Temporal Dependency Graph
- [ ] Design graph structure:
  - [ ] Nodes: Video chunks/topics
  - [ ] Edges: Temporal relationships (what follows what)
  - [ ] Weights: Relatedness scores
- [ ] Build temporal graph from lectures:
  - [ ] Extract topic sequence from transcripts
  - [ ] Identify dependencies between topics
  - [ ] Create edges between related consecutive topics
- [ ] Store graph with metadata:
  - [ ] Lecture structure
  - [ ] Topic progressions
  - [ ] Timestamp ranges

#### Coherence Scoring Algorithm
- [ ] Design coherence scoring function:
  - [ ] Score retrieval sets for temporal flow
  - [ ] Penalize jumps in time
  - [ ] Reward coherent progressions
- [ ] Implement coherence-aware retrieval:
  - [ ] When retrieving multiple segments:
    - [ ] Find all relevant segments
    - [ ] Select path through graph maintaining flow
    - [ ] Score: relevance + coherence
  - [ ] Return coherent segment sequences
- [ ] Test coherence scoring:
  - [ ] Manual inspection of retrieval sets
  - [ ] Compare with/without coherence
- [ ] Optimize coherence vs relevance trade-off

#### Testing & Evaluation
- [ ] Design evaluation queries:
  - [ ] Queries requiring multiple segments (50+ across all topics)
  - [ ] Step-by-step explanation queries
  - [ ] Complex concept queries
- [ ] Test temporal coherence systematically:
  - [ ] Do segments flow logically?
  - [ ] Are temporal jumps minimized?
  - [ ] Comprehensive evaluation across all courses
- [ ] Compare with baseline (no coherence):
  - [ ] Systematic evaluation of flow quality
  - [ ] Segment sequence quality metrics
- [ ] Document coherence improvements

**Deliverables:**
- [ ] Production temporal coherence system
- [ ] Complete temporal dependency graph for all lectures
- [ ] Coherence scoring algorithm
- [ ] Comprehensive evaluation vs baseline

---

### Contribution 3: Cross-Modal Reranking (Weeks 4-7)

#### Feature Extraction from Queries
- [ ] Design feature extraction pipeline:
  - [ ] Concept category classification:
    - [ ] Architecture (CNN, transformer, etc.)
    - [ ] Mathematical (formula, derivation, proof)
    - [ ] Implementation (code, API, how-to)
    - [ ] Conceptual (intuition, explanation)
  - [ ] Complexity indicators:
    - [ ] Has math notation?
    - [ ] Has code keywords?
    - [ ] Has visual keywords?
    - [ ] Question complexity
  - [ ] Domain keywords:
    - [ ] ML/DL terminology
    - [ ] Framework-specific terms
- [ ] Implement feature extraction:
  - [ ] NLP-based classification
  - [ ] Keyword matching
  - [ ] Pattern detection
- [ ] Validate feature extraction on diverse query types

#### Training Data Creation
- [ ] Create heuristic rules for modality preference:
  ```python
  rules = {
    "visual_keywords": ["diagram", "show", "visual", "architecture"] → video,
    "math_keywords": ["formula", "equation", "derive", "proof"] → pdf,
    "code_keywords": ["implement", "code", "api", "how to"] → documentation,
    "explanation_keywords": ["explain", "what is", "how does"] → video_or_pdf
  }
  ```
- [ ] Apply heuristics to comprehensive training dataset (500+ queries)
- [ ] Create labeled dataset: query → best modality
- [ ] Manual labeling and validation of queries
- [ ] Split dataset: train/validation/test

#### Modality Classifier Training
- [ ] Choose classifier:
  - [ ] Start simple: Rule-based classifier
  - [ ] Upgrade to: Logistic Regression / Random Forest
  - [ ] Advanced: BERT-based classifier (if time permits)
- [ ] Train classifier on labeled data
- [ ] Evaluate classifier performance:
  - [ ] Accuracy metrics
  - [ ] Confusion matrix
  - [ ] Per-class precision/recall
- [ ] Optimize classifier parameters
- [ ] Validate on held-out test set

#### Adaptive Retrieval Integration
- [ ] Implement modality-based weight adjustment:
  ```python
  if video_predicted:
    boost_video_chunks_by = 0.2  # 20% boost
  elif pdf_predicted:
    boost_pdf_chunks_by = 0.2
  elif doc_predicted:
    boost_documentation_by = 0.2
  ```
- [ ] Integrate into retrieval pipeline:
  - [ ] Predict modality for query
  - [ ] Adjust chunk scores based on prediction
  - [ ] Re-rank chunks
  - [ ] Return top results
- [ ] Validate adaptive retrieval:
  - [ ] Verify modality prediction works
  - [ ] Check that boosted modalities appear in results
  - [ ] Ensure quality doesn't decrease

#### Testing & Evaluation
- [ ] Design comprehensive evaluation queries (200+):
  - [ ] Visual questions (should prefer video)
  - [ ] Math questions (should prefer PDF)
  - [ ] Code questions (should prefer docs)
  - [ ] Mixed questions
- [ ] Test cross-modal reranking systematically:
  - [ ] Accuracy of modality prediction
  - [ ] User satisfaction with modality choices
  - [ ] Compare with uniform weighting
- [ ] Ablation study:
  - [ ] With vs without adaptive weighting
  - [ ] Measure improvement in user satisfaction
- [ ] Document cross-modal improvements

**Deliverables:**
- [ ] Production modality classifier
- [ ] Adaptive retrieval system
- [ ] Comprehensive evaluation vs uniform weighting
- [ ] Complete training dataset

---

### Contribution 4: Evaluation Infrastructure (Weeks 3-7)

#### Evaluation Dataset Creation
- [ ] Design evaluation question categories:
  - [ ] Basic concepts (20+ questions)
  - [ ] Intermediate concepts (40+ questions)
  - [ ] Advanced concepts (40+ questions)
  - [ ] Implementation questions (20+ questions)
  - [ ] Comparative questions (20+ questions)
- [ ] Create comprehensive evaluation questions (200+ total):
  - [ ] Ensure coverage of all topics across all courses
  - [ ] Include different question types
  - [ ] Add difficulty ratings
  - [ ] Add expected source types
- [ ] Create ground truth answers:
  - [ ] Key points to cover
  - [ ] Expected sources
  - [ ] Expected difficulty level
- [ ] Peer review evaluation questions:
  - [ ] Check for clarity
  - [ ] Verify difficulty ratings
  - [ ] Ensure ground truth is accurate

#### Evaluation Metrics Implementation
- [ ] Implement retrieval metrics:
  - [ ] Recall@K (K=3,5,10)
  - [ ] Precision@K
  - [ ] Mean Reciprocal Rank (MRR)
  - [ ] NDCG@K
- [ ] Implement video-specific metrics:
  - [ ] Timestamp accuracy
  - [ ] Video relevance scoring
  - [ ] Multi-modal coverage
  - [ ] Segment duration quality
- [ ] Implement answer quality metrics:
  - [ ] Faithfulness scoring (manual or NLI)
  - [ ] Correctness evaluation (LLM-as-judge)
  - [ ] Completeness scoring (ROUGE-L)
  - [ ] Citation accuracy
- [ ] Implement hallucination metrics:
  - [ ] Hallucination rate
  - [ ] Entity F1 score
  - [ ] Timestamp accuracy check
- [ ] Create evaluation pipeline:
  - [ ] Run all metrics automatically
  - [ ] Generate evaluation reports
  - [ ] Create visualizations

#### User Study Preparation
- [ ] Design user study protocol:
  - [ ] Participant recruitment plan
  - [ ] Informed consent forms
  - [ ] Study session structure (30-45 min)
- [ ] Create pre-test:
  - [ ] 10-15 ML/DL questions
  - [ ] Cover various difficulty levels
  - [ ] Multiple choice + short answer
- [ ] Create post-test:
  - [ ] Similar to pre-test (different questions)
  - [ ] Measure learning improvement
- [ ] Create user feedback survey:
  - [ ] System usability (SUS score)
  - [ ] Answer quality ratings
  - [ ] Multi-modal preference
  - [ ] Feature feedback
  - [ ] Open-ended suggestions
- [ ] Create system usage tasks:
  - [ ] 5-10 practice queries
  - [ ] Different question types
  - [ ] Cover different modalities
- [ ] Prepare study materials:
  - [ ] User instructions
  - [ ] Demo video (optional)
  - [ ] Practice session guide

**Deliverables:**
- [ ] Evaluation dataset (70-100 questions)
- [ ] Evaluation metrics pipeline
- [ ] User study materials ready
- [ ] IRB approval (if required)

---

## Phase 3: Integration (Week 8)

### System Integration
- [ ] Combine all 3 contributions into single system:
  - [ ] Timestamp-aware video chunks
  - [ ] Temporal coherence retrieval
  - [ ] Cross-modal reranking
- [ ] Resolve conflicts between components:
  - [ ] Ensure coherence doesn't interfere with modality selection
  - [ ] Balance relevance, coherence, and modality scores
  - [ ] Test all combinations
- [ ] Create unified retrieval pipeline:
  - [ ] Query → Feature extraction → Modality prediction
  - [ ] Multi-source retrieval (PDF + video)
  - [ ] Apply temporal coherence
  - [ ] Apply cross-modal reranking
  - [ ] Return final results
- [ ] Update prompt construction:
  - [ ] Include all 3 contributions in responses
  - [ ] Show timestamps, coherence info, modality rationale
  - [ ] Test prompt formats

### End-to-End Testing
- [ ] Test full system with comprehensive query suite (100+ queries):
  - [ ] Visual/conceptual queries
  - [ ] Mathematical queries
  - [ ] Implementation queries
  - [ ] Multi-segment queries
  - [ ] Different difficulty levels
  - [ ] All courses and topics covered
- [ ] Verify all components work together:
  - [ ] Video chunks are timestamp-aware
  - [ ] Multiple segments maintain coherence
  - [ ] Modality selection is appropriate
- [ ] Performance testing:
  - [ ] Measure query latency
  - [ ] Test with concurrent queries
  - [ ] Identify bottlenecks
- [ ] Bug fixing:
  - [ ] Fix integration issues
  - [ ] Handle edge cases
  - [ ] Error handling improvements

### Optimization
- [ ] Optimize retrieval speed:
  - [ ] Cache frequent queries
  - [ ] Optimize FAISS search
  - [ ] Batch processing where possible
- [ ] Optimize response quality:
  - [ ] Tune scoring weights
  - [ ] Adjust thresholds
  - [ ] Improve prompt templates
- [ ] System polish:
  - [ ] Improve error messages
  - [ ] Add logging
  - [ ] Create admin interface (optional)

**Deliverables:**
- [ ] Fully integrated system
- [ ] All 3 contributions working together
- [ ] End-to-end test results
- [ ] Performance benchmarks

---

## Phase 4: Evaluation (Weeks 9-10)

### Baseline Comparisons
- [ ] Implement baseline systems:
  - [ ] LLM-only (no retrieval)
  - [ ] PDF-only RAG
  - [ ] Video-only RAG
  - [ ] Naive multi-modal RAG (no contributions)
  - [ ] Individual contribution ablations:
    - [ ] Without timestamp-aware
    - [ ] Without temporal coherence
    - [ ] Without cross-modal reranking
- [ ] Run comprehensive evaluation on all baselines:
  - [ ] Use full evaluation dataset (200+ questions)
  - [ ] Collect all metrics
  - [ ] Generate comparison reports

### Ablation Studies
- [ ] Test each contribution independently:
  - [ ] Timestamp-aware vs fixed-interval chunking
  - [ ] With vs without temporal coherence
  - [ ] With vs without cross-modal reranking
- [ ] Test combinations:
  - [ ] Timestamp + Coherence (no reranking)
  - [ ] Timestamp + Reranking (no coherence)
  - [ ] Coherence + Reranking (no timestamp)
  - [ ] All three contributions
- [ ] Measure impact of each:
  - [ ] Precision@K improvements
  - [ ] Coherence scores
  - [ ] Modality prediction accuracy
  - [ ] User satisfaction (if available)

### Statistical Analysis
- [ ] Perform statistical significance testing:
  - [ ] Paired t-tests for metric comparisons
  - [ ] ANOVA for multiple conditions
  - [ ] Effect size calculations
- [ ] Create visualizations:
  - [ ] Bar charts comparing baselines
  - [ ] Line graphs showing ablation results
  - [ ] Tables with metrics and p-values
  - [ ] Qualitative examples
- [ ] Document findings:
  - [ ] Which contributions help most?
  - [ ] Where does system fail?
  - [ ] What are the limitations?

### Performance Evaluation
- [ ] Measure system performance:
  - [ ] Query latency (p50, p95, p99)
  - [ ] Throughput (queries/second)
  - [ ] Memory usage
  - [ ] Storage requirements
- [ ] Identify bottlenecks:
  - [ ] Slowest components
  - [ ] Resource constraints
  - [ ] Optimization opportunities

**Deliverables:**
- [ ] Complete evaluation results
- [ ] Comparison with all baselines
- [ ] Ablation study results
- [ ] Statistical analysis
- [ ] Performance benchmarks

---

## Phase 5: Final Deliverables (Weeks 11-12)

### Code & System
- [ ] Clean up codebase:
  - [ ] Remove debug code
  - [ ] Add comments (no emojis per rules.md)
  - [ ] Create README for main project
  - [ ] Add usage examples
- [ ] Prepare code repository:
  - [ ] Organize files according to rules.md structure
  - [ ] Add appropriate license
  - [ ] Create comprehensive documentation
  - [ ] Add setup instructions (local + Kaggle)
- [ ] Create demo:
  - [ ] Prepare comprehensive demo queries showcasing all 3 contributions
  - [ ] Create demo script that runs end-to-end
  - [ ] Test demo thoroughly (multiple times)
  - [ ] Prepare demo video/screencast
- [ ] Archive evaluation data:
  - [ ] Complete evaluation dataset (200+ questions with ground truth)
  - [ ] All evaluation results (metrics, baselines, ablations)
  - [ ] Statistical analysis data
  - [ ] Performance benchmarks

### Documentation
- [ ] System documentation:
  - [ ] Architecture overview (how components work together)
  - [ ] API documentation (all modules and functions)
  - [ ] Setup guide (local environment + Kaggle)
  - [ ] User guide (how to use the system)
- [ ] Project report:
  - [ ] Technical details of each contribution
  - [ ] Design decisions and trade-offs
  - [ ] Lessons learned during development
  - [ ] Future work suggestions
- [ ] Presentation materials:
  - [ ] Slides for class presentation (15-20 minutes)
  - [ ] Live demo script
  - [ ] Q&A preparation (common questions)
  - [ ] Backup plan (in case demo fails)

### Final Testing & Validation
- [ ] End-to-end system testing:
  - [ ] Test all 3 contributions working together
  - [ ] Test with comprehensive query suite (100+ queries)
  - [ ] Verify all baselines work
  - [ ] Confirm evaluation pipeline runs without errors
- [ ] Performance validation:
  - [ ] Measure actual query latency
  - [ ] Test system under load
  - [ ] Verify memory usage is acceptable
  - [ ] Document any limitations
- [ ] Code quality checks:
  - [ ] Run all tests (unit + integration)
  - [ ] Check test coverage (>80%)
  - [ ] Run linters and formatters
  - [ ] Ensure no hardcoded values
  - [ ] Verify no emojis in code (per rules.md)

### Final Checklist
- [ ] All 3 contributions implemented and working
- [ ] Base RAG system functional (PDF + video retrieval)
- [ ] Evaluation complete with all metrics
- [ ] All baselines tested and compared
- [ ] Ablation studies complete
- [ ] Code clean, documented, and tested
- [ ] Demo prepared and tested
- [ ] Presentation slides ready
- [ ] Project report complete
- [ ] All documentation up to date

---

## Weekly Checkpoints

- **Week 2:** Base system working (PDF + video retrieval)
- **Week 4:** First contribution complete (Timestamp-Aware Video RAG)
- **Week 6:** Second contribution complete (Temporal Coherence)
- **Week 7:** Third contribution complete (Cross-Modal Reranking)
- **Week 8:** Full system integrated and tested
- **Week 10:** Evaluation complete (all baselines and ablations)
- **Week 12:** Final deliverables complete and tested

---

## Notes

- **Production-scale system:** This is a full production system, not a prototype - process all available data
- **Parallel work encouraged:** Multiple team members can work on different tasks simultaneously
- **Regular sync:** Weekly team meetings to track progress
- **Flexible timeline:** Some tasks may take more/less time than estimated
- **Buffer time:** Build in extra time for unexpected issues
- **Focus on quality:** Prioritize comprehensive coverage over quick prototypes
- **Test early and often:** Don't wait until end to test
- **Document everything:** Keep track of decisions, results, and issues
- **Use AI assistant:** Claude is available to help with coding, debugging, and documentation
- **Follow rules.md:** Adhere to coding standards, especially no emojis in code

