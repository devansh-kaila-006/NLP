# Multi-Modal Video+PDF RAG for ML/DL Learning

## What This Project Is

An intelligent learning assistant that combines **academic PDFs** and **video lectures** to answer Machine Learning and Deep Learning questions with both theoretical explanations AND direct video timestamps for visual demonstrations.

### The Core Problem This Solves

Traditional RAG systems only use text. This system:
- Answers "What is backpropagation?" with mathematical formulas from PDFs
- **AND** provides direct links like `youtube.com/watch?v=xxx&t=750` showing exactly where in a lecture to watch the visual explanation
- Students get both theory (PDF) and intuitive visual explanations (video) in one response

---

## How It Works

### Pipeline Flow

```
PDF Sources → Text Extraction → Semantic Chunking → Embedding Generation
                                                              ↓
User Query → Embedding → Vector Search → Reranking → Context Selection → LLM Response
                                                              ↑
YouTube Playlists → Video Download → Whisper Transcription → Frame Extraction → Timestamp Segmentation
```

### Components

1. **PDF Processing**
   - Sources: CS229 notes, Deep Learning textbook, framework documentation
   - Extract text, chunk into 400-800 token segments
   - Store with metadata (source, page, topic)

2. **Video Processing** (The novel part)
   - Download Stanford course playlists (CS231n, CS224n, CS230)
   - Transcribe audio using Whisper (with timestamps)
   - Extract keyframes for slide/diagram detection
   - Chunk into 2-5 minute segments with timestamp metadata

3. **Multi-Modal Retrieval**
   - Query gets embedded and searched against BOTH PDF and video chunks
   - Returns top results from both modalities
   - Video chunks include: timestamp, lecture number, video_url

4. **Reranking Layer**
   - Cross-encoder re-scores retrieved chunks
   - Selects top 2-3 most relevant chunks (from either PDF or video)
   - Improves precision by 10-30%

5. **Response Generation**
   - LLM (Gemini 1.5 Flash free API) generates answer
   - Includes citations: `[Source: Deep Learning Book, Ch. 6, p.180]`
   - Includes video links: `[Video: CS231n Lecture 5, 12:30-15:45] youtube.com/watch?v=xxx&t=750`

---

## Three Novel Contributions

### 1. Timestamp-Aware Video RAG ⭐ (Easy)
**Problem:** Most systems chunk videos at fixed 3-minute intervals, splitting related concepts

**Solution:** Smart chunking using topic boundary detection
- Analyze transcripts for topic shifts
- Detect slide changes using frame comparison
- Chunk at semantic boundaries, not fixed time intervals

**Impact:** Better retrieval precision, more coherent chunks

### 2. Temporal Coherence ⭐⭐⭐ (Medium)
**Problem:** Traditional RAG might return video segments that jump around in time, confusing students

**Solution:** Ensure retrieved segments maintain logical flow
- Build temporal dependency graph of topics
- When retrieving multiple segments, select path that maintains flow
- Score retrieval sets for coherence

**Example:**
```
Query: "Explain backpropagation step by step"

Without coherence:
- Lecture 5, 12:30 (backprop overview)
- Lecture 2, 45:00 (optimization intro) ← jumps backward
- Lecture 5, 15:00 (computation graph)

With coherence:
- Lecture 5, 12:30 (backprop overview)
- Lecture 5, 15:00 (computation graph)
- Lecture 5, 18:00 (chain rule) ← flows forward
```

### 3. Cross-Modal Reranking ⭐⭐⭐⭐ (Hard)
**Problem:** Different queries need different modalities
- "How does convolution work?" → Video (visual)
- "What's the mathematical formula?" → PDF (math)
- "How do I implement this?" → Code docs

**Solution:** Learn which modality works best for each query type
- Extract features from query (concept type, complexity, keywords)
- Predict optimal modality (video vs PDF vs docs)
- Boost relevant modality by 20-30% during retrieval
- Rerank based on adjusted scores

**Impact:** Better modality selection, improved user satisfaction

---

## Example Response

**User Query:** "Explain how backpropagation works"

**System Response:**
```
Backpropagation computes gradients efficiently by applying the chain rule
recursively from output to input layers.

Mathematical Foundation:
∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w
[Source: Deep Learning Book, Ch. 6, p.205]

Step-by-Step:
1. Forward pass: Compute activations
2. Compute loss at output
3. Backward pass: Propagate gradients
4. Update weights

Visual Explanation:
Watch this segment for intuitive visual walkthrough:
[Video: CS231n Lecture 4, 15:30-22:45]
🔗 youtube.com/watch?v=abc123&t=930

Implementation (PyTorch):
```python
loss = criterion(output, target)
loss.backward()  # Computes gradients via backprop
optimizer.step()
```
[Source: PyTorch Documentation]
```

---

## Tech Stack

**Core:**
- Python 3.9+
- LangChain (RAG pipeline)
- Sentence Transformers (embeddings: all-MiniLM-L6-v2)
- FAISS (vector search)
- Gemini 1.5 Flash Free API (LLM)

**PDF Processing:**
- pypdf

**Video Processing:**
- yt-dlp (YouTube downloads)
- Whisper (speech-to-text)
- OpenCV (frame extraction)
- Tesseract (OCR for slides)

**Reranking:**
- CrossEncoder (ms-marco-MiniLM-L-6-v2)

---

## Implementation Timeline (4-person team)

**Phase 1: Base System (Weeks 1-2)**
- Gemini API setup
- PDF processing pipeline
- Video download + Whisper transcription
- Embeddings + FAISS index

**Phase 2: Novel Contributions (Weeks 3-7)**
- Person 1: Timestamp-Aware Video RAG (2-3 weeks)
- Person 2: Temporal Coherence (3-4 weeks)
- Person 3: Cross-Modal Reranking (4-5 weeks)
- Person 4: Evaluation dataset + user study prep

**Phase 3: Integration (Week 8)**
- Combine all contributions
- End-to-end testing

**Phase 4: Evaluation (Weeks 9-10)**
- Run evaluation (50-100 questions)
- Ablation studies
- Compare vs baselines (PDF-only, video-only, naive RAG)

**Phase 5: User Study (Weeks 11-12)**
- 10-15 participants
- Pre/post tests
- Qualitative feedback

**Phase 6: Paper Writing (Weeks 13-14)**
- Workshop paper submission

---

## Expected Outcomes

**Quantitative:**
- +40-50% factual accuracy vs LLM-only (grounded in sources)
- +25-35% comprehension vs PDF-only (visual explanations)
- Hallucination rate <5%
- Video timestamp accuracy >90%

**Qualitative:**
- Students learn better with multi-modal explanations
- Direct video links reduce search friction
- Comprehensive coverage (theory + visual + code)

---

## Why This Matters

This is the first educational RAG system that:
1. Combines academic PDFs AND video lectures intelligently
2. Provides precise video timestamps for targeted learning
3. Maintains temporal coherence across video segments
4. Adaptively selects best modality per query

**Publication Potential:** Workshop papers in educational AI venues (EDM, L@S, AIED)

---

*Document simplified from original 1850-line architecture document*
