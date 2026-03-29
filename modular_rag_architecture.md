# Multi-Modal Video+PDF RAG Architecture for ML/DL Learning Assistant

## Project Title
A Multi-Modal Retrieval-Augmented Generation Framework Integrating Academic Text and Educational Video Content with Temporal Coherence and Cross-Modal Reranking for Comprehensive Machine Learning and Deep Learning Explanations

---

# 1. Overview

This project implements a **cutting-edge Multi-Modal Retrieval-Augmented Generation (RAG)** pipeline that processes both **academic PDF documents** and **educational video content** from Stanford courses to build an intelligent learning assistant for Machine Learning and Deep Learning concepts.

The system uniquely integrates:
- **Academic PDF sources** (textbooks, course notes, documentation)
- **Video lecture content** (Stanford course playlists with temporal segmentation)
- **Multi-modal retrieval** (text, mathematics, code, diagrams, video segments)
- **Timestamp-aware responses** with direct links to specific lecture moments
- **Three novel contributions**: See Section 14

**Key Innovation:**
Unlike traditional RAG systems that process only text, this system retrieves and synthesizes explanations from **both PDF documents and video lectures**, providing students with comprehensive, multi-format learning resources including direct video timestamps for visual explanations.

**Three Novel Contributions:**
1. **Timestamp-Aware Video RAG**: Smart video chunking using topic boundary detection instead of fixed time intervals
2. **Temporal Coherence**: Retrieved video segments maintain logical flow and tell coherent stories
3. **Cross-Modal Reranking**: Learn which modality (PDF vs Video) works best for different query types

**Key Differentiators:**
- **Multi-source integration**: PDF textbooks + video lecture playlists
- **Video RAG with temporal retrieval**: Direct links to specific lecture timestamps
- **Multi-modal content**: Text, math equations, code, diagrams, and video explanations
- **Academic source grounding**: Authoritative Stanford courses and textbooks
- **Intelligent modality selection**: Cross-modal reranking for optimal explanation type

---

# 1.1 System Requirements

**Hardware:**
- CPU: 8+ cores recommended (video processing requires more power)
- RAM: 16GB minimum, 32GB+ recommended for video processing
- GPU: **Strongly recommended** for video processing and Whisper transcription (CUDA-compatible)
- Storage: 50-100GB for video downloads + PDF sources + vector indices + transcriptions
- Internet: Required for YouTube playlist downloads

**Software:**
- Python: 3.9+
- OS: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)
- FFmpeg: Required for video/audio processing
- yt-dlp: For YouTube video downloads

**Dependencies:**
```
langchain>=0.1.0
langchain-community>=0.0.10
langchain-google-genai>=1.0.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4  # or faiss-gpu for GPU acceleration
transformers>=4.35.0
torch>=2.0.0
pypdf>=3.17.0
numpy>=1.24.0
google-generativeai>=0.3.0

# Video Processing (NEW)
yt-dlp>=2023.0.0  # YouTube video downloads
whisper-openai>=20230314  # Speech-to-text transcription
opencv-python>=4.8.0  # Video frame extraction
pillow>=10.0.0  # Image processing
pytesseract>=0.3.10  # OCR for slides/text in video
moviepy>=1.0.3  # Video editing and processing
```

---

# 1.2 Quick Setup Guide (Gemini API)

**5-Minute Setup:**

**Step 1: Get API Key**
```bash
# Visit: https://makersuite.google.com/app/apikey
# Sign in with Google account
# Create API key (save it!)
```

**Step 2: Set Environment Variable**
```bash
# Linux/Mac
export GOOGLE_API_KEY="AIza...your-key-here"

# Windows (PowerShell)
$env:GOOGLE_API_KEY="AIza...your-key-here"

# Or create .env file:
echo "GOOGLE_API_KEY=AIza...your-key-here" > .env
```

**Step 3: Install Dependencies**
```bash
pip install google-generativeai langchain-google-genai \
            langchain langchain-community \
            sentence-transformers faiss-cpu \
            pypdf numpy torch
```

**Step 4: Test Connection**
```python
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("Explain gradient descent in one sentence.")
print(response.text)
```

**If successful, you're ready to build!** 🚀

**Common Issues:**
- `API key not valid`: Check your key, ensure no extra spaces
- `Quota exceeded`: Free tier has limits (15 req/min for Flash)
- `Module not found`: Run `pip install` again with `--upgrade`

**Free Tier Limits (Don't Worry):**
- Gemini 1.5 Flash: 15 requests/minute = 900/hour
- Enough for development, testing, and learning
- Only pay if you need production-scale usage

---

# 2. Architecture Type

Pipeline Type:

**Multi-Modal Video+PDF RAG + Semantic Retrieval + Cross-Encoder Reranking**

Pipeline Flow:

**PDF Processing Pipeline:**
PDF Documents → Text Extraction → Semantic Chunking → Embedding Generation → Vector Store

**Video Processing Pipeline:**
YouTube Playlists → Video Download → Audio Extraction → Whisper Transcription → Frame Sampling → Slide Detection → Timestamp Segmentation → Multi-Modal Embedding → Vector Store

**Query Processing Pipeline:**
User Query → Embedding Generation → Multi-Source Vector Search (PDF + Video) → Top-K Retrieval → Reranking Layer → Multi-Modal Context Selection → Prompt Construction with Timestamps → LLM Response Generation

**Multi-Modal Retrieval:**
- Text segments from PDFs
- Transcript segments from video (with timestamps)
- Diagrams/slides extracted from video frames
- Code examples from documentation
- Mathematical formulations from PDFs

**Response Format:**
- Text explanation with citations
- Direct video links with timestamps (e.g., youtube.com/watch?v=xxx&t=530)
- Mathematical formulas from source materials
- Code examples from documentation
- Diagram references from video slides

---

# 3. System Components

## 3.1 Document Loader

Responsible for loading academic PDF sources and educational video content.

### PDF Sources

**Text-Based Academic Resources:**
- CS229 Machine Learning notes (PDF)
- Deep Learning textbook (Goodfellow) (PDF)
- PyTorch documentation excerpts (web/PDF)
- Scikit-learn documentation excerpts (web/PDF)

**Library:** LangChain PDF Loader

### Video Sources

**Stanford Course Video Playlists:**
- Stanford CS231n: Convolutional Neural Networks (YouTube playlist)
- Stanford CS224n: Natural Language Processing / Transformers (YouTube playlist)
- Stanford CS230: Deep Learning (YouTube playlist)

**Video Processing Pipeline:**

1. **Video Download**
   - Tool: `yt-dlp`
   - Download entire playlists or individual lectures
   - Extract audio track for transcription
   - Store video metadata (duration, lecture number, course)

2. **Audio Transcription**
   - Tool: OpenAI Whisper (local or API)
   - Output: Timestamped transcript with speaker segmentation
   - Format: JSON with start/end times for each segment
   - Example: `{"text": "Backpropagation works by...", "start": 745.2, "end": 752.8}`

3. **Frame Extraction**
   - Tool: OpenCV (cv2)
   - Extract keyframes every 5 seconds or on scene change
   - Detect slide changes using frame comparison
   - Store frames with timestamps

4. **Slide/Text Detection**
   - Tool: Tesseract OCR or paddleocr
   - Extract text from video frames (slides, whiteboard)
   - Detect diagrams and visual elements
   - Store OCR results with timestamps

**Implementation:**
```python
import yt_dlp
import whisper
import cv2

# Download video
ydl_opts = {'format': 'bestvideo+bestaudio/best'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://youtube.com/playlist?list=...'])

# Transcribe audio
model = whisper.load_model('base')
result = model.transcribe('lecture.mp4', timestamp=True)

# Extract frames
video = cv2.VideoCapture('lecture.mp4')
frames = extract_keyframes(video, interval_seconds=5)
```

---

## 3.2 Semantic Chunking Module

Documents and video transcripts are split into concept-level segments.

### PDF Chunking

**Chunk Size:** 400–800 tokens
**Chunk Overlap:** 100 tokens

**Each chunk includes metadata:**
```json
{
  "concept_title": "Logistic Regression",
  "source_name": "CS229",
  "source_type": "pdf",
  "topic": "ML Theory",
  "difficulty": "Intermediate",
  "page_number": 45
}
```

### Video Chunking

**Temporal Segmentation:**
- **Chunk Size:** 2-5 minute segments (or topic-based)
- **Overlap:** 30 seconds between segments
- **Segmentation Strategy:**
  - Fixed time intervals (simple)
  - Topic boundary detection (advanced - using transcript analysis)
  - Slide change detection (when visual content changes)

**Each video chunk includes:**
```json
{
  "concept_title": "Backpropagation Intuition",
  "source_name": "CS231n",
  "source_type": "video",
  "topic": "Neural Networks",
  "difficulty": "Intermediate",
  "timestamp_start": "12:30",
  "timestamp_end": "15:45",
  "timestamp_seconds_start": 750.0,
  "timestamp_seconds_end": 945.0,
  "lecture_number": 5,
  "slide_number": 12,
  "video_url": "youtube.com/watch?v=xxx&t=750",
  "transcript": "Backpropagation works by computing gradients...",
  "frame_count": 30,
  "has_diagram": true,
  "ocr_text": "∂L/∂w = ..."
}
```

**Multi-Modal Chunk Storage:**
- **Text:** Transcript segment
- **Visual:** Key frames from the segment
- **OCR Text:** Text extracted from slides/diagrams
- **Audio:** Audio features (optional, for speaker identification)

**Implementation:**
```python
def chunk_video_transcript(transcript, chunk_duration_seconds=180):
    chunks = []
    current_chunk = []
    start_time = transcript[0]['start']

    for segment in transcript:
        current_chunk.append(segment)

        # Create new chunk every 3 minutes or at topic boundaries
        if segment['end'] - start_time >= chunk_duration_seconds:
            chunk = {
                'text': ' '.join([s['text'] for s in current_chunk]),
                'start_time': start_time,
                'end_time': segment['end'],
                'timestamp_start': format_timestamp(start_time),
                'timestamp_end': format_timestamp(segment['end'])
            }
            chunks.append(chunk)
            current_chunk = []
            start_time = segment['start']

    return chunks
```

---

## 3.3 Embedding Module

Embedding Model:

Sentence-Transformers

Recommended Model:

all-MiniLM-L6-v2

**Model Specifications:**
- Embedding Dimension: 384
- Max Sequence Length: 256 tokens
- Inference Speed: ~1000 docs/sec on CPU
- Model Size: ~80MB

**Alternative Models:**
- all-mpnet-base-v2 (768 dim, better accuracy, slower)
- e5-base-v2 (512 dim, optimized for retrieval)

Purpose:

Convert document chunks into semantic vectors for similarity search.

**Implementation:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks, show_progress_bar=True)
```

---

## 3.4 Vector Database

Database Used:

FAISS (Facebook AI Similarity Search)

**Index Configuration:**
- Index Type: IndexFlatIP (inner product) for cosine similarity
- Dimension: 384 (matches embedding model)
- Normalization: L2 normalize vectors for cosine similarity

**FAISS Index Options:**

| Index Type | Use Case | Pros | Cons |
|------------|----------|------|------|
| IndexFlatIP | <100K chunks | Exact search, simple | Slow for large datasets |
| IndexIVFFlat | 100K-1M chunks | Faster search | Requires training, approximate |
| IndexIVFPQ | >1M chunks | Memory efficient | More approximate, lower accuracy |

**GPU Acceleration:**
- Use `faiss-gpu` package for CUDA acceleration
- Recommended for >50K chunks
- 10-100x faster search than CPU

Purpose:

Efficient semantic similarity retrieval of relevant knowledge chunks.

**Implementation:**
```python
import faiss
index = faiss.IndexFlatIP(384)  # 384 = embedding dimension
faiss.normalize_L2(embeddings)  # For cosine similarity
index.add(embeddings)
```

---

## 3.5 Retrieval Module

Retrieval Strategy:

Top-K semantic similarity retrieval

Typical Value:

Top 5 chunks per query

Retrieval uses:

Vector similarity + metadata filtering

---

## 3.6 Reranking Module

Purpose:

Improve retrieval precision by selecting the most contextually relevant chunks before passing them to the LLM.

Model Type:

Cross-Encoder Reranker

Recommended Model:

cross-encoder/ms-marco-MiniLM-L-6-v2

**Model Specifications:**
- Scoring: Relevance score (0-1 range)
- Inference Speed: ~500 pairs/sec on CPU
- Model Size: ~110MB

Pipeline:

Retrieve Top 5 chunks
→ Rerank using cross-encoder
→ Score threshold filtering (optional)
→ Select Top 2–3 chunks

**Scoring Logic:**
- Scores represent query-chunk relevance probability
- Higher scores = more relevant
- Typical good scores: >0.7 for academic content

**Threshold Strategy:**
- **Conservative**: Use only chunks with score >0.8 (may return 0-1 chunks)
- **Balanced**: Use top-2 chunks regardless of score (recommended)
- **Fallback**: If all scores <0.5, expand retrieval to top-10 and re-rerank

**Alternative Models:**
- cross-encoder/mmarco-mMiniLMv2 (better for multilingual)
- bge-reranker-base (higher accuracy, slower)

Benefit:

Improves explanation accuracy (10-30% in benchmarks)
Reduces hallucinations by filtering irrelevant context
Improves context relevance for complex queries

**Implementation:**
```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pairs = [[query, chunk] for chunk in retrieved_chunks]
scores = reranker.predict(pairs)
top_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)][:3]
```

---

## 3.7 Prompt Construction Module

Prompt includes:

User question
Retrieved chunks from both PDF and video sources (ranked and filtered)
Source metadata with timestamps for video
Citation instructions including video links

**Prompt Strategy 1 - Basic (for simple definitions):**
```
You are a machine learning teaching assistant. Use the following retrieved context to answer the question.
Provide clear, concise explanations suitable for students.

Context:
{chunk_1}
Source: {source_1} (PDF, page {page})

{chunk_2}
Source: {source_2} (Video Lecture {lecture_num}, {timestamp_start})
Video Link: {video_url}

Question: {question}

Answer:
```

**Prompt Strategy 2 - Multi-Modal with Video Links (recommended):**
```
You are a machine learning teaching assistant. Answer the question using the provided context from both documents and video lectures.
Include citations and video timestamps where relevant.

Example:
Q: What is gradient descent?
A: Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss function. It computes the gradient of the loss with respect to parameters and moves in the opposite direction [Source: CS229 Notes, p.15].

For a visual explanation, watch this segment: [Video: CS229 Lecture 2, 12:30-15:45] youtube.com/watch?v=xxx&t=750

Now answer the following:

Context:
{retrieved_chunks_with_sources_and_timestamps}

Question: {question}

Include video links when available. Format: [Video: {source_name}, {timestamp_start}-{timestamp_end}] {video_url}

Answer:
```

**Prompt Strategy 3 - Chain-of-Thought (for complex explanations):**
```
You are a machine learning teaching assistant. Answer the question using the provided context from documents and video lectures.
Think step-by-step and explain your reasoning. Reference both theoretical explanations (PDFs) and visual demonstrations (videos).

Context:
{retrieved_chunks_with_sources_and_timestamps}

Question: {question}

Let's think through this step by step:
1. Identify the core concept
2. Explain the mechanism (reference PDF for theory)
3. Provide visual explanation (reference video with timestamp)
4. Give relevant examples
5. Connect to related concepts

Answer:
```

**Citation Format:**

**For PDF sources:**
- Inline: `[Source: Deep Learning Book, Ch. 6, p.180]`
- Or: `[PDF: CS229 Notes, Lecture 4]`

**For Video sources:**
- Inline: `[Video: CS231n Lecture 5, 12:30-15:45]`
- With link: `youtube.com/watch?v=xxx&t=750`
- Or: `[Video: CS224n Lecture 8, 45:20-48:10]`

**Multi-modal responses include:**
- Text explanation with PDF citations
- Direct video links with timestamps for visual learning
- Mathematical formulas from PDFs
- Code examples from documentation
- Diagram references from video slides

---

## 3.8 LLM Response Generation

LLM Role:

Generate grounded explanations using retrieved academic context.

LLM is used as:

Language reasoning engine

NOT as:

Primary knowledge source

---

## 3.9 Gemini API Setup

**Why Gemini Free API:**
- Free tier with generous limits (15 requests/minute for Flash, 2/minute for Pro)
- Excellent quality for technical ML/DL explanations
- Long context window (1M tokens for Pro, perfect for RAG)
- Fast response times
- Easy integration

**API Key Setup:**

1. **Get API Key:**
   - Go to: https://makersuite.google.com/app/apikey
   - Sign in with Google account
   - Create new API key
   - Copy the key (starts with `AIza...`)

2. **Set Environment Variable:**
   ```bash
   # Linux/Mac
   export GOOGLE_API_KEY="your-api-key-here"

   # Windows (Command Prompt)
   set GOOGLE_API_KEY=your-api-key-here

   # Windows (PowerShell)
   $env:GOOGLE_API_KEY="your-api-key-here"
   ```

   Or add to `.env` file:
   ```
   GOOGLE_API_KEY=your-api-key-here
   ```

3. **Install SDK:**
   ```bash
   pip install google-generativeai langchain-google-genai
   ```

**Gemini Models for RAG:**

| Model | Context Window | Speed | Best For | Free Tier Limit |
|-------|---------------|-------|----------|-----------------|
| Gemini 1.5 Flash | 1M tokens | Fast | Quick responses, RAG | 15 req/min |
| Gemini 1.5 Pro | 1M tokens | Medium | Complex explanations | 2 req/min |

**Implementation:**

```python
import google.generativeai as genai
import os

# Configure API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# For RAG with Flash (fast, recommended)
model = genai.GenerativeModel('gemini-1.5-flash')

# For complex explanations requiring more reasoning
# model = genai.GenerativeModel('gemini-1.5-pro')

def generate_response(question, retrieved_chunks):
    # Build prompt with context
    context = "\n\n".join([
        f"Source: {chunk['source']}\n{chunk['text']}"
        for chunk in retrieved_chunks
    ])

    prompt = f"""You are a machine learning teaching assistant.
Use the following context to answer the question.
Cite your sources using [Source: X] format.

Context:
{context}

Question: {question}

Answer:"""

    response = model.generate_content(prompt)
    return response.text
```

**LangChain Integration:**

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,  # Lower for more factual answers
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Use in RAG pipeline
response = llm.invoke(prompt)
```

**Free Tier Usage:**

**Typical RAG query uses:**
- Input: ~500-1000 tokens (question + retrieved chunks)
- Output: ~300-500 tokens (explanation)
- Total: ~1000-1500 tokens per query

**With Gemini 1.5 Flash free tier:**
- 15 requests/minute = 900 queries/hour
- Sufficient for development, testing, and light production use

**When to upgrade to paid:**
- You hit rate limits regularly
- You need higher request volume
- Production deployment with multiple users

**Alternative: Ollama (Local, Free):**

If you prefer complete privacy and no rate limits:

```bash
# Install Ollama from https://ollama.com
ollama pull llama3  # or mistral
ollama run llama3
```

Then use via LangChain:
```python
from langchain_community.llms import Ollama
llm = Ollama(model="llama3")  # Runs locally, no API key needed
```

**Recommendation:**
Start with **Gemini 1.5 Flash free API** for easiest setup and good quality. Upgrade to paid tier or switch to Ollama only if needed.

---

# 4. Knowledge Base Sources

The knowledge base consists of curated academic Machine Learning and Deep Learning material from **both PDF documents and video lecture playlists**.

---

## 4.1 Machine Learning Theory Source (PDF)

Stanford CS229 Machine Learning Notes (PDF)

**Format:** PDF course notes
**Coverage:**
- supervised learning
- linear regression
- logistic regression
- gradient descent
- bias–variance tradeoff
- regularization
- SVM
- EM algorithm

**Role:** Primary ML theory authority

---

## 4.2 Deep Learning Theory Source (PDF)

Deep Learning — Ian Goodfellow (PDF textbook)

**Format:** PDF textbook
**Coverage:**
- neural networks
- backpropagation
- activation functions
- optimization strategies
- regularization techniques
- convolutional networks
- sequence models

**Role:** Primary DL theoretical foundation

---

## 4.3 CNN Architecture Source (VIDEO)

Stanford CS231n: Convolutional Neural Networks (YouTube Video Playlist)

**Format:** Video lectures (YouTube playlist)
**URL:** https://www.youtube.com/playlist?list=3lgR6Z0mVjE
**Coverage:**
- convolution layers
- pooling layers
- receptive fields
- CNN architectures
- ResNet intuition
- transfer learning
- Visual explanations and diagrams

**Role:** Primary CNN architecture authority with visual demonstrations

**Video Advantages:**
- Visual explanations of convolution operations
- Real-time diagram drawings
- Architecture animations
- Step-by-step visual breakdowns

---

## 4.4 NLP and Transformer Architecture Source (VIDEO)

Stanford CS224n: Natural Language Processing (YouTube Video Playlist)

**Format:** Video lectures (YouTube playlist)
**Coverage:**
- word embeddings
- sequence models
- attention mechanism
- transformer architecture
- encoder-decoder models
- BERT and GPT family

**Role:** Primary Transformer architecture authority with lecture explanations

**Video Advantages:**
- Visual attention mechanism demonstrations
- Transformer architecture animations
- Step-by-step computation graphs
- Intuitive explanations of complex architectures

---

## 4.5 Applied Deep Learning Workflow Source (VIDEO)

Stanford CS230: Deep Learning (YouTube Video Playlist)

**Format:** Video lectures (YouTube playlist)
**Coverage:**
- hyperparameter tuning
- debugging neural networks
- training pipelines
- dataset splitting strategies
- Practical tips and tricks

**Role:** Applied deep learning engineering knowledge

**Video Advantages:**
- Practical debugging demonstrations
- Real-world workflow examples
- Industry best practices
- Common pitfalls and solutions

---

## 4.6 Framework Documentation Sources (PDF/Web)

### PyTorch Documentation (PDF/Web)

**Format:** Web documentation + PDF excerpts
**Coverage:**
- tensors
- autograd
- optimizers
- loss functions
- training loops
- Code examples

**Role:** Theory-to-implementation bridge

**Code Examples:**
- Actual implementation snippets
- API documentation
- Best practices

---

### Scikit-learn Documentation (PDF/Web)

**Format:** Web documentation + PDF excerpts
**Coverage:**
- regression models
- SVM
- preprocessing
- evaluation metrics
- Code examples

**Role:** Classical ML implementation understanding

---

## 4.7 Multi-Modal Integration

**PDF Sources Provide:**
- ✅ Mathematical formulations
- ✅ Detailed theoretical explanations
- ✅ Code examples
- ✅ Static diagrams

**Video Sources Provide:**
- ✅ Step-by-step visual explanations
- ✅ Animated demonstrations
- ✅ Instructor intuition and explanations
- ✅ Real-time diagram drawing
- ✅ Direct timestamps for targeted learning

**Combined Benefits:**
- Students get both theoretical depth (PDF) and intuitive explanations (video)
- Direct video links with timestamps for specific concepts
- Multi-modal learning (reading + watching + listening)
- Comprehensive coverage from multiple perspectives

---

# 5. Metadata Structure

Each chunk stored in the vector database contains different metadata based on source type.

### PDF Chunk Metadata

```json
{
  "concept_title": "Gradient Descent",
  "source_name": "Deep Learning Book",
  "source_type": "pdf",
  "topic": "Optimization",
  "difficulty_level": "Intermediate",
  "authority_level": "Primary",
  "text_chunk": "Gradient descent is an optimization algorithm...",
  "chapter": "4.1",
  "page_number": 78
}
```

### Video Chunk Metadata

```json
{
  "concept_title": "Backpropagation Intuition",
  "source_name": "CS231n Lecture 5",
  "source_type": "video",
  "topic": "Neural Networks",
  "difficulty_level": "Intermediate",
  "authority_level": "Primary",
  "text_chunk": "Backpropagation works by computing gradients...",
  "timestamp_start": "12:30",
  "timestamp_end": "15:45",
  "timestamp_seconds_start": 750.0,
  "timestamp_seconds_end": 945.0,
  "lecture_number": 5,
  "video_url": "youtube.com/watch?v=abc123&t=750",
  "transcript": "Backpropagation works by...",
  "has_diagram": true,
  "has_code": false,
  "ocr_text": "∂L/∂w = ..."
}
```

**Complete Metadata Schema:**

| Field | Type | Allowed Values | Purpose |
|-------|------|----------------|---------|
| concept_title | string | Any | Main concept name |
| source_name | string | CS229, Deep Learning Book, CS231n, CS224n, CS230, PyTorch, Scikit-learn | Document identifier |
| source_type | enum | pdf, video, documentation | Media type |
| topic | string | ML Theory, DL Theory, CNNs, Transformers, Optimization, etc. | High-level category |
| difficulty_level | enum | Beginner, Intermediate, Advanced | Content complexity |
| authority_level | enum | Primary, Secondary, Supplementary | Source reliability |
| timestamp_start | string | HH:MM:SS | Video start time (video only) |
| timestamp_end | string | HH:MM:SS | Video end time (video only) |
| video_url | string | YouTube URL | Direct link to timestamp (video only) |
| lecture_number | int | Any | Lecture number (video only) |
| page_number | int | Any | Page reference (PDF only) |
| chapter | string | Any | Chapter reference (PDF only) |
| has_diagram | boolean | true/false | Contains visual content (video) |
| has_code | boolean | true/false | Contains code examples |
| ocr_text | string | Any | Text extracted from video frames |

**Metadata Filtering Examples:**

```python
# Filter by source type (video vs PDF)
query = "neural networks" + " AND source_type:video"

# Filter by difficulty
query = "backpropagation" + " AND difficulty:Intermediate"

# Filter by video source
query = "convolution" + " AND source:CS231n" + " AND source_type:video"

# Get video explanations only
query = "transformers" + " AND source_type:video"

# Combine filters
query = "regularization" + " AND difficulty:Advanced" + " AND source:CS229"
```

**Authority Level Usage:**
- **Primary**: Stanford courses, foundational textbooks (highest weight)
- **Secondary**: Framework documentation (PyTorch, Scikit-learn)
- **Supplementary**: Blog posts, tutorials

**Metadata in Retrieval:**
- Boost video chunks for visual concepts (CNNs, architecture diagrams)
- Boost PDF chunks for mathematical formulations
- Filter by source type for different learning preferences
- Use timestamps for direct video linking in responses

---

# 6. Evaluation Methodology

## 6.1 Evaluation Dataset

**Size:** 30-50 conceptual ML/DL questions spanning:
- Basic concepts (10 questions): definitions, basic mechanisms
- Intermediate concepts (20 questions): algorithms, comparisons, intuition
- Advanced concepts (10-20 questions): derivations, architectural choices

**Example Questions by Difficulty:**

| Difficulty | Example Questions |
|------------|-------------------|
| Beginner | What is a loss function? What is overfitting? |
| Intermediate | Explain bias-variance tradeoff. How does Adam optimizer work? |
| Advanced | Why does ResNet solve vanishing gradients? Derive backpropagation equations. |

**Each evaluation sample contains:**
```json
{
  "question": "Explain the bias-variance tradeoff",
  "ground_truth_answer": "Key points to cover...",
  "topic": "ML Theory",
  "difficulty": "Intermediate",
  "expected_sources": ["CS229", "Deep Learning Book"],
  "key_concepts": ["bias", "variance", "model complexity", "overfitting", "underfitting"]
}
```

## 6.2 Evaluation Metrics

**Retrieval Metrics:**
- **Recall@K**: Percentage of relevant chunks in top-K retrieved (K=3,5,10)
  - Good: >0.8 for K=5
- **Precision@K**: Percentage of retrieved chunks that are relevant
  - Good: >0.7 for K=3
- **Mean Reciprocal Rank (MRR)**: 1/(rank of first relevant chunk)
  - Good: >0.85
- **Normalized Discounted Cumulative Gain (NDCG@K)**: Ranking quality
  - Good: >0.75 for K=5

**Video-Specific Metrics:**
- **Timestamp Accuracy**: Are retrieved video timestamps relevant to the query?
  - Measure: Manual verification of video segments
  - Target: >0.85
- **Video Relevance**: Do video segments actually explain the concept?
  - Measure: Human rating of video explanations
  - Target: >0.8
- **Multi-Modal Coverage**: Does retrieval use both PDF and video sources?
  - Measure: Percentage of queries with both source types
  - Target: >60% of queries have both PDF and video results
- **Segment Duration Quality**: Are video segments appropriately sized?
  - Measure: Optimal 2-5 minutes per segment
  - Target: >80% of segments in optimal range

**Reranking Metrics:**
- **Reranking Lift**: (NDCG_after_rerank - NDCG_before) / NDCG_before
  - Good: >10% improvement
- **Precision@3 (after rerank)**: Should be >0.85
- **Source Type Balance**: Does reranking maintain good PDF/Video mix?
  - Target: Not biased toward one source type

**Answer Quality Metrics:**
- **Faithfulness**: Does answer use only retrieved context?
  - Measure: Manual annotation or NLI-based classifier
  - Target: >0.9
- **Correctness**: Does answer match ground truth?
  - Measure: LLM-as-judge or human evaluation
  - Target: >0.85
- **Completeness**: Are all key concepts covered?
  - Measure: ROUGE-L against ground truth
  - Target: >0.75
- **Citation Accuracy**: Are sources and timestamps correctly cited?
  - Target: >0.95
- **Video Link Quality**: Do video timestamps point to relevant explanations?
  - Measure: Human verification
  - Target: >0.9

**Multi-Modal Learning Effectiveness:**
- **Comprehension**: Do users understand concepts better with video?
  - Measure: Pre/post quiz scores with and without video
  - Target: +15-20% improvement with video
- **Engagement**: Do users click on video links?
  - Measure: Video link click-through rate
  - Target: >60% for visual concepts
- **Preference**: Do users prefer multi-modal answers?
  - Measure: User survey
  - Target: >80% prefer multi-modal

**Hallucination Metrics:**
- **Hallucination Rate**: Percentage of responses with unsupported claims
  - Measure: NLI-based detection or human annotation
  - Target: <5%
- **Entity F1**: Precision/recall of technical terms in answer vs ground truth
  - Target: >0.8
- **Timestamp Hallucination**: Fake or incorrect video timestamps
  - Measure: Verify timestamps exist and are relevant
  - Target: 0% (must be accurate)

## 6.3 Baseline Comparisons

Compare against:
1. **LLM-only (No Retrieval)**: LLM knowledge only, no external sources
2. **PDF-only RAG**: Naive RAG with PDF sources only
3. **Video-only RAG**: RAG with video transcripts only
4. **Naive Multi-Modal RAG**: PDF + Video, no reranking
5. **Modular RAG (no rerank)**: Multi-modal pipeline but skip reranking
6. **Full system**: Multi-Modal RAG + Reranker

**Expected Improvements:**
- vs LLM-only: +40-50% in factual accuracy (grounded in sources)
- vs PDF-only RAG: +25-35% in comprehension (visual explanations)
- vs Video-only RAG: +20-30% in precision (PDF provides mathematical depth)
- Naive Multi-Modal vs Full: +15-20% with reranking
- Multi-modal vs single-source: +30-40% in user satisfaction

**Multi-Modal Advantage:**
- **PDF-only**: Good for math and theory, bad for visual intuition
- **Video-only**: Good for intuition, bad for precise formulas
- **Multi-modal**: Best of both, comprehensive understanding

**Ablation Studies:**
- Remove video timestamps: How much does this hurt user experience?
- Remove PDF sources: Can video alone suffice?
- Remove reranking: How much does precision drop?
- Remove temporal coherence: Does segment flow matter?
- Remove cross-modal reranking: Can simple rules work?
- Different video chunk sizes: What's optimal?

## 6.4 A/B Testing Framework

**Test Setup:**
- Split evaluation dataset: 70% test, 30% validation
- Run each question through all baselines
- Blind evaluation by domain expert
- Statistical significance testing (paired t-test)

**Success Criteria:**
- Full system outperforms all baselines with p<0.05
- Retrieval recall@5 >0.8
- Answer faithfulness >0.9
- Hallucination rate <5%  

---

# 7. Novel Contributions

This project introduces three novel contributions to advance multi-modal educational RAG systems:

## 7.1 Timestamp-Aware Video RAG

**Problem:**
Most video RAG systems chunk videos at fixed time intervals (e.g., every 3 minutes), which can:
- Split related concepts across chunks
- Include multiple unrelated topics in one chunk
- Miss natural topic boundaries in lectures

**Our Solution:**
Smart video chunking using topic boundary detection:
- Analyze video transcripts for topic shifts
- Detect slide changes using frame comparison
- Identify natural lecture segments (intro, main topic, examples)
- Chunk at semantic boundaries instead of fixed intervals

**Technical Approach:**
```python
# Topic Segmentation Algorithm:
1. Extract transcript with timestamps
2. Use TextTiling or similar algorithm to detect topic boundaries
3. Combine with slide change detection (frame comparison)
4. Identify coherent segments (2-5 minutes each)
5. Store segments with topic labels and timestamps
```

**Expected Benefits:**
- More semantically coherent chunks
- Better retrieval precision
- Segments that align with how lectures are structured

**Evaluation:**
- Compare with fixed-interval chunking
- Measure retrieval precision improvement
- User ratings of segment relevance

---

## 7.2 Temporal Coherence in Multi-Modal Retrieval

**Problem:**
When retrieving multiple video segments, traditional RAG might return:
- Segments that jump around in time
- Disconnected explanations that don't flow
- Segments from different parts of a lecture that confuse students

**Our Solution:**
Ensure retrieved video segments tell a coherent story:
- Build temporal dependency graph of topics
- When retrieving multiple segments, maintain temporal flow
- Score retrieval sets for coherence
- Prefer segments that follow logical progression

**Technical Approach:**
```python
# Temporal Coherence Algorithm:
1. Build graph: Topic A → Topic B → Topic C (as they appear in lecture)
2. When query requires multiple segments:
   - Find relevant segments
   - Select path through graph that maintains flow
   - Score: relevance + coherence
3. Return segments that tell coherent story
```

**Example:**
```
Query: "Explain backpropagation step by step"

Without coherence:
- Segment 1: Lecture 5, 12:30 (gradient descent basics)
- Segment 2: Lecture 2, 45:00 (optimization intro) ← jumps back
- Segment 3: Lecture 5, 15:00 (backprop details)

With coherence:
- Segment 1: Lecture 5, 12:30 (backprop overview)
- Segment 2: Lecture 5, 15:00 (computation graph)
- Segment 3: Lecture 5, 18:00 (chain rule application)
```

**Expected Benefits:**
- Easier to follow explanations
- Better learning outcomes
- More natural video viewing experience

**Evaluation:**
- User study: "Do these segments flow logically?"
- Compare coherence scores with vs without
- Measure learning effectiveness

---

## 7.3 Cross-Modal Reranking for Educational Content

**Problem:**
Different types of queries benefit from different modalities:
- "How does convolution work?" → Video (visual explanation)
- "What's the mathematical formula?" → PDF (math notation)
- "How do I implement this?" → Code documentation

Current systems don't adapt modality selection to query type.

**Our Solution:**
Learn which modality (PDF vs Video) works best for different query types:
- Train classifier to predict optimal modality
- Extract features from queries (concept type, complexity, keywords)
- Adjust retrieval weights based on modality prediction
- Rerank results to favor predicted best modality

**Technical Approach:**
```python
# Cross-Modal Reranking Algorithm:
1. Feature extraction from query:
   - Concept category (architecture, math, implementation, etc.)
   - Complexity indicators (formula, code, "how to", etc.)
   - Domain keywords (convolution, backprop, etc.)

2. Modality prediction:
   - Train classifier: Query → Best Modality
   - Training data: Manual labels or heuristic rules

3. Adaptive retrieval:
   - If video predicted: Boost video chunks by 20-30%
   - If PDF predicted: Boost PDF chunks by 20-30%
   - Rerank based on adjusted scores

4. Return results favoring predicted best modality
```

**Training Data Creation:**
```python
# Heuristic rules (start here):
query_features = {
    "has_visual_keywords": ["diagram", "show", "visual", "architecture"],
    "has_math_keywords": ["formula", "equation", "derive", "proof"],
    "has_code_keywords": ["implement", "code", "how to", "api"]
}

# Map to modality:
if visual_keywords > threshold: predict "video"
elif math_keywords > threshold: predict "pdf"
elif code_keywords > threshold: predict "documentation"
```

**Expected Benefits:**
- Better modality selection for queries
- Improved user satisfaction
- Higher learning effectiveness

**Evaluation:**
- Accuracy of modality prediction
- User satisfaction with modality choices
- Comparison with uniform modality weighting

---

## 7.4 Contribution Summary

| Contribution | Innovation | Difficulty | Impact | Timeline |
|--------------|-----------|-----------|--------|----------|
| **Timestamp-Aware Video RAG** | Smart video chunking using topic boundaries | ⭐⭐ (Easy) | Medium | 2-3 weeks |
| **Temporal Coherence** | Coherent multi-segment retrieval with flow | ⭐⭐⭐ (Medium) | High | 3-4 weeks |
| **Cross-Modal Reranking** | Adaptive modality selection based on query type | ⭐⭐⭐⭐ (Hard) | High | 4-5 weeks |

**Combined Impact:**
- All three contributions work together to create a coherent, intelligent multi-modal educational RAG system
- Each addresses a specific limitation in current RAG systems
- Together, they significantly advance the state-of-the-art in educational AI

---

# 8. Deployment & Production Considerations

## 7.1 API Design

**Recommended Endpoint Structure:**
```
POST /api/v1/query
{
  "question": "Explain backpropagation",
  "difficulty": "intermediate",  // optional
  "sources": ["CS229", "Deep Learning Book"],  // optional
  "max_chunks": 3  // optional
}

Response:
{
  "answer": "Backpropagation is...",
  "sources": [
    {"name": "Deep Learning Book", "chapter": "6", "relevance": 0.92},
    {"name": "CS229", "section": "4.3", "relevance": 0.87}
  ],
  "chunks_used": 2,
  "confidence": 0.89,
  "processing_time_ms": 450
}
```

**Additional Endpoints:**
- `GET /api/v1/health` - System health check
- `GET /api/v1/sources` - List available knowledge sources
- `POST /api/v1/feedback` - Submit answer feedback for improvement
- `GET /api/v1/stats` - Usage statistics (cached, not real-time)

## 7.2 Performance Optimization

**Latency Targets:**
- Simple query: <500ms
- Complex query: <1500ms
- P95 latency: <2000ms

**Optimization Strategies:**
1. **Caching Layer**
   - Cache frequent queries (Redis/Memcached)
   - TTL: 24 hours for concept definitions
   - Cache key: hash of question + difficulty + sources

2. **Batch Processing**
   - Batch embedding generation for multiple queries
   - Batch reranking for efficiency
   - Target: Process 10-50 queries per batch

3. **Asynchronous Pipeline**
   - Return intermediate results quickly
   - Full results via WebSocket or polling
   - Useful for complex multi-step explanations

4. **Index Optimization**
   - Use IVF indices for >100K chunks
   - GPU acceleration for FAISS search
   - Pre-compute embeddings for common queries

## 7.3 Scalability Considerations

**Vertical Scaling:**
- Single instance handles: ~100 QPS with caching
- Memory: 2GB base + 0.5GB per 100K chunks
- CPU: 4 cores for ~50 QPS, 8 cores for ~100 QPS

**Horizontal Scaling:**
- Stateless API servers (load balancer friendly)
- Shared vector store (Redis FAISS or dedicated vector DB)
- Separate embedding service (can be shared across apps)

**Resource Estimation:**
| Chunks | Memory | CPU (per query) | Latency |
|--------|--------|-----------------|---------|
| 10K | 100MB | 50ms | 200ms |
| 100K | 1GB | 100ms | 400ms |
| 1M | 10GB | 200ms | 800ms |

## 7.4 Monitoring & Observability

**Key Metrics to Track:**
- Query latency (p50, p95, p99)
- Cache hit rate
- Retrieval recall (sampled queries)
- Answer confidence scores
- User feedback (thumbs up/down)
- Error rates by module

**Logging:**
- Query logs (anonymized)
- Retrieval details (chunks retrieved, scores)
- Reranking scores
- LLM prompts and responses
- Error stack traces

---

# 8. Error Handling & Edge Cases

## 8.1 Error Scenarios

**No Relevant Chunks Found:**
- Trigger: All rerank scores <0.3 or retrieval returns <2 chunks
- Fallback: Expand search to top-20 chunks, re-rerank
- Ultimate fallback: "I couldn't find specific information in my knowledge base. This might be outside the covered topics."

**Ambiguous Queries:**
- Trigger: Multiple chunks with similar scores (<0.1 difference)
- Response: "Your question could refer to multiple concepts. Could you clarify?"
- Example: "What is regularization?" → L1, L2, dropout, etc.

**Out-of-Domain Questions:**
- Trigger: Questions about non-ML/DL topics
- Detection: Low max similarity score (<0.4) or classifier
- Response: "I specialize in ML/DL concepts. This question seems outside my knowledge base."

**PDF Parsing Failures:**
- Trigger: Malformed PDFs, scanned documents
- Handling: Log error, skip document, alert admin
- Recovery: Manual review and OCR if needed

## 8.2 Graceful Degradation

**If Reranker Fails:**
- Fall back to top-K from vector search
- Log error for monitoring
- Continue with reduced quality

**If Gemini API Fails:**
- **Rate Limit Error (429):**
  - Wait with exponential backoff (1s, 2s, 4s, 8s)
  - Retry up to 3 times
  - If still failing: "Service is busy. Please try again in a moment."
- **API Key Error (401):**
  - Check API key configuration
  - Return: "Service configuration error. Please contact admin."
- **Network/Timeout Error:**
  - Retry once after 2 seconds
  - If fails: Return retrieved chunks directly to user
  - Message: "Here are the relevant materials. I couldn't generate a summary due to a connection issue."
- **Other Errors:**
  - Log error details
  - Return retrieved chunks with fallback message
  - Alert admin for investigation

**If Vector Search Fails:**
- Use BM25 keyword search as backup
- Slower but functional

**Gemini-Specific Error Handling:**
```python
import time
from google.generativeai import GenerateContentError

def generate_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except GenerateContentError as e:
            if "rate limit" in str(e).lower():
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait_time)
                continue
            else:
                raise  # Re-raise non-rate-limit errors
    return None  # All retries failed
```

---

# 9. Maintenance & Updates

## 9.1 Knowledge Base Updates

**When to Update:**
- New course releases (quarterly)
- New textbook editions (annually)
- Bug fixes in source materials (as needed)

**Update Process:**
1. Add new PDFs to source directory
2. Run ingestion pipeline (idempotent)
3. Detect new/modified documents via hash comparison
4. Re-chunk and re-embed only changed documents
5. Update FAISS index (incremental if possible)
6. Run smoke tests on sample queries
7. Deploy update with rollback plan

**Versioning:**
- Track document versions in metadata
- Maintain previous index for rollback
- Tag releases: `v1.0.0-cs229-2024`

## 9.2 Model Updates

**Embedding Model Updates:**
- Frequency: Annually or when significantly better models available
- Testing: Benchmark on evaluation dataset before switching
- Rollout: A/B test against current model
- Rollback: Keep previous model index available

**Reranker Updates:**
- Similar process to embedding models
- More conservative updates (reranking quality critical)

## 9.3 Quality Assurance

**Automated Checks:**
- Daily smoke tests on 10 sample queries
- Weekly full evaluation run (30-50 questions)
- Regression detection (>5% drop in metrics = alert)

**Human Review:**
- Monthly review of low-confidence answers
- User feedback analysis weekly
- Update evaluation dataset quarterly

---

# 10. Why Modular RAG + Reranker

**Advantages Over Naive RAG:**

**Improved Retrieval Precision**
- Reranking adds 10-30% improvement in retrieval quality
- Cross-encoders better understand query-context relevance than bi-encoders
- Metadata filtering enables targeted knowledge access

**Better Explanation Grounding**
- All answers cite specific academic sources
- Reduced reliance on LLM internal knowledge
- Higher factual accuracy for technical content

**Reduced Hallucinations**
- Strict context boundaries in prompts
- Reranking filters irrelevant context that might confuse LLM
- Faithfulness scores >0.9 achievable

**Metadata-Aware Retrieval**
- Filter by difficulty for adaptive learning
- Filter by source for authoritative explanations
- Boost primary sources for higher quality

**Scalable Architecture Design**
- Independent modules can be optimized separately
- Easy to add new sources without re-architecting
- Can swap embedding/reranker/LLM components independently

**Higher Academic Robustness**
- Grounded in peer-reviewed academic sources
- Consistent explanations across queries
- Suitable for educational use cases

**Performance Comparison:**

| Metric | Naive RAG | Modular RAG | + Reranker |
|--------|-----------|-------------|------------|
| Precision@3 | 0.55 | 0.65 | 0.82 |
| Faithfulness | 0.75 | 0.82 | 0.91 |
| Hallucination Rate | 15% | 10% | 4% |
| Avg Latency | 300ms | 350ms | 450ms |

---

# 11. Expected Output Capabilities

The system provides comprehensive, multi-modal answers combining text, video, diagrams, and code.

**Example Query:** "Explain how backpropagation works"

**System Response:**

```
**Backpropagation Explained**

Backpropagation is an algorithm for training neural networks by computing gradients efficiently.
It works by applying the chain rule recursively from the output layer backwards.

**Mathematical Foundation:**
The gradient of the loss L with respect to weight w is computed as:
∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w

[Source: Deep Learning Book, Ch. 6, p.205]

**Step-by-Step Explanation:**
1. Forward pass: Compute activations
2. Compute loss at output
3. Backward pass: Propagate gradients backward
4. Update weights using computed gradients

**Visual Explanation:**
For an intuitive visual explanation of how gradients flow through the network,
watch this segment: [Video: CS231n Lecture 4, 15:30-22:45]
🔗 youtube.com/watch?v=abc123&t=930

The video shows:
- Real-time gradient computation graph
- Animation of backward pass
- Visual intuition for chain rule application

**Implementation (PyTorch):**
```python
loss = criterion(output, target)
loss.backward()  # Computes gradients via backprop
optimizer.step()  # Updates weights
```
[Source: PyTorch Documentation, Autograd section]

**Common Pitfalls:**
- Vanishing gradients in deep networks [Video: CS231n Lecture 5, 8:15-12:00]
- Exploding gradients and gradient clipping [PDF: CS229 Notes, p.89]

**Related Concepts:**
- Gradient descent optimization
- Computational graphs
- Chain rule in calculus
```

---

## Question Types Supported

**Basic Concepts:**
- "What is gradient descent?"
- "Explain the difference between classification and regression"
- "What is a neural network?"

**Intermediate Concepts:**
- "Explain backpropagation step by step"
- "Difference between CNN and MLP"
- "How does the Adam optimizer work?"

**Advanced Concepts:**
- "Why does ResNet solve vanishing gradients?"
- "Why transformers outperform RNNs"
- "Explain bias-variance tradeoff with examples"

**Comparative Questions:**
- "L1 vs L2 regularization: when to use which?"
- "CNN vs Vision Transformer: tradeoffs"
- "SGD vs Adam: convergence properties"

**Implementation Questions:**
- "How to implement dropout in PyTorch?"
- "Scikit-learn SVM parameters explained"
- "Batch normalization in practice"

**Visual/Architecture Questions:**
- "Show me how attention mechanism works visually"
- "Explain transformer architecture with diagrams"
- "How does convolution operation work on images?"

---

## Response Features

**All answers include:**
- ✅ Clear, step-by-step explanations
- ✅ PDF citations with page/chapter references
- ✅ **Video links with precise timestamps** for visual explanations
- ✅ Mathematical formulations from textbooks
- ✅ Code examples from documentation
- ✅ Diagram references from video lectures
- ✅ Difficulty-appropriate depth
- ✅ Connections to related concepts

**Multi-Modal Learning Experience:**
- **Read:** Detailed text explanations from PDFs
- **Watch:** Targeted video segments with exact timestamps
- **Learn:** Visual diagrams and animations from lectures
- **Practice:** Code examples from framework documentation
- **Understand:** Mathematical foundations from textbooks

---

# 12. Implementation Roadmap

**Team Structure: 4 people working in parallel**

**Phase 1: Base System (Weeks 1-2) - All 4 people**
- [ ] **Person 1:** Set up Gemini API, core RAG architecture
- [ ] **Person 2:** PDF processing pipeline (LangChain, chunking)
- [ ] **Person 3:** Video download + Whisper transcription setup
- [ ] **Person 4:** Embeddings + FAISS vector index setup
- [ ] **All:** Test basic multi-modal retrieval with 5-10 queries

**Checkpoint:** Working multi-modal RAG with PDF + video retrieval

---

**Phase 2: Novel Contributions (Weeks 3-7) - Parallel Development**

**Person 1: Timestamp-Aware Video RAG (2-3 weeks)**
- [ ] Implement topic boundary detection (TextTiling algorithm)
- [ ] Build slide change detection (frame comparison)
- [ ] Create smart video chunking at semantic boundaries
- [ ] Store chunks with topic labels and timestamps
- [ ] Test and optimize chunking quality
- [ ] Week 6-7: Help with integration, support others

**Person 2: Temporal Coherence (3-4 weeks)**
- [ ] Build temporal dependency graph of lecture topics
- [ ] Implement coherence scoring algorithm
- [ ] Create retrieval logic that maintains temporal flow
- [ ] Develop multi-segment selection with coherence
- [ ] Test and optimize coherence improvements
- [ ] Week 6-7: Help with integration, support others

**Person 3: Cross-Modal Reranking (4-5 weeks)**
- [ ] Design feature extraction from queries
- [ ] Create training data (heuristic rules or manual labels)
- [ ] Train modality classifier (video vs PDF vs documentation)
- [ ] Implement adaptive weight adjustment
- [ ] Integrate reranking into retrieval pipeline
- [ ] Test and optimize modality prediction accuracy
- [ ] Week 6-7: Help with integration, support others

**Person 4: Evaluation & User Study (Weeks 3-7)**
- [ ] Week 3-4: Build evaluation dataset (50-100 questions)
- [ ] Week 5-6: Design user study materials (surveys, pre/post tests)
- [ ] Week 6: Recruit participants (10-15 classmates)
- [ ] Week 7: Prepare evaluation infrastructure and metrics

**Checkpoint:** All 3 contributions implemented independently

---

**Phase 3: Integration (Week 8) - All 4 people**
- [ ] Combine all 3 contributions into single system
- [ ] Ensure components work together without conflicts
- [ ] End-to-end testing with full system
- [ ] Bug fixes and optimization
- [ ] Performance tuning

**Checkpoint:** Complete system with all 3 contributions working

---

**Phase 4: Evaluation (Weeks 9-10) - All 4 people**
- [ ] Run full evaluation (50-100 questions)
- [ ] Ablation studies (test each contribution independently)
- [ ] Compare with all baselines (PDF-only, video-only, naive multi-modal)
- [ ] Statistical analysis of results
- [ ] Create visualizations and analysis

**Checkpoint:** Complete evaluation results with all metrics

---

**Phase 5: User Study (Weeks 11-12) - All 4 people**
- [ ] Run user study sessions (10-15 participants)
- [ ] Collect pre/post test data
- [ ] Gather qualitative feedback
- [ ] Analyze user study results
- [ ] Correlate with system metrics

**Checkpoint:** User study completed and analyzed

---

**Phase 6: Paper Writing (Weeks 13-14) - All 4 people**
- [ ] Week 13:
  - Person 1: System description + Timestamp-Aware section
  - Person 2: Temporal Coherence section
  - Person 3: Cross-Modal Reranking section
  - Person 4: Evaluation + User Study sections
  - All: Introduction, Related Work (collaborative)
- [ ] Week 14:
  - All: Review, revise, polish
  - All: Create figures and tables
  - All: Final formatting and submission

**Checkpoint:** Workshop paper submitted (6-8 pages)

---

**Total Timeline: 14 weeks (one semester)**

**Alternative: MVP Timeline (10-11 weeks)**
- Skip extensive user study (5-8 participants)
- Smaller evaluation dataset (30-50 questions)
- Focus on workshop paper instead of full conference paper

---

**Weekly Checkpoints:**
- **Week 2:** Base system working
- **Week 4:** First contribution complete
- **Week 7:** All contributions implemented
- **Week 8:** Full system integrated
- **Week 10:** Evaluation complete
- **Week 12:** User study complete
- **Week 14:** Paper submitted

---

# 13. Next Steps

1. **Setup Development Environment**
   - Get Gemini API key from https://makersuite.google.com/app/apikey
   - Set GOOGLE_API_KEY environment variable
   - Install dependencies (including video processing tools)
   - Download sample PDFs (CS229 notes, Deep Learning Book chapters)
   - Download sample videos (1-2 lectures from CS231n)
   - Set up project structure

2. **Build MVP (Multi-Modal)**
   - Implement PDF processing pipeline (CS229 notes)
   - Implement video processing for 1 lecture (CS231n Lecture 1)
   - Test basic multi-modal retrieval with 5 questions
   - Validate video timestamps are accurate
   - Test end-to-end flow

3. **Iterate and Enhance**
   - Add more video lectures (full playlist)
   - Add reranking module
   - Improve prompt construction with video links
   - Add more PDF sources
   - Implement metadata filtering

4. **Evaluate and Deploy**
   - Run full evaluation (30-50 questions)
   - Conduct user study with classmates
   - Compare PDF-only vs Video-only vs Multi-modal
   - Set up monitoring
   - Deploy to production

**Recommended Starting Point:**
1. Start with CS229 PDF notes only (week 1-2)
2. Add video processing for CS231n Lecture 1 (week 3-4)
3. Test multi-modal retrieval with 5 core ML concepts (week 5)
4. Once working, scale to full playlists and add reranking

---

*Document Version: 3.1 - Three Contributions Edition*
*Last Updated: 2026-03-29*
*Status: Multi-modal RAG architecture with three novel contributions for publication-ready educational AI system*

**Three Novel Contributions:**
- ✅ **Timestamp-Aware Video RAG**: Smart video chunking using topic boundary detection
- ✅ **Temporal Coherence**: Retrieved video segments maintain logical flow and tell coherent stories
- ✅ **Cross-Modal Reranking**: Adaptive modality selection (PDF vs Video) based on query type

**Key Features:**
- ✅ Multi-source ingestion (PDF textbooks + YouTube video playlists)
- ✅ Video processing pipeline (Whisper transcription + frame extraction)
- ✅ Temporal segmentation with timestamp-aware retrieval
- ✅ Multi-modal response generation (text + video links + diagrams + code)
- ✅ Gemini Free API integration for LLM capabilities
- ✅ Cross-encoder reranking for precision
- ✅ Comprehensive evaluation methodology including video-specific metrics
- ✅ Production-ready architecture with error handling and deployment considerations
- ✅ 14-week implementation timeline for 4-person team
- ✅ Publication-ready for workshop and educational AI venues
