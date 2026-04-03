# Kaggle Video Processing Workflow

**Purpose:** Process Stanford course videos on Kaggle and integrate with local RAG system.

**Last Updated:** 2026-04-04

---

## Overview

This workflow processes video lectures on Kaggle (free GPU) and transfers the results back to your local system for integration with the PDF-based RAG system.

## Architecture

```
Kaggle (Free GPU)          Local System
=============              =============
1. Download videos    →    (skip - done on Kaggle)
2. Transcribe        →    (skip - done on Kaggle)
3. Extract frames    →    (skip - done on Kaggle)
4. Detect slides     →    (skip - done on Kaggle)
5. Run OCR          →    (skip - done on Kaggle)
6. Create chunks    →    Transfer chunks.zip
                      ↓
7. Load chunks      →    scripts/load_video_chunks.py
8. Build index      →    scripts/build_unified_index.py
9. Test retrieval   →    Ready for RAG!
```

---

## Phase 1: Kaggle Processing

### Step 1: Create Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/notebooks)
2. Click "New Notebook"
3. Choose "GPU P100" or "GPU T4" (free)
4. Name it: `stanford-courses-rag`

### Step 2: Upload Scripts

Upload all scripts from `video-processing/` directory to Kaggle:
- download_videos.py
- transcribe.py
- extract_frames.py
- detect_slides.py
- run_ocr.py
- chunk_videos.py
- requirements.txt

### Step 3: Install Dependencies

In the first cell of your Kaggle notebook:

```python
!pip install -q yt-dlp whisper-openai opencv-python pillow pytesseract transformers torch sentence-transformers
```

### Step 4: Configure Videos to Process

Edit `download_videos.py` to specify which videos to process:

```python
# Stanford CS231n: CNNs
VIDEO_IDS = {
    "cs231n": [
        "NfnWJUyUJYU",  # Lecture 1
        "siu3J-Yuv3Q",  # Lecture 2
        "c7M1kmKqoEg",  # Lecture 3
        # Add more as needed
    ],
}
```

**Recommended:** Start with 1-2 lectures per course to test the pipeline.

### Step 5: Run Processing Pipeline

Execute scripts in order:

```python
# Step 1: Download videos (10-20 min)
!python download_videos.py

# Step 2: Transcribe with Whisper (5-10 min per video)
!python transcribe.py

# Step 3: Extract frames (5-10 min per video)
!python extract_frames.py

# Step 4: Detect slide changes (10-15 min per video)
!python detect_slides.py

# Step 5: Run OCR (15-20 min per video)
!python run_ocr.py

# Step 6: Create smart chunks (5-10 min)
!python chunk_videos.py
```

### Step 6: Verify Outputs

Check that the following files were created:

```python
# List transcript files
!ls -lh transcripts/

# List chunk files
!ls -lh chunks/

# Check a sample chunk
import json
with open('chunks/NfnWJUyUJYU_chunks.json') as f:
    data = json.load(f)
    print(f"Total chunks: {len(data['chunks'])}")
    print(f"Sample chunk: {data['chunks'][0]}")
```

### Step 7: Prepare for Download

Zip the important files:

```python
# Zip chunks for download
!zip -r video_chunks.zip chunks/

# Also zip transcripts if needed
!zip -r transcripts.zip transcripts/
```

### Step 8: Download to Local

1. Go to "Output" section in Kaggle
2. Download `video_chunks.zip`
3. Extract to: `config/data/chunks/`

---

## Phase 2: Local Integration

### Step 1: Organize Downloaded Files

Extract the downloaded zip file:

```bash
# Create chunks directory if it doesn't exist
mkdir -p config/data/chunks

# Extract video chunks
unzip video_chunks.zip -d config/data/chunks/
```

Expected structure:
```
config/data/chunks/
├── pdf_chunks.json          (already exists from PDF processing)
├── NfnWJUyUJYU_chunks.json  (from Kaggle)
├── siu3J-Yuv3Q_chunks.json  (from Kaggle)
└── ...
```

### Step 2: Load Video Chunks

Run the video chunk loader:

```bash
python scripts/load_video_chunks.py
```

This will:
- Find all video chunk files
- Validate chunks
- Normalize to RAG system format
- Save to: `config/data/chunks/video_chunks.json`

**Expected output:**
```
============================================================
Video Chunk Loader
============================================================
Found 3 video chunk files
Processing: NfnWJUyUJYU_chunks.json
  Loaded 12 chunks from NfnWJUyUJYU_chunks.json
...
Total chunks: 36
Video chunk loading complete!
```

### Step 3: Build Unified Index

Combine PDF and video chunks into a single index:

```bash
python scripts/build_unified_index.py
```

This will:
- Load PDF chunks (162 chunks)
- Load video chunks (36+ chunks)
- Generate embeddings for all chunks
- Create unified FAISS index
- Test retrieval with sample queries

**Expected output:**
```
============================================================
Unified Index Builder
============================================================
Loaded 162 PDF chunks
Loaded 36 video chunks
Combined 162 PDF + 36 video = 198 total chunks

============================================================
Unified Index Statistics
============================================================
Total chunks: 198
PDF chunks: 162
Video chunks: 36
Embedding dimension: 384

Chunk type breakdown:
  PDF: 162 (81.8%)
  Video: 36 (18.2%)
```

### Step 4: Test Unified Retrieval

Create a test script to verify the unified system works:

```python
# test_unified_retrieval.py
from src.base_rag.retriever import MultiModalRetriever

# Load unified index
retriever = MultiModalRetriever(
    index_dir="config/data/unified_index"
)

# Test queries
queries = [
    "What is a neural network?",
    "Explain backpropagation with visual examples",
    "How do CNNs work?",
]

for query in queries:
    print(f"\nQuery: {query}")
    results = retriever.retrieve(query, k=5)

    for i, result in enumerate(results[:3], 1):
        chunk_type = result.get('chunk_type', 'unknown').upper()

        if chunk_type == 'VIDEO':
            source = result.get('source', 'Unknown')
            time = result.get('start_time', 0)
            print(f"  {i}. [{chunk_type}] {source} @ {time:.0f}s")
        else:
            source = result.get('source', 'Unknown')
            page = result.get('page', 'N/A')
            print(f"  {i}. [{chunk_type}] {source} p.{page}")
```

---

## Phase 3: Production Scale

### Processing More Videos

Once the pipeline works for 1-2 lectures, scale up:

**On Kaggle:**
```python
# Add more video IDs
VIDEO_IDS = {
    "cs231n": [
        "NfnWJUyUJYU",  # Lecture 1
        "siu3J-Yuv3Q",  # Lecture 2
        "c7M1kmKqoEg",  # Lecture 3
        # Add lectures 4-10...
    ],
    "cs224n": [
        # Add CS224n lectures...
    ],
    "cs230": [
        # Add CS230 lectures...
    ],
}
```

**Target: 500+ video chunks**
- ~10-15 lectures per course
- ~3-5 chunks per lecture
- ~30-50 chunks per course

### Update Local System

Each time you download new chunks from Kaggle:

```bash
# 1. Extract new chunks to config/data/chunks/
unzip new_video_chunks.zip -d config/data/chunks/

# 2. Reload all video chunks
python scripts/load_video_chunks.py

# 3. Rebuild unified index
python scripts/build_unified_index.py
```

---

## Troubleshooting

### Kaggle Issues

**Out of Memory (OOM)**
```python
# Solution: Process fewer videos at once
VIDEO_IDS = {
    "cs231n": ["NfnWJUyUJYU"],  # Just 1 video
}

# Or use smaller Whisper model
# In transcribe.py, change:
WHISPER_MODEL = "tiny"  # Instead of "base"
```

**Slow Processing**
```python
# Use smaller Whisper model
WHISPER_MODEL = "tiny"  # 5x faster

# Extract fewer frames
EXTRACT_INTERVAL = 10  # Instead of 5

# Skip OCR (optional)
RUN_OCR = False
```

**Video Download Fails**
```python
# Use yt-dlp with retries
import yt_dlp

ydl_opts = {
    'format': 'bestvideo+bestaudio/best',
    'retries': 10,
    'fragment_retries': 10,
}
```

### Local Integration Issues

**Chunks Not Found**
```bash
# Check chunks directory
ls -la config/data/chunks/

# Verify video chunks exist
ls config/data/chunks/*_chunks.json

# Check video_chunks.json was created
ls config/data/chunks/video_chunks.json
```

**Index Build Fails**
```python
# Check chunk format
import json
with open('config/data/chunks/video_chunks.json') as f:
    chunks = json.load(f)
    print(f"Total chunks: {len(chunks)}")
    print(f"Sample: {chunks[0]}")

# Verify required fields exist
required = ['chunk_id', 'text', 'source', 'chunk_type']
for field in required:
    assert field in chunks[0], f"Missing field: {field}"
```

**Retrieval Not Working**
```python
# Test index directly
from src.base_rag.retriever import MultiModalRetriever

retriever = MultiModalRetriever(
    index_dir="config/data/unified_index"
)

# Check index stats
stats = retriever.get_index_stats()
print(stats)

# Test simple query
results = retriever.retrieve("neural network", k=3)
print(f"Retrieved {len(results)} results")
```

---

## Performance Benchmarks

### Kaggle Processing Time

**Per 1-hour video:**
- Download: 5-10 minutes
- Transcription: 5-10 minutes (GPU, base model)
- Frame extraction: 5-10 minutes
- Slide detection: 10-15 minutes
- OCR: 15-20 minutes
- Chunking: 5-10 minutes

**Total: ~1-1.5 hours per hour of video**

**For 10 hours of video: ~10-15 hours**

### Local System

**Index building:**
- 200 chunks: ~2-3 minutes
- 500 chunks: ~5-8 minutes
- 1000 chunks: ~10-15 minutes

**Query latency:**
- Single query: ~100-200ms
- Top-5 retrieval: ~200-400ms

---

## Next Steps

After unified index is ready:

1. **Test Contribution 1: Timestamp-Aware Video RAG**
   - Video chunks already use topic boundaries
   - Timestamps are included in metadata
   - Ready for temporal coherence

2. **Implement Contribution 2: Temporal Coherence**
   - Build temporal dependency graph
   - Implement coherence scoring
   - Test with multi-segment queries

3. **Implement Contribution 3: Cross-Modal Reranking**
   - Train modality classifier
   - Implement adaptive retrieval
   - Test with different query types

4. **Evaluation**
   - Create evaluation dataset
   - Run baseline comparisons
   - Measure improvements

---

## Checklist

**Kaggle Processing:**
- [ ] Create Kaggle notebook with GPU
- [ ] Upload video processing scripts
- [ ] Install dependencies
- [ ] Configure video IDs
- [ ] Run processing pipeline
- [ ] Verify chunk outputs
- [ ] Download video_chunks.zip

**Local Integration:**
- [ ] Extract chunks to config/data/chunks/
- [ ] Run scripts/load_video_chunks.py
- [ ] Run scripts/build_unified_index.py
- [ ] Test retrieval with sample queries
- [ ] Verify video chunks appear in results

**Scale Up:**
- [ ] Process more videos on Kaggle
- [ ] Download and integrate new chunks
- [ ] Rebuild unified index
- [ ] Target: 500+ video chunks

---

## File Locations

**Kaggle Output:**
```
kaggle/working/
├── chunks/
│   ├── NfnWJUyUJYU_chunks.json
│   ├── siu3J-Yuv3Q_chunks.json
│   └── ...
└── video_chunks.zip
```

**Local System:**
```
config/data/
├── chunks/
│   ├── pdf_chunks.json              (162 chunks)
│   ├── video_chunks.json            (36+ chunks)
│   ├── NfnWJUyUJYU_chunks.json      (from Kaggle)
│   └── ...
├── unified_index/
│   ├── index.faiss
│   ├── metadata.pkl
│   └── chunk_manifest.json
└── pdf_index/
    └── (existing PDF index)
```

---

**Happy Processing!** 🎥→📝→🧠
