# Video Processing for Multi-Modal RAG

**Purpose:** Process YouTube video lectures from Stanford courses to extract transcripts, frames, and chunks for the RAG system.

**Platform:** Kaggle Notebooks (free GPU access)

**Last Updated:** 2026-03-29

---

## Quick Start

### 1. Create Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/notebooks)
2. Click "New Notebook"
3. Choose "GPU P100" or "GPU T4" (free)
4. Name it: `video-processing-rag`

### 2. Upload Scripts

Upload all Python scripts from this folder to Kaggle:
- `download_videos.py`
- `transcribe.py`
- `extract_frames.py`
- `detect_slides.py`
- `run_ocr.py`
- `chunk_videos.py`

### 3. Install Dependencies

```python
!pip install -q yt-dlp whisper-openai opencv-python pillow pytesseract transformers torch
```

### 4. Run Pipeline

Execute scripts in order:
```python
# Step 1: Download videos
!python download_videos.py

# Step 2: Transcribe audio
!python transcribe.py

# Step 3: Extract frames
!python extract_frames.py

# Step 4: Detect slides
!python detect_slides.py

# Step 5: Run OCR
!python run_ocr.py

# Step 6: Create chunks
!python chunk_videos.py
```

### 5. Download Outputs

Download generated files from Kaggle:
- Go to "Output" section
- Download:
  - `transcripts/*.json`
  - `chunks/*.json`
  - `metadata/processing_log.txt`

---

## Detailed Instructions

### Video Sources

**Stanford Course Playlists:**

1. **CS231n: Convolutional Neural Networks**
   - URL: `https://www.youtube.com/playlist?list=3lgR6Z0mVjE`
   - Videos to process: Lectures 1-5 (initially)

2. **CS224n: Natural Language Processing**
   - URL: `https://www.youtube.com/playlist?list=oMBZnVuxJ346c`
   - Videos to process: Lectures 1-5 (initially)

3. **CS230: Deep Learning**
   - URL: `https://www.youtube.com/playlist?list=8A7g0e7_0d0`
   - Videos to process: Lectures 1-3 (initially)

### Script Descriptions

#### 1. download_videos.py

**Purpose:** Download YouTube videos and extract audio

**Input:**
- YouTube playlist URLs
- Video IDs

**Output:**
- `videos/*.mp4` (video files)
- `audio/*.wav` (audio files)

**Configuration:**
```python
PLAYLISTS = {
    "cs231n": "https://www.youtube.com/playlist?list=3lgR6Z0mVjE",
    "cs224n": "https://www.youtube.com/playlist?list=oMBZnVuxJ346c",
    "cs230": "https://www.youtube.com/playlist?list=8A7g0e7_0d0"
}

VIDEO_IDS = {
    "cs231n": ["NfnWJUyUJYU", "siu3J-Yuv3Q", "c7M1kmKqoEg"],  # Lectures 1-3
    # Add more as needed
}
```

**Runtime:** 10-20 minutes per playlist

---

#### 2. transcribe.py

**Purpose:** Transcribe audio using Whisper with timestamps

**Input:**
- `audio/*.wav` files

**Output:**
- `transcripts/{video_id}.json`

**Format:**
```json
{
  "video_id": "NfnWJUyUJYU",
  "title": "CS231n Lecture 1",
  "transcript": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "Welcome to CS231n..."
    },
    {
      "start": 5.2,
      "end": 10.5,
      "text": "Today we're going to talk about..."
    }
  ],
  "duration": 3600.0
}
```

**Configuration:**
```python
# Whisper model (faster models for testing)
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large

# Language (auto-detect if None)
LANGUAGE = "en"  # English

# Output format
OUTPUT_DIR = "transcripts"
```

**Runtime:** 5-10 minutes per hour of video (on GPU)

---

#### 3. extract_frames.py

**Purpose:** Extract keyframes from videos at regular intervals

**Input:**
- `videos/*.mp4` files

**Output:**
- `frames/{video_id}/frame_{timestamp}.jpg`

**Configuration:**
```python
# Extract every N seconds
EXTRACT_INTERVAL = 5  # seconds

# Output format
OUTPUT_DIR = "frames"

# Frame quality
FRAME_QUALITY = 90  # JPEG quality (1-100)
```

**Runtime:** 5-10 minutes per hour of video

---

#### 4. detect_slides.py

**Purpose:** Detect slide changes in video frames

**Input:**
- `frames/{video_id}/*.jpg`

**Output:**
- `slides/{video_id}_slides.json`

**Format:**
```json
{
  "video_id": "NfnWJUyUJYU",
  "slides": [
    {
      "slide_number": 1,
      "timestamp": 0.0,
      "frame_path": "frames/NfnWJUyUJYU/frame_0.jpg",
      "change_detected": true
    },
    {
      "slide_number": 2,
      "timestamp": 125.5,
      "frame_path": "frames/NfnWJUyUJYU/frame_125.jpg",
      "change_detected": true
    }
  ]
}
```

**Configuration:**
```python
# Slide detection threshold
SIMILARITY_THRESHOLD = 0.85  # Below this = slide changed

# Comparison method
METHOD = "ssim"  # Options: ssim, mse, diff

# Output directory
OUTPUT_DIR = "slides"
```

**Runtime:** 10-15 minutes per hour of video

---

#### 5. run_ocr.py

**Purpose:** Extract text from video frames using OCR

**Input:**
- `frames/{video_id}/*.jpg`

**Output:**
- `ocr/{video_id}_ocr.json`

**Format:**
```json
{
  "video_id": "NfnWJUyUJYU",
  "ocr_results": [
    {
      "timestamp": 0.0,
      "frame_path": "frames/NfnWJUyUJYU/frame_0.jpg",
      "text": "CS231n: Convolutional Neural Networks\nLecture 1: Introduction",
      "confidence": 0.95
    },
    {
      "timestamp": 125.5,
      "frame_path": "frames/NfnWJUyUJYU/frame_125.jpg",
      "text": "Computer Vision History",
      "confidence": 0.87
    }
  ]
}
```

**Configuration:**
```python
# OCR engine
OCR_ENGINE = "tesseract"  # Options: tesseract, paddleocr, easyocr

# Language
LANGUAGE = "eng"  # English

# Minimum confidence
MIN_CONFIDENCE = 0.7

# Output directory
OUTPUT_DIR = "ocr"
```

**Runtime:** 15-20 minutes per hour of video

---

#### 6. chunk_videos.py

**Purpose:** Create smart video chunks using topic boundaries

**Input:**
- `transcripts/{video_id}.json`
- `slides/{video_id}_slides.json`
- `ocr/{video_id}_ocr.json`

**Output:**
- `chunks/{video_id}_chunks.json`

**Format:**
```json
{
  "video_id": "NfnWJUyUJYU",
  "chunks": [
    {
      "chunk_id": 1,
      "start_time": 0.0,
      "end_time": 180.5,
      "topic": "Introduction to Computer Vision",
      "transcript": "Welcome to CS231n...",
      "has_diagram": true,
      "ocr_text": "CS231n: Convolutional Neural Networks",
      "slide_number": 1
    },
    {
      "chunk_id": 2,
      "start_time": 180.5,
      "end_time": 360.0,
      "topic": "History of Computer Vision",
      "transcript": "Let's talk about the history...",
      "has_diagram": true,
      "ocr_text": "Computer Vision History",
      "slide_number": 2
    }
  ]
}
```

**Configuration:**
```python
# Chunk duration limits
MIN_CHUNK_DURATION = 60  # seconds
MAX_CHUNK_DURATION = 300  # seconds
TARGET_CHUNK_DURATION = 180  # seconds (3 minutes)

# Topic segmentation
USE_TOPIC_SEGMENTATION = True
SEGMENTATION_METHOD = "texttiling"  # Options: texttiling, slide_based

# Output directory
OUTPUT_DIR = "chunks"
```

**Runtime:** 5-10 minutes per video

---

## Kaggle-Specific Tips

### GPU Usage

**Check GPU:**
```python
!nvidia-smi
```

**Use GPU for Whisper:**
```python
import torch
import whisper

# Check if GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model with GPU
model = whisper.load_model("base", device=device)
```

### Memory Management

**Clear cache:**
```python
import torch
import gc

# Clear GPU cache
torch.cuda.empty_cache()
gc.collect()
```

**Batch processing:**
```python
# Process videos in batches to avoid OOM
BATCH_SIZE = 3  # Process 3 videos at a time

for i in range(0, len(video_ids), BATCH_SIZE):
    batch = video_ids[i:i+BATCH_SIZE]
    process_batch(batch)
    torch.cuda.empty_cache()
```

### Save Progress

**Checkpointing:**
```python
import json

# Save progress after each video
PROGRESS_FILE = "progress.json"

def save_progress(processed_videos):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({"processed": processed_videos}, f)

def load_progress():
    try:
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)["processed"]
    except FileNotFoundError:
        return []
```

**Resume from checkpoint:**
```python
# Skip already processed videos
processed = load_progress()
remaining = [v for v in video_ids if v not in processed]

for video_id in remaining:
    process_video(video_id)
    processed.append(video_id)
    save_progress(processed)
```

### Data Persistence

**Create Kaggle Dataset:**
1. Go to "Output" section
2. Click "Save Version"
3. Choose "Save & Run All" (to create output)
4. Go to "Your Datasets"
5. Create new dataset from notebook output
6. Use dataset in future sessions

**Download outputs:**
```python
# Zip outputs for easy download
!zip -r transcripts.zip transcripts/
!zip -r chunks.zip chunks/
```

---

## Troubleshooting

### Common Issues

**Issue 1: Out of Memory (OOM)**
```python
# Solution: Reduce batch size
BATCH_SIZE = 1  # Process one at a time

# Or use smaller Whisper model
WHISPER_MODEL = "tiny"  # Instead of "base"
```

**Issue 2: Video Download Fails**
```python
# Solution: Use yt-dlp with retries
import yt_dlp

ydl_opts = {
    'format': 'bestvideo+bestaudio/best',
    'retry_sleep_functions': {
        'http': lambda x: 5,  # Wait 5 seconds on error
    },
    'retries': 10
}
```

**Issue 3: Whisper Slow**
```python
# Solution: Use smaller model
WHISPER_MODEL = "tiny"  # Fastest
# or
WHISPER_MODEL = "base"  # Balance of speed/accuracy

# Enable fp16 for faster inference
model = whisper.load_model("base", device="cuda")
result = model.transcribe(audio, fp16=True)
```

**Issue 4: OCR Fails**
```python
# Solution: Preprocess images
import cv2

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh
```

---

## Expected Outputs

### File Structure After Processing

```
kaggle/working/
├── videos/
│   ├── NfnWJUyUJYU.mp4
│   └── ...
├── audio/
│   ├── NfnWJUyUJYU.wav
│   └── ...
├── transcripts/
│   ├── NfnWJUyUJYU.json
│   └── ...
├── frames/
│   ├── NfnWJUyUJYU/
│   │   ├── frame_0.jpg
│   │   ├── frame_5.jpg
│   │   └── ...
│   └── ...
├── slides/
│   ├── NfnWJUyUJYU_slides.json
│   └── ...
├── ocr/
│   ├── NfnWJUyUJYU_ocr.json
│   └── ...
├── chunks/
│   ├── NfnWJUyUJYU_chunks.json
│   └── ...
├── metadata/
│   ├── processing_log.txt
│   └── statistics.json
└── progress.json
```

### Processing Statistics

**Expected runtime (per hour of video):**
- Download: 5-10 minutes
- Transcription: 5-10 minutes (GPU)
- Frame extraction: 5-10 minutes
- Slide detection: 10-15 minutes
- OCR: 15-20 minutes
- Chunking: 5-10 minutes

**Total: ~1-1.5 hours per hour of video**

**For 10 hours of video: ~10-15 hours total processing time**

---

## Integration with Main System

### Transfer Data to Local

1. **Download from Kaggle:**
   - Zip the outputs: `transcripts.zip`, `chunks.zip`
   - Download to local machine

2. **Extract to project:**
   ```bash
   unzip transcripts.zip -d data/transcripts/
   unzip chunks.zip -d data/chunks/
   ```

3. **Load in main system:**
   ```python
   # src/base_rag/video_loader.py
   import json

   def load_chunks(video_id):
       with open(f"data/chunks/{video_id}_chunks.json") as f:
           return json.load(f)
   ```

### Data Validation

**Validate outputs before using:**
```python
def validate_chunks(chunks_file):
    with open(chunks_file) as f:
        data = json.load(f)

    # Check required fields
    for chunk in data["chunks"]:
        assert "start_time" in chunk
        assert "end_time" in chunk
        assert "transcript" in chunk
        assert chunk["end_time"] > chunk["start_time"]

    print("✓ Chunks validated")
```

---

## Optimization Tips

### Speed Up Processing

1. **Use smaller Whisper model:**
   ```python
   WHISPER_MODEL = "tiny"  # 5x faster than "base"
   ```

2. **Extract fewer frames:**
   ```python
   EXTRACT_INTERVAL = 10  # Instead of 5
   ```

3. **Skip OCR (optional):**
   ```python
   RUN_OCR = False  # If not needed
   ```

4. **Process fewer videos:**
   ```python
   # Start with 1-2 lectures per course
   VIDEO_IDS = {
       "cs231n": ["NfnWJUyUJYU"],  # Just 1 lecture
   }
   ```

### Reduce Memory Usage

1. **Delete intermediate files:**
   ```python
   import os

   # Delete videos after transcription
   os.remove("videos/video.mp4")
   os.remove("audio/video.wav")
   ```

2. **Process in smaller batches:**
   ```python
   BATCH_SIZE = 1  # Process one at a time
   ```

3. **Clear cache regularly:**
   ```python
   torch.cuda.empty_cache()
   gc.collect()
   ```

---

## Next Steps

After video processing is complete:

1. **Transfer data to local system**
2. **Load chunks into main RAG system**
3. **Generate embeddings for chunks**
4. **Store in FAISS index**
5. **Test retrieval with sample queries**

See main project README for integration instructions.

---

## Questions or Issues?

**Common Problems:**
- Check GPU availability: `!nvidia-smi`
- Check disk space: `!df -h`
- Check memory: `!free -h`

**Still stuck?**
- Create GitHub issue in main repo
- Tag with `video-processing` label
- Include error messages and logs

---

**Happy Processing!** 🎥→📝→🧠
