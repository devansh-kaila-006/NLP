"""
Configuration for Multi-Modal RAG Pipeline
Centralized configuration for all components
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===============================
# Paths
# ===============================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
INDICES_DIR = PROCESSED_DIR / "indices"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
VIDEO_DIR = DATA_DIR / "videos"
TRANSCRIPTS_DIR = PROCESSED_DIR / "transcripts"
FRAMES_DIR = PROCESSED_DIR / "frames"
VIDEO_CHUNKS_DIR = PROCESSED_DIR / "video_chunks"

for dir_path in [PDF_DIR, PROCESSED_DIR, CHUNKS_DIR, EMBEDDINGS_DIR,
                 INDICES_DIR, CACHE_DIR, LOGS_DIR, VIDEO_DIR,
                 TRANSCRIPTS_DIR, FRAMES_DIR, VIDEO_CHUNKS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ===============================
# Data Sources Configuration
# ===============================
PDF_SOURCES = {
    "ML_Course_Notes": {
        "type": "pdf",
        "path": str(PDF_DIR / "ML.pdf"),
        "priority": "high",
        "description": "Stanford CS229 Machine Learning notes"
    },
    "DL_Textbook": {
        "type": "pdf",
        "path": str(PDF_DIR / "DL.pdf"),
        "priority": "high",
        "description": "Deep Learning textbook by Ian Goodfellow"
    },
    "Scikit_learn_Docs": {
        "type": "zip",
        "path": str(PDF_DIR / "scikit-learn-docs.zip"),
        "extract_to": str(CACHE_DIR / "sklearn_docs"),
        "priority": "medium",
        "description": "Scikit-learn documentation"
    },
    "PyTorch_Docs": {
        "type": "web",
        "url": "https://docs.pytorch.org/docs/2.11/index.html",
        "cache_dir": str(CACHE_DIR / "pytorch_docs"),
        "priority": "medium",
        "description": "PyTorch 2.11 documentation"
    }
}

# ===============================
# Video Sources Configuration (YouTube Playlists)
# ===============================
VIDEO_SOURCES = {
    "Stanford_ML": {
        "type": "youtube_playlist",
        "url": "https://www.youtube.com/playlist?list=PLoROMvodv4rNyWOpJg_Yh4NSqI4Z4vOYy",
        "course": "CS229 Machine Learning",
        "instructor": "Andrew Ng",
        "priority": "high",
        "description": "Stanford CS229 Machine Learning course by Andrew Ng"
    },
    "Stanford_NLP": {
        "type": "youtube_playlist",
        "url": "https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D",
        "course": "CS224n Natural Language Processing",
        "instructor": "Chris Manning",
        "priority": "high",
        "description": "Stanford CS224n NLP course with Deep Learning"
    },
    "Stanford_CV": {
        "type": "youtube_playlist",
        "url": "https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16",
        "course": "CS231n Computer Vision",
        "instructor": "Fei-Fei Li, Justin Johnson, Serena Yeung",
        "priority": "high",
        "description": "Stanford CS231n Computer Vision course"
    },
    "MIT_DL": {
        "type": "youtube_playlist",
        "url": "https://www.youtube.com/playlist?list=PLUl4u3cNGP63URZnh5iqBzDTDYPUTQT-8",
        "course": "MIT 6.S191 Introduction to Deep Learning",
        "instructor": "MIT",
        "priority": "high",
        "description": "MIT 6.S191: Introduction to Deep Learning"
    },
    "MIT_DL_Alternative": {
        "type": "youtube_playlist",
        "url": "https://www.youtube.com/playlist?list=PLUl4u3cNGP60YyhMjYmXuVmX562QcClSp",
        "course": "MIT Deep Learning",
        "instructor": "MIT",
        "priority": "high",
        "description": "MIT Deep Learning course (6.S191) - Alternative URL"
    }
}

# ===============================
# Text Chunking Configuration
# ===============================
CHUNKING_CONFIG = {
    "chunk_size": 400,              # Target chunk size in tokens
    "chunk_overlap": 50,            # Overlap between chunks
    "min_chunk_size": 100,          # Minimum chunk size
    "max_chunk_size": 800,          # Maximum chunk size
    "semantic_chunking": True,      # Use semantic chunking
    "hierarchical": True,           # Hierarchical chunking (chapter → section → paragraph)
}

# ===============================
# Video Processing Configuration
# ===============================
VIDEO_PROCESSING_CONFIG = {
    # Download settings
    "download_format": "mp4",
    "resolution": "720p",           # Video resolution for download
    "download_subtitles": True,     # Download available subtitles

    # Transcription settings (Whisper)
    "whisper_model": "base",        # tiny, base, small, medium, large
    "whisper_language": "en",       # Auto-detect if None
    "transcription_timestamps": True,  # Include word-level timestamps

    # Frame extraction
    "frame_interval": 5,            # Extract frame every N seconds
    "frame_quality": 85,            # JPEG quality (1-100)
    "slide_detection": True,        # Detect slide changes

    # Video chunking
    "min_chunk_duration": 120,      # Minimum chunk length in seconds (2 min)
    "max_chunk_duration": 300,      # Maximum chunk length in seconds (5 min)
    "semantic_chunking": True,      # Use semantic boundaries (topic shifts, slides)
    "chunk_overlap": 15,            # Overlap between video chunks in seconds
}

# ===============================
# Embedding Configuration
# ===============================
EMBEDDING_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",  # Sentence transformer model
    "batch_size": 32,                    # Batch size for embedding generation
    "normalize": True,                   # Normalize embeddings for cosine similarity
    "device": "cpu",                     # Device: "cpu" or "cuda"
}

# Alternative models (uncomment to use):
# "all-mpnet-base-v2" - Better quality, slower (768 dim)
# "e5-base-v2" - Optimized for retrieval (512 dim)

# ===============================
# Vector Store Configuration (FAISS)
# ===============================
FAISS_CONFIG = {
    "index_type": "IndexFlatIP",    # Inner product for cosine similarity
    "dimension": 384,                # Embedding dimension (all-MiniLM-L6-v2)
    "normalize": True,               # Normalize vectors
}

# ===============================
# Reranking Configuration
# ===============================
RERANKING_CONFIG = {
    "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "top_k": 5,                      # Retrieve top K chunks
    "top_n": 3,                      # Rerank to top N chunks
    "batch_size": 32,                # Batch size for reranking
    "threshold": 0.5,                # Minimum relevance score
}

# ===============================
# Enhanced Reranking Configuration
# ===============================
ENHANCED_RERANKING_CONFIG = {
    "primary_model": "cross-encoder/ms-marco-electra-base",  # Better model for higher precision
    "fallback_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Fallback if primary unavailable
    "top_n": 5,                     # Rerank to top N chunks
    "batch_size": 32,               # Batch size for reranking
    "threshold": 0.3,               # Base minimum relevance score (lowered for dynamic thresholding)

    # Advanced features
    "enable_query_expansion": True,      # Expand queries with related terms
    "enable_diversity_reranking": True,  # Use MMR for diverse results
    "enable_multi_stage": True,          # Use coarse-to-fine reranking
    "enable_ensemble": False,            # Future: ensemble multiple rerankers
}

# ===============================
# Optimized Reranking Configuration (Performance-Optimized)
# ===============================
OPTIMIZED_RERANKING_CONFIG = {
    # Fast model for simple queries
    "primary_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    # Enhanced model for complex queries only
    "enhanced_model": "cross-encoder/ms-marco-electra-base",

    "top_n": 5,
    "batch_size": 32,
    "threshold": 0.5,

    # Performance optimization features
    "enable_query_complexity_detection": True,  # Detect complex vs simple queries
    "enable_caching": True,                      # Cache reranking results
    "enable_parallel_processing": False,         # Future: parallel processing

    # Selective feature usage for performance
    "enable_query_expansion": False,     # Disabled for performance
    "enable_diversity_reranking": False,  # Disabled for performance
    "enable_multi_stage": True,          # Enabled but optimized

    # Complexity detection parameters
    "complex_query_min_length": 50,      # Queries longer than this are complex
    "complex_query_keywords": [
        'explain', 'describe', 'analyze', 'compare',
        'difference', 'how does', 'what is the relationship'
    ]
}

# ===============================
# LLM Configuration (Gemini)
# ===============================
# Available models: gemini-2.0-flash, gemini-2.5-flash, gemini-3.1-flash-lite-preview
# Using gemini-3.1-flash-lite-preview for latest capabilities
LLM_CONFIG = {
    "model": os.getenv("GEMINI_MODEL", "models/gemini-3.1-flash-lite-preview"),
    "api_key": os.getenv("GOOGLE_API_KEY"),
    "temperature": 0.3,              # Lower for more factual answers
    "top_p": 0.95,                   # Match Deno config
    "top_k": 40,                     # Match Deno config
    "max_output_tokens": 1024,
    "timeout": 30,                   # Request timeout in seconds
}

# ===============================
# Retrieval Configuration
# ===============================
RETRIEVAL_CONFIG = {
    "top_k": 5,                      # Default number of chunks to retrieve
    "min_relevance_score": 0.3,      # Minimum relevance threshold
    "use_reranker": True,            # Enable/disable reranking
}

# ===============================
# Logging Configuration
# ===============================
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d]: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "rag_pipeline.log"),
            "mode": "a"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
}

# ===============================
# Performance Configuration
# ===============================
PERFORMANCE_CONFIG = {
    "cache_embeddings": True,        # Cache generated embeddings
    "batch_retrieval": True,         # Enable batch retrieval
    "max_concurrent_requests": 5,    # Max concurrent API requests
}

# ===============================
# Metadata Configuration
# ===============================
METADATA_SCHEMA = {
    # Required metadata fields
    "required_fields": [
        "text",
        "source_name",
        "source_type",
        "chunk_id"
    ],
    # Optional metadata fields
    "optional_fields": {
        "pdf": ["chapter", "section", "page_start", "page_end"],
        "html": ["module", "class", "function", "url", "section"],
        "zip": ["file_name", "section", "subsection"],
        "web": ["url", "title", "section"],
        "video": [
            "video_id",            # YouTube video ID
            "video_title",         # Video title
            "video_url",           # Full video URL with timestamp
            "playlist_name",       # Playlist/course name
            "lecture_number",      # Lecture number in playlist
            "timestamp_start",     # Chunk start time (seconds)
            "timestamp_end",       # Chunk end time (seconds)
            "duration",            # Chunk duration (seconds)
            "instructor",          # Course instructor
            "slide_number",        # Slide number (if detected)
            "frame_path"           # Path to representative frame
        ]
    }
}

# ===============================
# Validation Functions
# ===============================

def validate_config():
    """Validate configuration settings"""
    errors = []

    # Check API key
    if not LLM_CONFIG["api_key"] or LLM_CONFIG["api_key"] == "your-api-key-here":
        errors.append("GOOGLE_API_KEY not set in .env file")

    # Check PDF/ZIP data sources exist
    for source_name, source_config in PDF_SOURCES.items():
        if source_config["type"] in ["pdf", "zip"]:
            path = Path(source_config["path"])
            if not path.exists():
                errors.append(f"Data source not found: {source_name} at {path}")

    # Note: Video sources are validated during download (not checked here)
    # Video processing is optional and not yet implemented

    return errors


def get_embedding_dimension():
    """Get embedding dimension based on model"""
    model = EMBEDDING_CONFIG["model_name"]
    dimensions = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "e5-base-v2": 512
    }
    return dimensions.get(model, 384)


# Update FAISS dimension based on embedding model
FAISS_CONFIG["dimension"] = get_embedding_dimension()


if __name__ == "__main__":
    # Test configuration
    errors = validate_config()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Configuration valid")
        print(f"📁 Data directory: {DATA_DIR}")
        print(f"📊 Embedding model: {EMBEDDING_CONFIG['model_name']}")
        print(f"📏 Embedding dimension: {FAISS_CONFIG['dimension']}")
        print(f"🤖 LLM model: {LLM_CONFIG['model']}")
