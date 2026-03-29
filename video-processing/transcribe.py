"""
Video Processing Script 2/6: Transcribe Audio using Whisper

This script transcribes audio files using OpenAI's Whisper model.
Requires GPU for faster processing.

Usage:
    python transcribe.py

Input:
    audio/*.wav (audio files from download_videos.py)

Output:
    transcripts/{video_id}.json (timestamped transcripts)
"""

import os
import json
import torch
import whisper
import logging
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Whisper model options: tiny, base, small, medium, large
# "tiny" is fastest, "large" is most accurate
WHISPER_MODEL = "base"  # Recommended for balance of speed/accuracy

# Language (None = auto-detect)
LANGUAGE = "en"  # English

# Device (cuda for GPU, cpu for CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directories
AUDIO_DIR = Path("audio")
TRANSCRIPT_DIR = Path("transcripts")
METADATA_DIR = Path("metadata")

# Transcription options
TEMPERATURE = 0.0  # Sampling temperature (0.0 = deterministic)
BEAM_SIZE = 5      # Beam size for decoding
BEST_OF = 5        # Number of independent samples

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def create_directories():
    """Create output directories."""
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    METADATA_DIR.mkdir(exist_ok=True)
    print("✓ Directories created")

def check_gpu():
    """Check GPU availability and info."""
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ No GPU available, using CPU (slower)")

def load_model():
    """Load Whisper model."""
    print(f"\nLoading Whisper model: {WHISPER_MODEL}")
    print(f"Device: {DEVICE}")

    try:
        model = whisper.load_model(WHISPER_MODEL, device=DEVICE)
        print(f"✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

def transcribe_audio(model, audio_path, video_id):
    """Transcribe audio file using Whisper."""
    print(f"\n{'='*60}")
    print(f"Transcribing: {video_id}")
    print(f"{'='*60}")

    transcript_path = TRANSCRIPT_DIR / f"{video_id}.json"

    # Check if already transcribed
    if transcript_path.exists():
        print(f"✓ Transcript already exists: {transcript_path}")
        with open(transcript_path) as f:
            return json.load(f)

    # Check if audio file exists
    if not audio_path.exists():
        print(f"✗ Audio file not found: {audio_path}")
        return None

    print(f"Audio file: {audio_path}")
    print(f"Model: {WHISPER_MODEL}")
    print(f"Device: {DEVICE}")
    print("Transcribing... (this may take a few minutes)")

    try:
        # Transcribe with timestamps
        result = model.transcribe(
            str(audio_path),
            language=LANGUAGE,
            temperature=TEMPERATURE,
            beam_size=BEAM_SIZE,
            best_of=BEST_OF,
            fp16=(DEVICE == "cuda")  # Use fp16 on GPU
        )

        # Extract transcript segments
        segments = []
        for segment in result['segments']:
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })

        # Create output structure
        transcript = {
            'video_id': video_id,
            'language': result['language'],
            'duration': result['duration'],
            'text': result['text'],
            'segments': segments
        }

        # Save to file
        with open(transcript_path, 'w') as f:
            json.dump(transcript, f, indent=2)

        print(f"✓ Transcription complete: {transcript_path}")
        print(f"  Duration: {result['duration']:.1f} seconds")
        print(f"  Segments: {len(segments)}")
        print(f"  Characters: {len(result['text'])}")

        return transcript

    except Exception as e:
        print(f"✗ Error transcribing {video_id}: {e}")
        logger.error(f"Transcription error for {video_id}: {e}")
        return None

def process_all_audios(model):
    """Process all audio files in the audio directory."""
    # Get all audio files
    audio_files = list(AUDIO_DIR.glob("*.wav"))

    if not audio_files:
        print(f"⚠ No audio files found in {AUDIO_DIR}")
        return []

    print(f"\nFound {len(audio_files)} audio files")

    # Process each audio file
    transcripts = []
    success_count = 0
    failed_count = 0

    for audio_path in sorted(audio_files):
        video_id = audio_path.stem  # filename without extension

        transcript = transcribe_audio(model, audio_path, video_id)

        if transcript:
            transcripts.append(transcript)
            success_count += 1
        else:
            failed_count += 1

    # Summary
    print("\n" + "="*60)
    print("TRANSCRIPTION SUMMARY")
    print("="*60)
    print(f"Total audio files: {len(audio_files)}")
    print(f"Successfully transcribed: {success_count}")
    print(f"Failed: {failed_count}")

    # Save processing log
    log_file = METADATA_DIR / "transcription_log.json"
    with open(log_file, 'w') as f:
        json.dump({
            'total': len(audio_files),
            'success': success_count,
            'failed': failed_count,
            'model': WHISPER_MODEL,
            'device': DEVICE
        }, f, indent=2)

    print(f"\n✓ Log saved: {log_file}")

    return transcripts

def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ GPU cache cleared")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    print("="*60)
    print("Whisper Audio Transcription")
    print("="*60)

    # Create directories
    create_directories()

    # Check GPU
    check_gpu()

    # Clear GPU cache
    clear_gpu_cache()

    # Load model
    model = load_model()
    if not model:
        print("✗ Failed to load model. Exiting.")
        return

    # Process all audio files
    transcripts = process_all_audios(model)

    # Clear GPU cache
    clear_gpu_cache()

    print("\n✓ Transcription complete!")

if __name__ == "__main__":
    main()
