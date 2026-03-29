"""
Video Processing Script 5/6: OCR on Video Frames

This script extracts text from video frames using OCR (Optical Character Recognition).
This helps capture text from slides and diagrams.

Usage:
    python run_ocr.py

Input:
    frames/{video_id}/frame_*.jpg (frames from extract_frames.py)

Output:
    ocr/{video_id}_ocr.json (extracted text with timestamps)
"""

import os
import cv2
import json
import pytesseract
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# OCR engine
OCR_ENGINE = "tesseract"  # Options: tesseract, paddleocr

# Language
LANGUAGE = "eng"  # English

# Minimum confidence score (0-1)
MIN_CONFIDENCE = 0.7

# Directories
FRAMES_DIR = Path("frames")
OCR_DIR = Path("ocr")
METADATA_DIR = Path("metadata")

# Tesseract config
TESSERACT_CONFIG = r'--oem 3 --psm 6'  # Page segmentation mode

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def create_directories():
    """Create output directories."""
    OCR_DIR.mkdir(exist_ok=True)
    METADATA_DIR.mkdir(exist_ok=True)
    print("✓ Directories created")

def preprocess_image(image_path):
    """Preprocess image for better OCR results."""
    img = cv2.imread(str(image_path))

    if img is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(binary)

    return denoised

def extract_text(frame_path, timestamp):
    """Extract text from frame using OCR."""
    try:
        # Preprocess image
        processed = preprocess_image(frame_path)

        if processed is None:
            return {
                'timestamp': timestamp,
                'frame_path': str(frame_path),
                'text': "",
                'confidence': 0.0,
                'error': 'Failed to load image'
            }

        # Perform OCR
        data = pytesseract.image_to_data(
            processed,
            config=TESSERACT_CONFIG,
            lang=LANGUAGE,
            output_type=pytesseract.Output.DICT
        )

        # Extract text and confidence
        text_parts = []
        confidences = []

        for i, text in enumerate(data['text']):
            confidence = int(data['conf'][i])

            # Filter low confidence results
            if confidence > 0 and text.strip():
                if confidence >= MIN_CONFIDENCE * 100:
                    text_parts.append(text.strip())
                    confidences.append(confidence)

        # Combine text
        full_text = ' '.join(text_parts)

        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        avg_confidence = avg_confidence / 100.0  # Convert to 0-1 scale

        return {
            'timestamp': timestamp,
            'frame_path': str(frame_path),
            'text': full_text,
            'confidence': avg_confidence,
            'word_count': len(text_parts)
        }

    except Exception as e:
        return {
            'timestamp': timestamp,
            'frame_path': str(frame_path),
            'text': "",
            'confidence': 0.0,
            'error': str(e)
        }

def process_video_ocr(video_id):
    """Process OCR for all frames of a video."""
    print(f"\n{'='*60}")
    print(f"OCR Processing: {video_id}")
    print(f"{'='*60}")

    # Get frames directory
    frames_dir = FRAMES_DIR / video_id

    if not frames_dir.exists():
        print(f"✗ Frames directory not found: {frames_dir}")
        return None

    # Get frame files
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))

    if not frame_files:
        print(f"⚠ No frames found for {video_id}")
        return None

    print(f"Found {len(frame_files)} frames")
    print(f"OCR engine: {OCR_ENGINE}")
    print(f"Language: {LANGUAGE}")
    print(f"Min confidence: {MIN_CONFIDENCE}")

    # Check if already processed
    metadata_file = METADATA_DIR / f"{video_id}_ocr.json"
    if metadata_file.exists():
        print(f"✓ OCR already processed for {video_id}")
        with open(metadata_file) as f:
            return json.load(f)

    # Process each frame
    ocr_results = []
    total_text = 0
    high_confidence_count = 0

    for i, frame_path in enumerate(frame_files):
        # Get timestamp from filename
        timestamp = float(frame_path.stem.replace("frame_", ""))

        # Extract text
        result = extract_text(frame_path, timestamp)

        # Only keep results with text
        if result['text'] and result['confidence'] > 0:
            ocr_results.append(result)
            total_text += len(result['text'].split())

            if result['confidence'] >= MIN_CONFIDENCE:
                high_confidence_count += 1

        # Progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(frame_files)} frames")

    # Create metadata
    metadata = {
        'video_id': video_id,
        'engine': OCR_ENGINE,
        'language': LANGUAGE,
        'min_confidence': MIN_CONFIDENCE,
        'total_frames': len(frame_files),
        'frames_with_text': len(ocr_results),
        'total_words': total_text,
        'high_confidence_frames': high_confidence_count,
        'results': ocr_results
    }

    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Processed {len(frame_files)} frames")
    print(f"  Frames with text: {len(ocr_results)}")
    print(f"  Total words extracted: {total_text}")
    print(f"  High confidence frames: {high_confidence_count}")
    print(f"✓ Metadata saved: {metadata_file}")

    return metadata

def process_all_videos():
    """Process OCR for all videos."""
    # Get all frame directories
    frame_dirs = [d for d in FRAMES_DIR.iterdir() if d.is_dir()]

    if not frame_dirs:
        print(f"⚠ No frame directories found in {FRAMES_DIR}")
        print("Run extract_frames.py first")
        return []

    print(f"\nFound {len(frame_dirs)} frame directories")

    # Process each video
    all_metadata = []
    success_count = 0
    failed_count = 0

    for frame_dir in sorted(frame_dirs):
        video_id = frame_dir.name

        metadata = process_video_ocr(video_id)

        if metadata:
            all_metadata.append(metadata)
            success_count += 1
        else:
            failed_count += 1

    # Summary
    print("\n" + "="*60)
    print("OCR PROCESSING SUMMARY")
    print("="*60)
    print(f"Total videos: {len(frame_dirs)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {failed_count}")

    total_words = sum(m['total_words'] for m in all_metadata)
    print(f"Total words extracted: {total_words}")

    # Save processing log
    log_file = METADATA_DIR / "ocr_log.json"
    with open(log_file, 'w') as f:
        json.dump({
            'total': len(frame_dirs),
            'success': success_count,
            'failed': failed_count,
            'total_words': total_words,
            'engine': OCR_ENGINE,
            'language': LANGUAGE
        }, f, indent=2)

    print(f"\n✓ Log saved: {log_file}")

    return all_metadata

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    print("="*60)
    print("OCR on Video Frames")
    print("="*60)

    # Create directories
    create_directories()

    # Process all videos
    process_all_videos()

    print("\n✓ OCR processing complete!")

if __name__ == "__main__":
    main()
