"""
Video Processing Script 4/6: Detect Slide Changes in Frames

This script detects slide changes in video frames by comparing consecutive frames.
This helps identify topic boundaries in lectures.

Usage:
    python detect_slides.py

Input:
    frames/{video_id}/frame_*.jpg (frames from extract_frames.py)

Output:
    slides/{video_id}_slides.json (slide change timestamps)
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

# ============================================================================
# CONFIGURATION
# ============================================================================

# Slide detection threshold (0-1)
# Lower = more sensitive, Higher = less sensitive
SIMILARITY_THRESHOLD = 0.85

# Comparison method: "ssim" (recommended) or "mse"
METHOD = "ssim"

# Directories
FRAMES_DIR = Path("frames")
SLIDES_DIR = Path("slides")
METADATA_DIR = Path("metadata")

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def create_directories():
    """Create output directories."""
    SLIDES_DIR.mkdir(exist_ok=True)
    METADATA_DIR.mkdir(exist_ok=True)
    print("✓ Directories created")

def load_frame(frame_path):
    """Load and preprocess frame for comparison."""
    img = cv2.imread(str(frame_path))

    if img is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray

def compute_similarity(frame1_path, frame2_path, method="ssim"):
    """Compute similarity between two frames."""
    # Load frames
    img1 = load_frame(frame1_path)
    img2 = load_frame(frame2_path)

    if img1 is None or img2 is None:
        return 0.0

    # Resize to match (in case dimensions differ)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Compute similarity
    if method == "ssim":
        # Structural Similarity Index (0-1, higher = more similar)
        similarity, _ = ssim(img1, img2, full=True)
    elif method == "mse":
        # Mean Squared Error (lower = more similar, convert to 0-1)
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        max_mse = 255 ** 2
        similarity = 1 - (mse / max_mse)
    else:
        raise ValueError(f"Unknown method: {method}")

    return similarity

def detect_slides(video_id, frames_metadata):
    """Detect slide changes in video frames."""
    print(f"\n{'='*60}")
    print(f"Detecting slides: {video_id}")
    print(f"{'='*60}")

    # Get frames directory
    frames_dir = FRAMES_DIR / video_id

    if not frames_dir.exists():
        print(f"✗ Frames directory not found: {frames_dir}")
        return None

    # Get frame files
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))

    if len(frame_files) < 2:
        print(f"⚠ Need at least 2 frames, found {len(frame_files)}")
        return None

    print(f"Found {len(frame_files)} frames")
    print(f"Method: {METHOD}")
    print(f"Threshold: {SIMILARITY_THRESHOLD}")

    # Check if already processed
    metadata_file = METADATA_DIR / f"{video_id}_slides.json"
    if metadata_file.exists():
        print(f"✓ Slides already detected for {video_id}")
        with open(metadata_file) as f:
            return json.load(f)

    # Detect slides
    slides = []
    current_slide = {
        'slide_number': 1,
        'timestamp': 0.0,
        'frame_path': str(frame_files[0]),
        'change_detected': True  # First frame is always a slide
    }
    slides.append(current_slide)

    slide_count = 1

    for i in range(1, len(frame_files)):
        frame1_path = frame_files[i-1]
        frame2_path = frame_files[i]

        # Compute similarity
        similarity = compute_similarity(frame1_path, frame2_path, METHOD)

        # Get timestamp from filename
        timestamp = float(frame2_path.stem.replace("frame_", ""))

        # Check if slide changed
        if similarity < SIMILARITY_THRESHOLD:
            slide_count += 1
            new_slide = {
                'slide_number': slide_count,
                'timestamp': timestamp,
                'frame_path': str(frame2_path),
                'change_detected': True,
                'similarity_with_previous': similarity
            }
            slides.append(new_slide)
            print(f"  Slide {slide_count} at {timestamp:.1f}s (similarity: {similarity:.3f})")
        else:
            # No change, but still track for debugging
            if len(slides) > 0:
                slides[-1]['similarity_with_previous'] = similarity

    # Create metadata
    metadata = {
        'video_id': video_id,
        'method': METHOD,
        'threshold': SIMILARITY_THRESHOLD,
        'total_slides': slide_count,
        'slides': slides
    }

    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Detected {slide_count} slides")
    print(f"✓ Metadata saved: {metadata_file}")

    return metadata

def process_all_videos():
    """Detect slides for all videos."""
    # Get all frame metadata files
    metadata_files = list(METADATA_DIR.glob("*_frames.json"))

    if not metadata_files:
        print(f"⚠ No frame metadata found in {METADATA_DIR}")
        print("Run extract_frames.py first")
        return []

    print(f"\nFound {len(metadata_files)} frame metadata files")

    # Process each video
    all_metadata = []
    success_count = 0
    failed_count = 0

    for metadata_file in sorted(metadata_files):
        # Load frame metadata
        with open(metadata_file) as f:
            frames_metadata = json.load(f)

        video_id = frames_metadata['video_id']

        # Detect slides
        slide_metadata = detect_slides(video_id, frames_metadata)

        if slide_metadata:
            all_metadata.append(slide_metadata)
            success_count += 1
        else:
            failed_count += 1

    # Summary
    print("\n" + "="*60)
    print("SLIDE DETECTION SUMMARY")
    print("="*60)
    print(f"Total videos: {len(metadata_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {failed_count}")

    total_slides = sum(m['total_slides'] for m in all_metadata)
    print(f"Total slides detected: {total_slides}")

    # Save processing log
    log_file = METADATA_DIR / "slide_detection_log.json"
    with open(log_file, 'w') as f:
        json.dump({
            'total': len(metadata_files),
            'success': success_count,
            'failed': failed_count,
            'total_slides': total_slides,
            'method': METHOD,
            'threshold': SIMILARITY_THRESHOLD
        }, f, indent=2)

    print(f"\n✓ Log saved: {log_file}")

    return all_metadata

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    print("="*60)
    print("Slide Detection in Video Frames")
    print("="*60)

    # Create directories
    create_directories()

    # Process all videos
    process_all_videos()

    print("\n✓ Slide detection complete!")

if __name__ == "__main__":
    main()
