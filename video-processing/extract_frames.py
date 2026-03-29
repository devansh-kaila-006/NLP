"""
Video Processing Script 3/6: Extract Keyframes from Videos

This script extracts keyframes from videos at regular intervals.
These frames will be used for slide detection and OCR.

Usage:
    python extract_frames.py

Input:
    videos/*.mp4 (video files from download_videos.py)

Output:
    frames/{video_id}/frame_{timestamp}.jpg
"""

import os
import cv2
import json
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Extract frames every N seconds
EXTRACT_INTERVAL = 5  # seconds

# Frame quality (JPEG)
FRAME_QUALITY = 90

# Directories
VIDEO_DIR = Path("videos")
FRAMES_DIR = Path("frames")
METADATA_DIR = Path("metadata")

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def create_directories():
    """Create output directories."""
    FRAMES_DIR.mkdir(exist_ok=True)
    METADATA_DIR.mkdir(exist_ok=True)
    print("✓ Directories created")

def extract_frames(video_id, video_path):
    """Extract frames from video at regular intervals."""
    print(f"\n{'='*60}")
    print(f"Extracting frames: {video_id}")
    print(f"{'='*60}")

    # Create video-specific output directory
    output_dir = FRAMES_DIR / video_id
    output_dir.mkdir(exist_ok=True)

    # Check if already processed
    metadata_file = METADATA_DIR / f"{video_id}_frames.json"
    if metadata_file.exists():
        print(f"✓ Frames already extracted for {video_id}")
        with open(metadata_file) as f:
            return json.load(f)

    # Open video file
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"✗ Error opening video: {video_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Extract interval: {EXTRACT_INTERVAL} seconds")

    # Calculate frames to extract
    frame_interval = int(fps * EXTRACT_INTERVAL)
    frames_to_extract = list(range(0, total_frames, frame_interval))

    print(f"Frames to extract: {len(frames_to_extract)}")

    # Extract frames
    extracted_frames = []
    frame_count = 0

    for frame_number in frames_to_extract:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read frame
        ret, frame = cap.read()

        if not ret:
            print(f"✗ Error reading frame {frame_number}")
            continue

        # Calculate timestamp
        timestamp = frame_number / fps

        # Save frame
        frame_filename = output_dir / f"frame_{int(timestamp)}.jpg"
        cv2.imwrite(str(frame_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_QUALITY])

        extracted_frames.append({
            'frame_number': frame_number,
            'timestamp': timestamp,
            'filename': f"frame_{int(timestamp)}.jpg",
            'path': str(frame_filename)
        })

        frame_count += 1

        # Progress
        if frame_count % 10 == 0:
            print(f"Extracted {frame_count}/{len(frames_to_extract)} frames")

    # Release video
    cap.release()

    # Save metadata
    metadata = {
        'video_id': video_id,
        'fps': fps,
        'total_frames': total_frames,
        'duration': duration,
        'extract_interval': EXTRACT_INTERVAL,
        'frames_extracted': frame_count,
        'frames': extracted_frames
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Extracted {frame_count} frames")
    print(f"✓ Saved to: {output_dir}")
    print(f"✓ Metadata saved: {metadata_file}")

    return metadata

def process_all_videos():
    """Extract frames from all videos."""
    # Get all video files
    video_files = list(VIDEO_DIR.glob("*.mp4"))

    if not video_files:
        print(f"⚠ No video files found in {VIDEO_DIR}")
        return []

    print(f"\nFound {len(video_files)} video files")

    # Process each video
    all_metadata = []
    success_count = 0
    failed_count = 0

    for video_path in sorted(video_files):
        video_id = video_path.stem

        metadata = extract_frames(video_id, video_path)

        if metadata:
            all_metadata.append(metadata)
            success_count += 1
        else:
            failed_count += 1

    # Summary
    print("\n" + "="*60)
    print("FRAME EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total videos: {len(video_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {failed_count}")

    total_frames = sum(m['frames_extracted'] for m in all_metadata)
    print(f"Total frames extracted: {total_frames}")

    # Save processing log
    log_file = METADATA_DIR / "extraction_log.json"
    with open(log_file, 'w') as f:
        json.dump({
            'total': len(video_files),
            'success': success_count,
            'failed': failed_count,
            'total_frames': total_frames,
            'extract_interval': EXTRACT_INTERVAL
        }, f, indent=2)

    print(f"\n✓ Log saved: {log_file}")

    return all_metadata

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    print("="*60)
    print("Frame Extraction from Videos")
    print("="*60)

    # Create directories
    create_directories()

    # Process all videos
    process_all_videos()

    print("\n✓ Frame extraction complete!")

if __name__ == "__main__":
    main()
