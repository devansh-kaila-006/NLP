"""
Video Processing Script 1/6: Download Videos from YouTube

This script downloads YouTube videos and extracts audio for transcription.
Run this first in the Kaggle pipeline.

Usage:
    python download_videos.py

Output:
    videos/*.mp4 (video files)
    audio/*.wav (audio files)
"""

import os
import yt_dlp
import json
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Stanford course playlists
PLAYLISTS = {
    "cs231n": "https://www.youtube.com/playlist?list=3lgR6Z0mVjE",
    "cs224n": "https://www.youtube.com/playlist?list=oMBZnVuxJ346c",
    "cs230": "https://www.youtube.com/playlist?list=8A7g0e7_0d0"
}

# Specific videos to download (start with 1-3 per course)
VIDEO_IDS = {
    "cs231n": ["NfnWJUyUJYU", "siu3J-Yuv3Q", "c7M1kmKqoEg"],  # Lectures 1-3
    "cs224n": ["jkm2pYfwxqs", "RzqCxZhGt1w", "MjLWGvDfq4g"],  # Lectures 1-3
    "cs230": ["H_IkZyTaM8k", "JbqEjLwPqWM", "UvJRHIz3M8c"]   # Lectures 1-3
}

# Output directories
VIDEO_DIR = Path("videos")
AUDIO_DIR = Path("audio")
METADATA_DIR = Path("metadata")

# Download options
YDL_OPTS = {
    'format': 'bestvideo+bestaudio/best',
    'outtmpl': str(VIDEO_DIR / '%(id)s.%(ext)s'),
    'quiet': False,
    'no_warnings': False,
    'extract_flat': False,
}

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def create_directories():
    """Create output directories if they don't exist."""
    VIDEO_DIR.mkdir(exist_ok=True)
    AUDIO_DIR.mkdir(exist_ok=True)
    METADATA_DIR.mkdir(exist_ok=True)
    print("✓ Directories created")

def get_video_info(video_id):
    """Get video metadata from YouTube."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return {
                'id': info['id'],
                'title': info['title'],
                'duration': info['duration'],
                'channel': info['channel'],
                'upload_date': info['upload_date'],
                'url': url
            }
        except Exception as e:
            print(f"✗ Error getting info for {video_id}: {e}")
            return None

def download_video(video_id):
    """Download video from YouTube."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    print(f"\n{'='*60}")
    print(f"Downloading: {video_id}")
    print(f"{'='*60}")

    # Get video info first
    info = get_video_info(video_id)
    if not info:
        return False

    print(f"Title: {info['title']}")
    print(f"Duration: {info['duration']} seconds ({info['duration']/60:.1f} minutes)")

    # Check if already downloaded
    video_path = VIDEO_DIR / f"{video_id}.mp4"
    if video_path.exists():
        print(f"✓ Video already exists: {video_path}")
        return True

    # Download video
    try:
        with yt_dlp.YoutubeDL(YDL_OPTS) as ydl:
            ydl.download([url])

        print(f"✓ Downloaded: {video_path}")
        return True

    except Exception as e:
        print(f"✗ Error downloading {video_id}: {e}")
        return False

def extract_audio(video_id):
    """Extract audio from video file."""
    video_path = VIDEO_DIR / f"{video_id}.mp4"
    audio_path = AUDIO_DIR / f"{video_id}.wav"

    if not video_path.exists():
        print(f"✗ Video not found: {video_path}")
        return False

    if audio_path.exists():
        print(f"✓ Audio already exists: {audio_path}")
        return True

    print(f"Extracting audio from {video_id}...")

    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(AUDIO_DIR / '%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(str(video_path), download=True)

        print(f"✓ Extracted audio: {audio_path}")
        return True

    except Exception as e:
        print(f"✗ Error extracting audio from {video_id}: {e}")
        return False

def save_metadata(video_id, info):
    """Save video metadata to JSON file."""
    metadata_path = METADATA_DIR / f"{video_id}_metadata.json"

    with open(metadata_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"✓ Saved metadata: {metadata_path}")

def process_video(video_id):
    """Process a single video: download and extract audio."""
    print(f"\n{'#'*60}")
    print(f"# Processing Video: {video_id}")
    print(f"{'#'*60}")

    # Get video info
    info = get_video_info(video_id)
    if not info:
        return False

    # Download video
    if not download_video(video_id):
        return False

    # Extract audio
    if not extract_audio(video_id):
        return False

    # Save metadata
    save_metadata(video_id, info)

    print(f"\n✓ Successfully processed: {video_id}")
    return True

def main():
    """Main execution function."""
    print("="*60)
    print("YouTube Video Downloader")
    print("="*60)

    # Create directories
    create_directories()

    # Collect all video IDs
    all_videos = []
    for course, videos in VIDEO_IDS.items():
        print(f"\nCourse: {course}")
        for video_id in videos:
            all_videos.append((course, video_id))

    print(f"\nTotal videos to download: {len(all_videos)}")

    # Process each video
    success_count = 0
    failed_videos = []

    for course, video_id in all_videos:
        if process_video(video_id):
            success_count += 1
        else:
            failed_videos.append((course, video_id))

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Total videos: {len(all_videos)}")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed: {len(failed_videos)}")

    if failed_videos:
        print("\nFailed videos:")
        for course, video_id in failed_videos:
            print(f"  - {course}: {video_id}")

    # Save processing log
    log_file = METADATA_DIR / "download_log.json"
    with open(log_file, 'w') as f:
        json.dump({
            'total': len(all_videos),
            'success': success_count,
            'failed': len(failed_videos),
            'failed_videos': failed_videos
        }, f, indent=2)

    print(f"\n✓ Log saved: {log_file}")

if __name__ == "__main__":
    main()
