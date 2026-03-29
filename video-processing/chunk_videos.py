"""
Video Processing Script 6/6: Create Smart Video Chunks

This script creates smart video chunks using topic boundary detection from transcripts
and slide changes. This is Contribution 1: Timestamp-Aware Video RAG.

Usage:
    python chunk_videos.py

Input:
    transcripts/{video_id}.json (from transcribe.py)
    slides/{video_id}_slides.json (from detect_slides.py)
    ocr/{video_id}_ocr.json (from run_ocr.py)

Output:
    chunks/{video_id}_chunks.json (smart video chunks with topics)
"""

import os
import json
from pathlib import Path
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

# Chunk duration limits (in seconds)
MIN_CHUNK_DURATION = 60    # 1 minute minimum
MAX_CHUNK_DURATION = 300   # 5 minutes maximum
TARGET_CHUNK_DURATION = 180  # 3 minutes target

# Topic segmentation method
USE_TOPIC_SEGMENTATION = True
SEGMENTATION_METHOD = "combined"  # Options: "texttiling", "slide_based", "combined"

# Directories
TRANSCRIPT_DIR = Path("transcripts")
SLIDES_DIR = Path("slides")  # Actually in metadata/
OCR_DIR = Path("ocr")  # Actually in metadata/
METADATA_DIR = Path("metadata")
CHUNKS_DIR = Path("chunks")

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def create_directories():
    """Create output directories."""
    CHUNKS_DIR.mkdir(exist_ok=True)
    METADATA_DIR.mkdir(exist_ok=True)
    print("✓ Directories created")

def load_transcript(video_id):
    """Load transcript for video."""
    transcript_file = TRANSCRIPT_DIR / f"{video_id}.json"

    if not transcript_file.exists():
        print(f"✗ Transcript not found: {transcript_file}")
        return None

    with open(transcript_file) as f:
        return json.load(f)

def load_slides(video_id):
    """Load slide detection data for video."""
    slides_file = METADATA_DIR / f"{video_id}_slides.json"

    if not slides_file.exists():
        print(f"⚠ Slides metadata not found: {slides_file}")
        return None

    with open(slides_file) as f:
        return json.load(f)

def load_ocr(video_id):
    """Load OCR data for video."""
    ocr_file = METADATA_DIR / f"{video_id}_ocr.json"

    if not ocr_file.exists():
        print(f"⚠ OCR metadata not found: {ocr_file}")
        return None

    with open(ocr_file) as f:
        return json.load(f)

def detect_topic_boundaries(transcript_segments):
    """Detect topic boundaries in transcript segments."""
    if not USE_TOPIC_SEGMENTATION:
        return []

    print("  Detecting topic boundaries...")

    # Simple algorithm: look for semantic shifts
    # In production, use TextTiling or similar
    boundaries = []

    for i in range(1, len(transcript_segments)):
        seg1 = transcript_segments[i-1]
        seg2 = transcript_segments[i]

        # Check if there's a large gap in time
        time_gap = seg2['start'] - seg1['end']

        if time_gap > 10:  # More than 10 seconds gap
            boundaries.append({
                'timestamp': seg2['start'],
                'type': 'time_gap',
                'gap': time_gap
            })

        # Check for topic keywords (simplified)
        topic_indicators = ["now", "next", "let's talk about", "moving on"]
        text = seg2['text'].lower()

        for indicator in topic_indicators:
            if indicator in text:
                boundaries.append({
                    'timestamp': seg2['start'],
                    'type': 'topic_keyword',
                    'keyword': indicator
                })
                break

    return boundaries

def create_smart_chunks(video_id, transcript, slides_data, ocr_data):
    """Create smart video chunks using multiple signals."""
    print(f"\n{'='*60}")
    print(f"Creating chunks: {video_id}")
    print(f"{'='*60}")

    transcript_segments = transcript['segments']
    total_duration = transcript['duration']

    print(f"Total duration: {total_duration:.1f} seconds")
    print(f"Total segments: {len(transcript_segments)}")
    print(f"Method: {SEGMENTATION_METHOD}")

    # Detect topic boundaries
    topic_boundaries = detect_topic_boundaries(transcript_segments)
    print(f"Topic boundaries detected: {len(topic_boundaries)}")

    # Collect all boundary points
    boundary_points = set()

    # Add topic boundaries
    for boundary in topic_boundaries:
        boundary_points.add(boundary['timestamp'])

    # Add slide changes if available
    if slides_data and 'slides' in slides_data:
        for slide in slides_data['slides']:
            boundary_points.add(slide['timestamp'])

    # Convert to sorted list
    boundary_points = sorted(list(boundary_points))

    # Add start and end
    boundary_points = [0.0] + boundary_points + [total_duration]

    # Create chunks from boundaries
    chunks = []

    for i in range(len(boundary_points) - 1):
        start_time = boundary_points[i]
        end_time = boundary_points[i + 1]
        duration = end_time - start_time

        # Skip chunks that are too short (merge with next)
        if duration < MIN_CHUNK_DURATION and i < len(boundary_points) - 2:
            continue

        # Collect segments in this chunk
        chunk_segments = []
        chunk_text = []

        for segment in transcript_segments:
            # Check if segment overlaps with chunk time range
            if (segment['start'] >= start_time and segment['start'] < end_time) or \
               (segment['end'] > start_time and segment['end'] <= end_time) or \
               (segment['start'] <= start_time and segment['end'] >= end_time):
                chunk_segments.append(segment)
                chunk_text.append(segment['text'])

        # Skip empty chunks
        if not chunk_segments:
            continue

        # Get topic/title
        topic = extract_topic(chunk_text, start_time, video_id)

        # Check for diagrams (from OCR)
        has_diagram = False
        ocr_text = ""

        if ocr_data and 'results' in ocr_data:
            for result in ocr_data['results']:
                if start_time <= result['timestamp'] < end_time:
                    if result['text']:
                        ocr_text += " " + result['text']
                        has_diagram = True

        # Get slide number if available
        slide_number = None
        if slides_data and 'slides' in slides_data:
            for slide in slides_data['slides']:
                if abs(slide['timestamp'] - start_time) < 5:  # Within 5 seconds
                    slide_number = slide['slide_number']
                    break

        # Create chunk
        chunk = {
            'chunk_id': len(chunks) + 1,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'topic': topic,
            'transcript': ' '.join(chunk_text).strip(),
            'segment_count': len(chunk_segments),
            'has_diagram': has_diagram,
            'ocr_text': ocr_text.strip(),
            'slide_number': slide_number,
            'video_id': video_id,
            'video_url': f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}"
        }

        chunks.append(chunk)

        print(f"  Chunk {chunk['chunk_id']}: {chunk['topic']}")
        print(f"    Time: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)")
        print(f"    Segments: {chunk['segment_count']}")
        print(f"    Has diagram: {has_diagram}")

    # Merge very short chunks with neighbors
    chunks = merge_short_chunks(chunks)

    return chunks

def extract_topic(texts, timestamp, video_id):
    """Extract topic title from text segments."""
    # Combine first few sentences
    combined_text = ' '.join(texts[:3]) if texts else ""

    # Look for topic patterns
    # This is simplified - could use NLP in production
    topic_indicators = [
        "today we're going to talk about",
        "let's discuss",
        "now let's move on to",
        "next topic",
        "in this lecture",
        "welcome to"
    ]

    for indicator in topic_indicators:
        if indicator in combined_text.lower():
            # Extract text after indicator
            start_idx = combined_text.lower().find(indicator)
            topic = combined_text[start_idx + len(indicator):].strip()
            # Take first sentence or up to 50 chars
            if '.' in topic:
                topic = topic.split('.')[0]
            topic = topic[:50] + ("..." if len(topic) > 50 else "")
            return topic.capitalize()

    # Fallback: use timestamp-based title
    minute = int(timestamp // 60)
    return f"Segment at {minute} minutes"

def merge_short_chunks(chunks):
    """Merge very short chunks with neighbors."""
    if len(chunks) <= 1:
        return chunks

    merged = [chunks[0]]

    for chunk in chunks[1:]:
        last_chunk = merged[-1]

        # If last chunk is too short, merge
        if last_chunk['duration'] < MIN_CHUNK_DURATION:
            # Merge with current chunk
            merged_chunk = {
                **last_chunk,
                'end_time': chunk['end_time'],
                'duration': chunk['end_time'] - last_chunk['start_time'],
                'transcript': last_chunk['transcript'] + " " + chunk['transcript'],
                'segment_count': last_chunk['segment_count'] + chunk['segment_count'],
                'ocr_text': last_chunk['ocr_text'] + " " + chunk['ocr_text']
            }
            merged[-1] = merged_chunk
        else:
            merged.append(chunk)

    return merged

def process_video(video_id):
    """Process video to create chunks."""
    print(f"\n{'#'*60}")
    print(f"# Processing Video: {video_id}")
    print(f"{'#'*60}")

    # Load data
    transcript = load_transcript(video_id)
    if not transcript:
        return None

    slides_data = load_slides(video_id)
    ocr_data = load_ocr(video_id)

    # Check if already processed
    chunks_file = CHUNKS_DIR / f"{video_id}_chunks.json"
    if chunks_file.exists():
        print(f"✓ Chunks already created for {video_id}")
        with open(chunks_file) as f:
            return json.load(f)

    # Create chunks
    chunks = create_smart_chunks(video_id, transcript, slides_data, ocr_data)

    if not chunks:
        print(f"✗ No chunks created for {video_id}")
        return None

    # Create metadata
    metadata = {
        'video_id': video_id,
        'total_chunks': len(chunks),
        'method': SEGMENTATION_METHOD,
        'chunks': chunks
    }

    # Save chunks
    with open(chunks_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Created {len(chunks)} chunks")
    print(f"✓ Saved to: {chunks_file}")

    return metadata

def process_all_videos():
    """Process all videos to create chunks."""
    # Get all transcripts
    transcript_files = list(TRANSCRIPT_DIR.glob("*.json"))

    if not transcript_files:
        print(f"⚠ No transcripts found in {TRANSCRIPT_DIR}")
        print("Run transcribe.py first")
        return []

    print(f"\nFound {len(transcript_files)} transcripts")

    # Process each video
    all_metadata = []
    success_count = 0
    failed_count = 0

    for transcript_file in sorted(transcript_files):
        video_id = transcript_file.stem

        metadata = process_video(video_id)

        if metadata:
            all_metadata.append(metadata)
            success_count += 1
        else:
            failed_count += 1

    # Summary
    print("\n" + "="*60)
    print("CHUNK CREATION SUMMARY")
    print("="*60)
    print(f"Total videos: {len(transcript_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {failed_count}")

    total_chunks = sum(m['total_chunks'] for m in all_metadata)
    print(f"Total chunks created: {total_chunks}")

    # Calculate statistics
    durations = []
    for metadata in all_metadata:
        for chunk in metadata['chunks']:
            durations.append(chunk['duration'])

    if durations:
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        print(f"Chunk duration stats:")
        print(f"  Average: {avg_duration:.1f}s ({avg_duration/60:.1f} minutes)")
        print(f"  Min: {min_duration:.1f}s")
        print(f"  Max: {max_duration:.1f}s")

    # Save processing log
    log_file = METADATA_DIR / "chunking_log.json"
    with open(log_file, 'w') as f:
        json.dump({
            'total': len(transcript_files),
            'success': success_count,
            'failed': failed_count,
            'total_chunks': total_chunks,
            'method': SEGMENTATION_METHOD,
            'avg_chunk_duration': avg_duration if durations else 0
        }, f, indent=2)

    print(f"\n✓ Log saved: {log_file}")

    return all_metadata

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    print("="*60)
    print("Smart Video Chunking")
    print("Contribution 1: Timestamp-Aware Video RAG")
    print("="*60)

    # Create directories
    create_directories()

    # Process all videos
    process_all_videos()

    print("\n" + "="*60)
    print("✓ Video processing pipeline complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Download output files from Kaggle")
    print("2. Extract chunks to local project:")
    print("   unzip chunks.zip -d data/chunks/")
    print("3. Load chunks in main RAG system")

if __name__ == "__main__":
    main()
