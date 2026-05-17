"""
SRT Transcript Loader
Parses SRT subtitle files with timestamps for video processing
"""

import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class TranscriptSegment:
    """Represents a segment of transcript with timing"""
    segment_id: int
    start_time: float  # seconds
    end_time: float    # seconds
    text: str
    duration: float

    def __repr__(self):
        mins, secs = divmod(self.start_time, 60)
        return f"[{int(mins):02d}:{int(secs):02d}] {self.text[:50]}..."


class SRTLoader:
    """
    Load and parse SRT subtitle files

    SRT Format:
    1
    00:00:05,000 --> 00:01:05,000
    transcript text here
    """

    def __init__(self):
        self.logger = None  # Can be set later

    def parse_timestamp(self, timestamp_str: str) -> float:
        """
        Convert SRT timestamp to seconds

        Args:
            timestamp_str: Format "00:00:05,000"

        Returns:
            Time in seconds (float)
        """
        # Format: HH:MM:SS,mmm
        match = re.match(r'(\d+):(\d+):(\d+),(\d+)', timestamp_str)
        if match:
            hours, minutes, seconds, milliseconds = map(int, match.groups())
            return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        return 0.0

    def load_srt(self, srt_path: str) -> List[TranscriptSegment]:
        """
        Load SRT file and parse segments

        Args:
            srt_path: Path to SRT file

        Returns:
            List of TranscriptSegment objects
        """
        srt_path = Path(srt_path)

        if not srt_path.exists():
            raise FileNotFoundError(f"SRT file not found: {srt_path}")

        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        segments = []
        blocks = re.split(r'\n\s*\n', content.strip())

        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue

            try:
                # Parse segment ID
                segment_id = int(lines[0].strip())

                # Parse timestamp line
                timestamp_match = re.match(
                    r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})',
                    lines[1]
                )

                if not timestamp_match:
                    continue

                start_time = self.parse_timestamp(timestamp_match.group(1))
                end_time = self.parse_timestamp(timestamp_match.group(2))

                # Parse transcript text (may span multiple lines)
                text = ' '.join(lines[2:]).strip()

                segment = TranscriptSegment(
                    segment_id=segment_id,
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                    duration=end_time - start_time
                )

                segments.append(segment)

            except (ValueError, IndexError) as e:
                # Skip malformed segments
                if self.logger:
                    self.logger.warning(f"Skipping malformed segment in {srt_path}: {e}")
                continue

        if self.logger:
            self.logger.info(f"Loaded {len(segments)} segments from {srt_path.name}")

        return segments

    def load_multiple_srts(
        self,
        srt_dir: str,
        pattern: str = "*.srt"
    ) -> Dict[str, List[TranscriptSegment]]:
        """
        Load multiple SRT files from directory

        Args:
            srt_dir: Directory containing SRT files
            pattern: Glob pattern for files

        Returns:
            Dictionary mapping filename to segments
        """
        srt_dir = Path(srt_dir)
        all_segments = {}

        srt_files = list(srt_dir.glob(pattern))

        for srt_file in sorted(srt_files):
            try:
                segments = self.load_srt(srt_file)
                all_segments[srt_file.name] = segments
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to load {srt_file.name}: {e}")

        if self.logger:
            self.logger.info(f"Loaded {len(all_segments)} SRT files from {srt_dir}")

        return all_segments


if __name__ == "__main__":
    # Test SRT loader
    loader = SRTLoader()

    # Load Lecture 1
    segments = loader.load_srt("data/transcripts/cs229/CS229_L01_I_Introduction_Lecture_1.srt")

    print(f"Loaded {len(segments)} segments")
    print("\nFirst 5 segments:")
    for segment in segments[:5]:
        print(segment)
        print()

    # Calculate total duration
    total_duration = sum(seg.duration for seg in segments)
    print(f"\nTotal duration: {total_duration / 60:.1f} minutes")
