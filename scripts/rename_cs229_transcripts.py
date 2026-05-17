"""
Rename CS229 Transcript Files to Systematic Format
Extracts lecture numbers and topics from NoteGPT filenames
"""

import re
from pathlib import Path

def extract_lecture_info(filename):
    """Extract lecture number and topic from NoteGPT filename"""
    # Pattern: NoteGPT_TRANSCRIPT_Stanford CS229 ... Lecture {N}.srt
    lecture_match = re.search(r'Lecture (\d+)', filename)
    if not lecture_match:
        return None, None

    lecture_num = int(lecture_match.group(1))

    # Extract topic by removing common prefixes/suffixes
    # Remove: NoteGPT_TRANSCRIPT_, Stanford CS229, I, 2022, Lecture N, .srt
    topic = filename

    # Remove common prefixes
    topic = topic.replace('NoteGPT_TRANSCRIPT_', '')
    topic = topic.replace('Stanford CS229 ', '')
    topic = topic.replace('Stanford ', '')
    topic = topic.replace('Machine Learning ', '')

    # Remove dates and lecture info
    topic = re.sub(r'\s*I\s*2022\s*I\s*', ' - ', topic)
    topic = re.sub(r'\s*I\s*Lecture\s*\d+\s*', '', topic)
    topic = re.sub(r'\.srt$', '', topic)

    # Clean up extra spaces and dashes
    topic = re.sub(r'\s+-\s+', ' - ', topic)
    topic = topic.strip()

    # Convert to filename-safe format
    topic_safe = topic.replace(' ', '_').replace('-', '_')
    topic_safe = re.sub(r'_+', '_', topic_safe)  # Multiple underscores to single
    topic_safe = topic_safe.strip('_')

    return lecture_num, topic_safe

def main():
    """Rename all CS229 transcript files"""
    transcripts_dir = Path("data/transcripts/cs229")

    if not transcripts_dir.exists():
        print(f"Error: Directory {transcripts_dir} not found")
        return

    # Get all SRT files
    srt_files = list(transcripts_dir.glob("*.srt"))
    print(f"Found {len(srt_files)} transcript files\n")

    # Create mapping of old names to new names
    file_mapping = {}
    for srt_file in srt_files:
        lecture_num, topic = extract_lecture_info(srt_file.name)

        if lecture_num is None:
            print(f"Warning: Could not extract lecture number from {srt_file.name}")
            continue

        # New filename: CS229_LectureXX_Topic.srt
        new_name = f"CS229_L{lecture_num:02d}_{topic}.srt"
        file_mapping[srt_file] = transcripts_dir / new_name

    # Sort by lecture number and display
    sorted_files = sorted(file_mapping.items(), key=lambda x: extract_lecture_info(x[0].name)[0])

    print("Planned Renames:")
    print("=" * 80)
    for old_file, new_file in sorted_files:
        lecture_num = extract_lecture_info(old_file.name)[0]
        print(f"Lecture {lecture_num:2d}: {old_file.name[:60]}...")
        print(f"         -> {new_file.name}")
        print()

    # Confirm and rename
    print("=" * 80)
    response = input("Proceed with renaming? (yes/no): ")

    if response.lower() in ['yes', 'y']:
        for old_file, new_file in sorted_files:
            if new_file.exists():
                print(f"Warning: {new_file.name} already exists, skipping...")
                continue

            old_file.rename(new_file)
            print(f"[OK] Renamed: {new_file.name}")

        print(f"\n[OK] Successfully renamed {len(sorted_files)} files")

        # Create metadata file
        create_metadata_file(transcripts_dir, sorted_files)
    else:
        print("Renaming cancelled")

def create_metadata_file(transcripts_dir, file_mapping):
    """Create a metadata file with lecture information"""
    metadata_path = transcripts_dir / "CS229_Metadata.txt"

    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write("CS229 Machine Learning Course - Lecture Metadata\n")
        f.write("=" * 80 + "\n\n")

        for old_file, new_file in file_mapping:
            lecture_num, topic = extract_lecture_info(old_file.name)

            # Convert topic back to readable format
            topic_readable = topic.replace('_', ' ').title()

            f.write(f"Lecture {lecture_num:2d}: {topic_readable}\n")
            f.write(f"  File: {new_file.name}\n")
            f.write(f"  Original: {old_file.name}\n")
            f.write("\n")

    print(f"\n[OK] Created metadata file: {metadata_path}")

if __name__ == "__main__":
    main()
