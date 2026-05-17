"""
Complete Chunk Analysis - Analyze all content in the RAG system
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_pickle
from collections import Counter

def analyze_chunks():
    """Analyze all chunks in the system"""

    chunks = load_pickle('data/processed/chunks/chunks.pkl')

    print('=' * 60)
    print('COMPLETE CHUNK ANALYSIS')
    print('=' * 60)
    print(f'\nTotal chunks: {len(chunks):,}')

    # Source distribution
    source_counts = Counter([c.get('source_name') for c in chunks])
    print('\n[Source Distribution]')
    for source, count in source_counts.most_common():
        print(f'  {source}: {count:,} chunks')

    # Analyze each source
    for source in source_counts.keys():
        source_chunks = [c for c in chunks if c.get('source_name') == source]

        print(f'\n\n{"=" * 60}')
        print(f'SOURCE: {source}')
        print('=' * 60)
        print(f'Total chunks: {len(source_chunks):,}')

        # Get unique chapters/sections
        chapters = [c.get('chapter', c.get('section', 'N/A')) for c in source_chunks]
        unique_chapters = list(set(chapters))[:20]  # First 20 unique
        print(f'\n[Chapters/Sections] (first 20):')
        for i, chapter in enumerate(unique_chapters, 1):
            print(f'  {i}. {chapter}')

        # Sample chunks to see content type
        print(f'\n[Sample Content] (first 3 chunks):')
        for i, chunk in enumerate(source_chunks[:3], 1):
            text_preview = chunk['text'][:150].replace('\n', ' ')
            print(f'  Chunk {i}: {text_preview}...')
            print()

    print('\n' + '=' * 60)
    print('ANALYSIS COMPLETE')
    print('=' * 60)

if __name__ == '__main__':
    analyze_chunks()
