"""
Complete Chunk Analysis - Analyze all content in the RAG system
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_pickle
from collections import Counter

def clean_text(text):
    """Remove non-ascii characters"""
    return ''.join([c if ord(c) < 128 else '?' for c in text])

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
        chapters = []
        for c in source_chunks:
            chapter = c.get('chapter', c.get('section', 'N/A'))
            chapters.append(chapter)

        unique_chapters = list(set(chapters))[:15]  # First 15 unique
        print(f'\n[Chapters/Sections] (first 15):')
        for i, chapter in enumerate(unique_chapters, 1):
            chapter_clean = clean_text(str(chapter))
            print(f'  {i}. {chapter_clean}')

        # Show content types by checking keywords
        print(f'\n[Content Analysis]')
        all_text = ' '.join([c['text'][:500] for c in source_chunks[:10]])
        all_text_lower = all_text.lower()

        topics = {
            'RNN': 'recurrent',
            'CNN': 'convolution',
            'gradient': 'gradient',
            'backprop': 'backprop',
            'optimization': 'optim',
            'loss': 'loss function',
            'neural': 'neural network',
            'deep learning': 'deep learning',
            'machine learning': 'machine learning',
        }

        for topic, keyword in topics.items():
            if keyword in all_text_lower:
                count = all_text_lower.count(keyword)
                print(f'  {topic}: found (mentioned {count} times)')
            else:
                print(f'  {topic}: not found')

if __name__ == '__main__':
    analyze_chunks()
