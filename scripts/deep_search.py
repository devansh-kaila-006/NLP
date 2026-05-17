"""
Deep Search - Find where topics actually are in the chunks
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_pickle

def deep_search():
    """Search for specific topics across all chunks"""

    chunks = load_pickle('data/processed/chunks/chunks.pkl')

    topics = {
        'RNN': ['rnn', 'recurrent neural network'],
        'CNN': ['cnn', 'convolutional neural network'],
        'backprop': ['backprop', 'back propagation'],
        'gradient descent': ['gradient descent'],
    }

    print('=' * 60)
    print('DEEP TOPIC SEARCH ACROSS ALL CHUNKS')
    print('=' * 60)
    print(f'\nTotal chunks to search: {len(chunks):,}\n')

    for topic_name, keywords in topics.items():
        print(f'\n{"=" * 60}')
        print(f'TOPIC: {topic_name.upper()}')
        print('=' * 60)

        found_count = 0
        sources_found = set()

        for chunk in chunks:
            text_lower = chunk['text'].lower()

            # Check if any keyword matches
            for keyword in keywords:
                if keyword in text_lower:
                    found_count += 1
                    source = chunk.get('source_name', 'Unknown')

                    if source not in sources_found and found_count <= 5:
                        sources_found.add(source)
                        # Get a preview
                        preview_start = chunk['text'].lower().find(keyword)
                        start = max(0, preview_start - 50)
                        end = min(len(chunk['text']), preview_start + 150)

                        print(f'\nFound in {source} (chunk {chunk.get("chunk_id", "N/A")})')
                        print(f'Location: {keyword}')
                        print(f'Preview: ...{chunk["text"][start:end]}...')
                        print()

                    break  # Only count once per chunk

        print(f'\nTotal mentions: {found_count}')
        print(f'Sources with content: {list(sources_found)}')

if __name__ == '__main__':
    deep_search()
