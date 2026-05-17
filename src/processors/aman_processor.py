"""
Aman.ai Article Processor
Converts scraped articles into chunks and creates FAISS index
"""

import pickle
import re
from pathlib import Path
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer


class AmanProcessor:
    """Process scraped Aman.ai articles into RAG-ready chunks"""

    def __init__(self, input_dir: str = "data/aman_primers",
                 output_dir: str = "data/processed/aman_primers",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize processor

        Args:
            input_dir: Directory with scraped markdown files
            output_dir: Directory for processed output
            embedding_model: Model name for embeddings
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        print(f"Model loaded successfully")

    def load_articles(self) -> List[Dict[str, Any]]:
        """
        Load all scraped articles

        Returns:
            List of article dicts
        """
        articles = []
        md_files = list(self.input_dir.rglob("*.md"))

        print(f"Found {len(md_files)} markdown files")

        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract metadata from frontmatter
                metadata = self._parse_metadata(content, md_file)
                articles.append(metadata)

            except Exception as e:
                print(f"Error loading {md_file}: {e}")

        print(f"Loaded {len(articles)} articles")
        return articles

    def _parse_metadata(self, content: str, filepath: Path) -> Dict[str, Any]:
        """Parse article metadata and content"""
        lines = content.split('\n')

        metadata = {
            'source_name': filepath.stem,
            'category': filepath.parent.name,
            'filepath': str(filepath),
            'content': content
        }

        # Extract metadata from frontmatter-like structure
        for line in lines[:20]:  # Check first 20 lines for metadata
            if line.startswith('**Source**:'):
                metadata['url'] = line.split('**Source**:')[1].strip()
            elif line.startswith('**Category**:'):
                metadata['category'] = line.split('**Category**:')[1].strip()
            elif line.startswith('**Word Count**:'):
                metadata['word_count'] = int(line.split('**Word Count**:')[1].strip())

        return metadata

    def chunk_article(self, article: Dict[str, Any],
                     chunk_size: int = 500,
                     overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Split article into semantic chunks

        Args:
            article: Article dict with content
            chunk_size: Target chunk size (words)
            overlap: Overlap between chunks (words)

        Returns:
            List of chunk dicts
        """
        content = article['content']
        chunks = []

        # Remove frontmatter
        content = re.sub(r'^---.*?---\s*', '', content, flags=re.DOTALL)

        # Split by headings first (semantic boundaries)
        sections = re.split(r'\n#{1,3}\s+', content)

        current_chunk = []
        current_size = 0
        chunk_num = 0

        for section in sections:
            if not section.strip():
                continue

            words = section.split()
            section_size = len(words)

            # If section fits in current chunk
            if current_size + section_size <= chunk_size:
                current_chunk.extend(words)
                current_size += section_size
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(self._create_chunk(
                        ' '.join(current_chunk),
                        chunk_num,
                        article
                    ))
                    chunk_num += 1

                # Start new chunk (with overlap)
                overlap_words = current_chunk[-overlap:] if current_chunk else []
                current_chunk = overlap_words + words
                current_size = len(current_chunk)

        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                ' '.join(current_chunk),
                chunk_num,
                article
            ))

        return chunks

    def _create_chunk(self, text: str, chunk_num: int,
                     article: Dict[str, Any]) -> Dict[str, Any]:
        """Create a chunk dict with metadata"""
        return {
            'chunk_id': f"{article['source_name']}_chunk_{chunk_num}",
            'text': text.strip(),
            'source_name': article['source_name'],
            'source_type': 'aman_primer',
            'category': article['category'],
            'url': article.get('url', ''),
            'chunk_number': chunk_num,
            'word_count': len(text.split())
        }

    def process_all_articles(self, chunk_size: int = 500,
                           overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Process all articles into chunks

        Args:
            chunk_size: Target chunk size
            overlap: Overlap between chunks

        Returns:
            List of all chunks
        """
        print("="*80)
        print("PROCESSING AMAN.AI ARTICLES")
        print("="*80)

        # Load articles
        articles = self.load_articles()

        if not articles:
            print("No articles to process!")
            return []

        # Chunk articles
        print(f"\nChunking articles (chunk_size={chunk_size}, overlap={overlap})...")
        all_chunks = []

        for article in articles:
            chunks = self.chunk_article(article, chunk_size, overlap)
            all_chunks.extend(chunks)
            print(f"  {article['source_name']}: {len(chunks)} chunks")

        print(f"\nTotal chunks: {len(all_chunks)}")

        # Generate embeddings
        print(f"\nGenerating embeddings for {len(all_chunks)} chunks...")
        texts = [chunk['text'] for chunk in all_chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)

        # Create FAISS index
        print(f"\nCreating FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))

        print(f"FAISS index created with {index.ntotal} vectors")

        # Save to disk
        print(f"\nSaving to {self.output_dir}...")

        # Save index
        index_path = self.output_dir / "aman_index.faiss"
        faiss.write_index(index, str(index_path))

        # Save chunks
        chunks_path = self.output_dir / "aman_metadata.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump(all_chunks, f)

        print(f"  [OK] Saved index: {index_path}")
        print(f"  [OK] Saved chunks: {chunks_path}")

        # Statistics
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print(f"  Articles: {len(articles)}")
        print(f"  Chunks: {len(all_chunks)}")
        print(f"  Embedding dimension: {dimension}")
        print(f"  Output directory: {self.output_dir}")
        print("="*80)

        return all_chunks


def main():
    """Main processing function"""
    import argparse

    parser = argparse.ArgumentParser(description="Process Aman.ai articles")
    parser.add_argument('--input', type=str, default='data/aman_primers', help='Input directory')
    parser.add_argument('--output', type=str, default='data/processed/aman_primers', help='Output directory')
    parser.add_argument('--chunk-size', type=int, default=500, help='Chunk size in words')
    parser.add_argument('--overlap', type=int, default=50, help='Overlap between chunks')

    args = parser.parse_args()

    processor = AmanProcessor(
        input_dir=args.input,
        output_dir=args.output
    )

    processor.process_all_articles(
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )


if __name__ == "__main__":
    main()
