"""
Comprehensive Phase 2 Code Review and Testing
Tests each component individually
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger


def test_imports():
    """Test all imports work correctly"""
    print("\n" + "=" * 60)
    print("Test 1: Imports")
    print("=" * 60)

    tests = [
        ("PDF Loader", "from src.loaders.pdf_loader import PDFLoader"),
        ("ZIP Loader", "from src.loaders.zip_loader import ZIPLoader"),
        ("Web Loader", "from src.loaders.web_loader import WebLoader"),
        ("Document Loader", "from src.loaders.document_loader import DocumentLoader"),
        ("Semantic Chunker", "from src.processors.semantic_chunker import SemanticChunker"),
    ]

    failed = []
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"[OK] {name}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed.append((name, e))

    if failed:
        print(f"\n[FAIL] {len(failed)} imports failed")
        return False
    else:
        print(f"\n[OK] All imports successful")
        return True


def test_pdf_loader():
    """Test PDF loader with available PDFs"""
    print("\n" + "=" * 60)
    print("Test 2: PDF Loader")
    print("=" * 60)

    from src.loaders.pdf_loader import PDFLoader

    pdf_dir = Path("data/pdfs")
    test_files = [
        ("ML.pdf", "ML_Course_Notes"),
        ("DL.pdf", "DL_Textbook"),
    ]

    available_files = [(f, n) for f, n in test_files if (pdf_dir / f).exists()]

    if not available_files:
        print("[SKIP] No PDF files found in data/pdfs/")
        print("  Place ML.pdf and/or DL.pdf in data/pdfs/ to test")
        return True

    loader = PDFLoader()

    for filename, source_name in available_files:
        print(f"\nTesting {filename}...")
        try:
            chapters = loader.extract_chapters(pdf_dir / filename, source_name)
            print(f"  [OK] Extracted {len(chapters)} chapters")

            if chapters:
                print(f"  Sample chapter:")
                ch = chapters[0]
                print(f"    Title: {ch.get('title', 'N/A')[:50]}...")
                print(f"    Pages: {ch.get('page_start', 'N/A')}-{ch.get('page_end', 'N/A')}")
                print(f"    Text length: {len(ch.get('text', ''))} chars")
        except Exception as e:
            print(f"  [FAIL] {e}")
            return False

    return True


def test_chunker():
    """Test semantic chunker"""
    print("\n" + "=" * 60)
    print("Test 3: Semantic Chunker")
    print("=" * 60)

    from src.processors.semantic_chunker import SemanticChunker

    # Create test document
    test_doc = {
        'text': """
        Chapter 1: Introduction to Machine Learning

        Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.

        Supervised Learning
        Supervised learning involves training a model on labeled data. The model learns to map inputs to outputs based on examples.

        Unsupervised Learning
        Unsupervised learning deals with unlabeled data. The algorithm must find patterns in the data without explicit guidance.

        Reinforcement Learning
        Reinforcement learning is about training agents to make decisions in an environment to maximize rewards.
        """ * 3,  # Repeat to get more content
        'source_name': 'Test_Source',
        'source_type': 'pdf',
        'chapter': '1',
        'title': 'Introduction',
        'chunk_id': 'test_doc'
    }

    chunker = SemanticChunker()

    try:
        chunks = chunker.create_chunks_with_overlap(test_doc['text'], test_doc)
        print(f"[OK] Created {len(chunks)} chunks")

        if chunks:
            print(f"\nSample chunks:")
            for i, chunk in enumerate(chunks[:3]):
                print(f"\n  Chunk {i + 1}:")
                print(f"    ID: {chunk['chunk_id']}")
                print(f"    Tokens: {chunk['token_count']}")
                print(f"    Preview: {chunk['text'][:100]}...")

            # Check token counts
            token_counts = [c['token_count'] for c in chunks]
            avg_tokens = sum(token_counts) // len(token_counts)

            print(f"\nToken statistics:")
            print(f"  Min: {min(token_counts)}")
            print(f"  Max: {max(token_counts)}")
            print(f"  Avg: {avg_tokens}")

            # Verify chunks are in reasonable range
            if min(token_counts) < 100:
                print(f"[WARN] Some chunks are too small (< 100 tokens)")
            if max(token_counts) > 800:
                print(f"[WARN] Some chunks are too large (> 800 tokens)")

        return True

    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_loader():
    """Test document loader orchestrator"""
    print("\n" + "=" * 60)
    print("Test 4: Document Loader Orchestrator")
    print("=" * 60)

    from src.loaders.document_loader import DocumentLoader

    loader = DocumentLoader()

    # Test configuration loading
    print("[OK] DocumentLoader initialized")

    # Check sources
    from src.config import PDF_SOURCES
    print(f"[OK] Found {len(PDF_SOURCES)} configured sources:")
    for name, config in PDF_SOURCES.items():
        print(f"  - {name}: {config['type']}")

    return True


def test_end_to_end():
    """Test full ingestion pipeline with available data"""
    print("\n" + "=" * 60)
    print("Test 5: End-to-End Pipeline (with available data)")
    print("=" * 60)

    from src.loaders.document_loader import DocumentLoader
    from src.processors.semantic_chunker import SemanticChunker

    loader = DocumentLoader()
    chunker = SemanticChunker()

    # Check what data is available
    from src.config import PDF_SOURCES
    pdf_dir = Path("data/pdfs")

    available_sources = []
    for name, config in PDF_SOURCES.items():
        if config['type'] == 'pdf':
            pdf_path = Path(config['path'])
            if pdf_path.exists():
                available_sources.append(name)

    if not available_sources:
        print("[SKIP] No PDF data files found")
        print("  Place PDF files in data/pdfs/ to test end-to-end")
        return True

    print(f"Found {len(available_sources)} available source(s): {available_sources}")

    try:
        # Load documents
        print("\nLoading documents...")
        documents = loader.load_all_sources(source_filter=available_sources)

        if not documents:
            print("[FAIL] No documents loaded")
            return False

        print(f"[OK] Loaded {len(documents)} documents")

        # Normalize
        print("Normalizing documents...")
        normalized = loader.normalize_documents(documents)
        print(f"[OK] Normalized {len(normalized)} documents")

        # Chunk
        print("Chunking documents...")
        chunks = chunker.chunk_hierarchical(normalized)

        if not chunks:
            print("[FAIL] No chunks created")
            return False

        print(f"[OK] Created {len(chunks)} chunks")

        # Show sample
        print(f"\nSample chunk:")
        chunk = chunks[0]
        print(f"  ID: {chunk['chunk_id']}")
        print(f"  Source: {chunk['source_name']}")
        print(f"  Tokens: {chunk['token_count']}")
        print(f"  Preview: {chunk['text'][:150]}...")

        return True

    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Phase 2 Code Review and Testing")
    print("=" * 60)

    results = {
        "Imports": test_imports(),
        "PDF Loader": test_pdf_loader(),
        "Semantic Chunker": test_chunker(),
        "Document Loader": test_document_loader(),
        "End-to-End Pipeline": test_end_to_end(),
    }

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed or skipped")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
