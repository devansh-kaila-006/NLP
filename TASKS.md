# Implementation Tasks - PDF-Only RAG Pipeline

**Project:** Multi-Modal RAG for ML/DL Learning (PDF Phase)
**Timeline:** 5 Days
**Team:** 1 Person
**Last Updated:** 2026-05-04

---

## Progress Overview

- [ ] Phase 1: Foundation (Day 1) - 0/5 complete
- [ ] Phase 2: Data Ingestion (Day 2) - 0/6 complete
- [ ] Phase 3: Embeddings + Index (Day 3) - 0/3 complete
- [ ] Phase 4: Retrieval Pipeline (Day 4) - 0/4 complete
- [ ] Phase 5: CLI + Testing (Day 5) - 0/4 complete

**Total Progress:** 0/22 tasks (0%)

---

## Phase 1: Foundation (Day 1)

### 1.1 Project Structure Setup
- [ ] Create directory structure
  ```bash
  mkdir -p data/pdfs data/processed/{chunks,embeddings,indices} data/cache
  mkdir -p src/{loaders,processors,embeddings,vector_store}
  mkdir -p src/{retrieval,reranking,generation,pipeline,utils}
  mkdir -p scripts tests
  ```

- [ ] Create all `__init__.py` files in src/ subdirectories
- [ ] Create `.gitignore` file
- [ ] Verify directory structure matches specification

### 1.2 Dependencies Setup
- [ ] Create `requirements.txt` with all dependencies
  - langchain, langchain-community
  - pypdf (PDF processing)
  - beautifulsoup4, requests, lxml (web scraping)
  - sentence-transformers, faiss-cpu (embeddings + vector store)
  - transformers (reranker)
  - google-generativeai (Gemini API)
  - numpy, python-dotenv
  - pytest (testing)

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify installations: test imports for critical packages

### 1.3 Configuration Files
- [ ] Create `src/config.py`
  - PDF_SOURCES dictionary (ML.pdf, DL.pdf, sklearn zip, PyTorch web)
  - Chunking parameters (chunk_size=400, overlap=50)
  - Embedding model (all-MiniLM-L6-v2)
  - Reranker model (ms-marco-MiniLM-L-6-v2)
  - LLM config (Gemini Flash)
  - Path configurations

- [ ] Create `.env` template
  - GOOGLE_API_KEY placeholder
  - Add to .gitignore

### 1.4 Utility Modules
- [ ] Create `src/utils/logger.py`
  - Setup logging configuration
  - File handler + console handler
  - Different log levels for modules
  - Timestamp formatting

- [ ] Create `src/utils/helpers.py`
  - `save_pickle()` - Save objects to disk
  - `load_pickle()` - Load objects from disk
  - `ensure_dir()` - Create directory if not exists
  - `get_file_size()` - Get file size in MB
  - Timer context manager for profiling

### 1.5 Verification
- [ ] Test logger functionality
- [ ] Test helper functions
- [ ] Verify config loads correctly
- [ ] Test environment variable loading

**Phase 1 Complete:** ✅ Foundation ready for data ingestion

---

## Phase 2: Data Ingestion (Day 2)

### 2.1 PDF Loader
- [ ] Create `src/loaders/pdf_loader.py`
  - `extract_text_from_pdf()` - Extract text from PDF
  - `detect_chapters()` - Detect chapter boundaries
    - Try PDF outline/TOC first
    - Fallback to regex patterns
    - Fallback to font size analysis
  - `extract_pages_with_metadata()` - Extract text per page
  - Returns: List of chapters with page ranges

- [ ] Test with ML.pdf (34 pages)
  - Verify chapter detection
  - Verify text extraction
  - Check metadata

- [ ] Test with DL.pdf (189 pages)
  - Verify chapter detection
  - Verify text extraction quality
  - Check memory usage

### 2.2 ZIP Loader (scikit-learn-docs.zip)
- [ ] Create `src/loaders/zip_loader.py`
  - `extract_zip()` - Extract ZIP to cache directory
  - `list_html_files()` - List all HTML files
  - `parse_html_files()` - Parse HTML with BeautifulSoup
  - `detect_sections()` - Detect documentation structure
  - Returns: List of sections with text

- [ ] Test with scikit-learn-docs.zip
  - Extract to cache/sklearn_docs/
  - Parse HTML structure
  - Extract API documentation sections
  - Verify text quality

### 2.3 Web Loader (PyTorch docs)
- [ ] Create `src/loaders/web_loader.py`
  - `scrape_url()` - Scrape single URL
  - `follow_links()` - Follow internal documentation links
  - `parse_html()` - Parse HTML content
  - `cache_pages()` - Cache HTML locally
  - `rate_limiting()` - Respect robots.txt, add delays
  - Returns: List of documentation pages

- [ ] Test with PyTorch docs URL
  - Scrape index page
  - Follow key documentation links
  - Cache to data/cache/pytorch_docs/
  - Verify extracted text

### 2.4 Main Document Loader
- [ ] Create `src/loaders/document_loader.py`
  - `load_all_sources()` - Main orchestrator
  - Route to appropriate loader based on source type
  - Normalize all outputs to standard format
  - Combine all sources into single list
  - Returns: List of document objects with metadata

- [ ] Test multi-source loading
  - Load all 4 sources
  - Verify metadata consistency
  - Check for duplicates
  - Log source statistics

### 2.5 Semantic Chunker
- [ ] Create `src/processors/semantic_chunker.py`
  - `chunk_hierarchical()` - Hierarchical chunking
    - Chapter level → Section level → Paragraph chunks
  - `detect_boundaries()` - Detect semantic boundaries
  - `add_overlap()` - Add token overlap between chunks
  - `enrich_metadata()` - Add chapter, section, page metadata
  - `save_chunks()` - Save chunks to disk
  - Returns: List of chunks with rich metadata

- [ ] Test hierarchical chunking
  - Test on ML.pdf (smaller)
  - Test on DL.pdf (larger)
  - Verify chunk sizes (300-500 tokens)
  - Verify metadata quality
  - Check overlap between chunks

### 2.6 Ingestion Script
- [ ] Create `scripts/01_ingest_pdfs.py`
  - Load configuration
  - Initialize logger
  - Load all sources via document_loader
  - Chunk via semantic_chunker
  - Save chunks to data/processed/chunks/
  - Display statistics (chunks per source, avg chunk size)

- [ ] Run full ingestion
  - Execute script on all 4 sources
  - Monitor for errors
  - Verify output files
  - Check processing time

**Phase 2 Complete:** ✅ All data sources ingested and chunked

---

## Phase 3: Embeddings + Index (Day 3)

### 3.1 Embedding Generator
- [ ] Create `src/embeddings/embedding_generator.py`
  - `__init__()` - Load sentence-transformer model
  - `generate_embeddings()` - Generate embeddings for chunks
  - `batch_encode()` - Process in batches for efficiency
  - `save_embeddings()` - Save to .npy file
  - `load_embeddings()` - Load from disk
  - Returns: numpy array of embeddings

- [ ] Test embedding generation
  - Load model (all-MiniLM-L6-v2)
  - Test on small batch
  - Verify output shape (N chunks × 384 dims)
  - Check speed (chunks per second)

### 3.2 Vector Store Manager
- [ ] Create `src/vector_store/faiss_manager.py`
  - `create_index()` - Create FAISS index
    - Use IndexFlatIP for cosine similarity
    - Normalize vectors
  - `add_embeddings()` - Add embeddings to index
  - `save_index()` - Save to disk
  - `load_index()` - Load from disk
  - `search()` - Search index with query embedding
  - Returns: FAISS index

- [ ] Test FAISS operations
  - Create test index
  - Add test embeddings
  - Save/load index
  - Test search functionality
  - Verify speed (<100ms for query)

### 3.3 Index Building Script
- [ ] Create `scripts/02_build_index.py`
  - Load chunks from disk
  - Generate embeddings for all chunks
  - Create FAISS index
  - Save index + metadata
  - Display statistics (index size, dimension, chunk count)

- [ ] Build production index
  - Run on all ingested chunks
  - Monitor memory usage
  - Verify index file created
  - Test search with sample queries

**Phase 3 Complete:** ✅ Vector index ready for retrieval

---

## Phase 4: Retrieval Pipeline (Day 4)

### 4.1 Retriever
- [ ] Create `src/retrieval/retriever.py`
  - `__init__()` - Load index, chunks, embedding model
  - `retrieve()` - Retrieve top-K chunks
    - Embed query
    - Search FAISS index
    - Return top-K results with scores
  - `retrieve_with_filter()` - Filtered retrieval (optional)
  - Returns: List of retrieved chunks with relevance scores

- [ ] Test retrieval
  - Test with sample queries
  - Verify relevance scores
  - Check metadata preservation
  - Test edge cases (no results, all low scores)

### 4.2 Reranker
- [ ] Create `src/reranking/cross_encoder_reranker.py`
  - `__init__()` - Load cross-encoder model
  - `rerank()` - Rerank chunks
    - Score query-chunk pairs
    - Sort by score
    - Return top-N chunks
  - `batch_rerank()` - Batch processing for efficiency
  - Returns: Re-ranked list of chunks

- [ ] Test reranking
  - Load model (ms-marco-MiniLM-L-6-v2)
  - Test on retrieved chunks
  - Compare before/after rankings
  - Measure latency

### 4.3 Gemini Generator
- [ ] Create `src/generation/gemini_generator.py`
  - `__init__()` - Setup Gemini API
    - Load API key from env
    - Initialize model (gemini-1.5-flash)
  - `build_prompt()` - Construct RAG prompt
    - Include retrieved chunks
    - Add citation instructions
    - Format sources with metadata
  - `generate_answer()` - Generate response
    - Call Gemini API
    - Handle rate limits (retry with backoff)
    - Handle errors gracefully
  - `format_response()` - Format final response
    - Add citations
    - Include source metadata
  - Returns: Formatted answer with sources

- [ ] Test Gemini generation
  - Test API connection
  - Test prompt construction
  - Test answer generation
  - Verify citation formatting
  - Test error handling

### 4.4 RAG Pipeline (Main Orchestrator)
- [ ] Create `src/pipeline/rag_pipeline.py`
  - `__init__()` - Initialize all components
    - Load index, chunks, models
    - Setup retriever, reranker, generator
  - `query()` - End-to-end query processing
    - Retrieve top-K chunks
    - Rerank to top-N
    - Generate answer
    - Format response
  - `batch_query()` - Process multiple queries
  - `get_stats()` - Get pipeline statistics
  - Returns: Answer + sources + metadata

- [ ] Test end-to-end pipeline
  - Test with various query types
  - Test with/without reranking
  - Verify response quality
  - Check latency (<2s per query)
  - Test error handling

**Phase 4 Complete:** ✅ Full RAG pipeline operational

---

## Phase 5: CLI + Testing (Day 5)

### 5.1 Query CLI
- [ ] Create `scripts/03_query_cli.py`
  - Interactive command-line interface
  - Load pipeline on startup
  - Accept user queries
  - Display formatted answers
  - Show sources with metadata
  - Support continuous querying
  - Exit command (/quit, /exit)
  - Help command (/help)

- [ ] Test CLI
  - Test interactive loop
  - Test query formatting
  - Test display of sources
  - Test exit conditions
  - Test with various queries

### 5.2 Evaluation Script
- [ ] Create `scripts/04_eval_system.py`
  - Load evaluation questions (10-20 sample questions)
  - Run queries through pipeline
  - Measure metrics:
    - Retrieval latency
    - Generation latency
    - Total latency
  - Save results to file
  - Display summary statistics

- [ ] Run evaluation
  - Execute evaluation script
  - Review results
  - Identify bottlenecks
  - Log performance metrics

### 5.3 Unit Tests
- [ ] Create `tests/test_pipeline.py`
  - Test document loader
  - Test chunker
  - Test embedding generator
  - Test retriever
  - Test reranker
  - Test pipeline end-to-end

- [ ] Run tests
  - Execute pytest
  - Verify all tests pass
  - Fix any failing tests
  - Achieve >80% code coverage

### 5.4 Documentation
- [ ] Create `README.md`
  - Project overview
  - Installation instructions
  - Usage guide
  - Example queries
  - Troubleshooting
  - API reference (if exposing as module)

- [ ] Create usage examples
  - Example queries with expected outputs
  - Code snippets for common tasks
  - Performance benchmarks

**Phase 5 Complete:** ✅ System tested and documented

---

## Additional Tasks (Optional)

### Performance Optimization
- [ ] Add caching for frequently asked queries
- [ ] Implement batch processing for embeddings
- [ ] Optimize FAISS index parameters
- [ ] Profile and optimize bottlenecks

### Feature Enhancements
- [ ] Add query history
- [ ] Add relevance score threshold
- [ ] Add source filtering (query from specific source)
- [ ] Add query suggestions
- [ ] Export results to file

### Production Readiness
- [ ] Add Docker configuration
- [ ] Add monitoring/logging
- [ ] Add error reporting
- [ ] Add configuration validation
- [ ] Add health checks

---

## Milestones

- [ ] **Milestone 1:** Can load and chunk PDFs (End of Day 2)
- [ ] **Milestone 2:** Vector index built and searchable (End of Day 3)
- [ ] **Milestone 3:** First end-to-end query working (End of Day 4)
- [ ] **Milestone 4:** Interactive CLI ready (End of Day 5)

---

## Notes

- **API Key Needed:** Gemini API key from https://makersuite.google.com/app/apikey
- **Data Sources:**
  - ML.pdf (34 pages)
  - DL.pdf (189 pages)
  - scikit-learn-docs.zip (91MB)
  - PyTorch docs (web)
- **Expected Chunk Count:** ~5,000-10,000 chunks total
- **Expected Index Size:** ~50-100MB
- **Expected Query Latency:** <2 seconds

---

*Last Updated: 2026-05-04*
*Status: Ready to begin Phase 1*
