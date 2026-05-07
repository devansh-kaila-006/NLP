"""
Main Document Loader - Orchestrates loading from multiple sources
Supports PDF, ZIP, and Web sources
"""

from pathlib import Path
from typing import List, Dict, Any

from src.loaders.pdf_loader import PDFLoader
from src.loaders.zip_loader import ZIPLoader
from src.loaders.web_loader import WebLoader
from src.utils.logger import LoggerMixin
from src.config import PDF_SOURCES


class DocumentLoader(LoggerMixin):
    """
    Main document loader that orchestrates loading from multiple sources
    """

    def __init__(self):
        """Initialize document loader"""
        self.pdf_loader = PDFLoader()
        self.zip_loader = ZIPLoader()
        self.web_loader = WebLoader()

    def load_source(self, source_name: str, source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Load documents from a single source

        Args:
            source_name: Name of the source
            source_config: Configuration dictionary for the source

        Returns:
            List of documents/chapters/sections
        """
        source_type = source_config["type"]
        self.logger.info(f"Loading source: {source_name} (type: {source_type})")

        try:
            if source_type == "pdf":
                return self._load_pdf(source_name, source_config)
            elif source_type == "zip":
                return self._load_zip(source_name, source_config)
            elif source_type == "web":
                return self._load_web(source_name, source_config)
            else:
                self.logger.error(f"Unknown source type: {source_type}")
                return []

        except Exception as e:
            self.logger.error(f"Failed to load {source_name}: {e}")
            return []

    def _load_pdf(self, source_name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load PDF source"""
        pdf_path = config["path"]
        return self.pdf_loader.extract_chapters(pdf_path, source_name)

    def _load_zip(self, source_name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load ZIP source"""
        zip_path = config["path"]
        extract_to = config.get("extract_to", "data/cache/extracted")
        return self.zip_loader.load_zip_documentation(zip_path, extract_to, source_name)

    def _load_web(self, source_name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load web source"""
        url = config["url"]
        cache_dir = config.get("cache_dir", "data/cache/web_docs")
        max_pages = config.get("max_pages", 50)
        return self.web_loader.load_documentation(url, cache_dir, source_name, max_pages)

    def load_all_sources(
        self,
        sources_config: Dict[str, Dict[str, Any]] = None,
        source_filter: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load all configured sources

        Args:
            sources_config: Configuration dictionary (uses PDF_SOURCES if None)
            source_filter: List of source names to load (loads all if None)

        Returns:
            List of all documents from all sources
        """
        if sources_config is None:
            sources_config = PDF_SOURCES

        # Filter sources if requested
        if source_filter:
            sources_config = {
                name: config
                for name, config in sources_config.items()
                if name in source_filter
            }

        self.logger.info(f"Loading {len(sources_config)} sources: {list(sources_config.keys())}")

        all_documents = []
        source_stats = {}

        for source_name, source_config in sources_config.items():
            try:
                documents = self.load_source(source_name, source_config)

                if documents:
                    all_documents.extend(documents)
                    source_stats[source_name] = {
                        'count': len(documents),
                        'type': source_config['type'],
                        'status': 'success'
                    }
                    self.logger.info(f"Loaded {len(documents)} documents from {source_name}")
                else:
                    source_stats[source_name] = {
                        'count': 0,
                        'type': source_config['type'],
                        'status': 'empty'
                    }
                    self.logger.warning(f"No documents loaded from {source_name}")

            except Exception as e:
                source_stats[source_name] = {
                    'count': 0,
                    'type': source_config['type'],
                    'status': 'error',
                    'error': str(e)
                }
                self.logger.error(f"Error loading {source_name}: {e}")

        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("Source Loading Summary:")
        for source_name, stats in source_stats.items():
            self.logger.info(f"  {source_name}: {stats['count']} documents ({stats['status']})")
        self.logger.info(f"Total: {len(all_documents)} documents loaded")
        self.logger.info("=" * 60)

        return all_documents

    def normalize_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize documents to standard format

        Args:
            documents: List of documents from various sources

        Returns:
            Normalized documents
        """
        normalized = []

        for doc in documents:
            # Ensure required fields exist
            if 'text' not in doc or not doc['text']:
                continue

            # Create normalized copy
            norm_doc = {
                'text': doc['text'],
                'source_name': doc.get('source_name', 'Unknown'),
                'source_type': doc.get('source_type', 'unknown'),
                'char_count': len(doc.get('text', '')),
            }

            # Add optional metadata based on source type
            if doc.get('source_type') == 'pdf':
                norm_doc.update({
                    'chapter': doc.get('chapter_number', ''),
                    'title': doc.get('title', ''),
                    'page_start': doc.get('page_start', 0),
                    'page_end': doc.get('page_end', 0),
                })
            elif doc.get('source_type') in ['html', 'web', 'zip']:
                norm_doc.update({
                    'title': doc.get('title', ''),
                    'url': doc.get('url', ''),
                    'file_path': doc.get('file_path', ''),
                    'section': doc.get('level', 'h1'),
                })

            # Generate unique ID
            doc_id = f"{norm_doc['source_name']}_"
            if norm_doc['source_type'] == 'pdf':
                doc_id += f"ch{norm_doc.get('chapter', '0')}"
            else:
                doc_id += f"{hash(norm_doc.get('title', '') or norm_doc.get('url', '')) % 10000:04d}"

            norm_doc['chunk_id'] = doc_id
            normalized.append(norm_doc)

        self.logger.info(f"Normalized {len(normalized)} documents")
        return normalized


def load_all_documents(
    sources_config: Dict[str, Dict[str, Any]] = None,
    source_filter: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to load all documents

    Args:
        sources_config: Configuration dictionary
        source_filter: List of source names to load

    Returns:
        List of normalized documents
    """
    loader = DocumentLoader()
    documents = loader.load_all_sources(sources_config, source_filter)
    return loader.normalize_documents(documents)


if __name__ == "__main__":
    # Test document loader
    import sys

    # Load specific sources if provided
    source_filter = sys.argv[1:] if len(sys.argv) > 1 else None

    loader = DocumentLoader()

    # Load only PDF sources for testing
    if source_filter:
        sources = {k: v for k, v in PDF_SOURCES.items() if k in source_filter}
    else:
        sources = PDF_SOURCES

    documents = loader.load_all_sources(sources)
    normalized = loader.normalize_documents(documents)

    print(f"\nLoaded {len(normalized)} documents:")
    for doc in normalized[:5]:  # Show first 5
        print(f"\n[{doc['source_name']}] {doc.get('title', 'N/A')}")
        print(f"  Type: {doc['source_type']}")
        print(f"  Length: {doc['char_count']} chars")
        print(f"  Preview: {doc['text'][:150]}...")
