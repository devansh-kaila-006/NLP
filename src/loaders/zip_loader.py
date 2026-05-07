"""
ZIP Loader for extracting and parsing HTML documentation from ZIP archives
Supports scikit-learn-docs.zip
"""

import zipfile
from pathlib import Path
from typing import List, Dict, Any
from bs4 import BeautifulSoup

from src.utils.logger import LoggerMixin
from src.utils.helpers import ensure_dir


class ZIPLoader(LoggerMixin):
    """
    Load and parse HTML documentation from ZIP archives
    """

    def __init__(self):
        """Initialize ZIP loader"""
        self.supported_extensions = ['.html', '.htm']

    def extract_zip(self, zip_path: str | Path, extract_to: str | Path) -> Path:
        """
        Extract ZIP archive to directory

        Args:
            zip_path: Path to ZIP file
            extract_to: Directory to extract to

        Returns:
            Path to extraction directory
        """
        zip_path = Path(zip_path)
        extract_to = Path(extract_to)

        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")

        self.logger.info(f"Extracting {zip_path.name} to {extract_to}")

        # Create extraction directory
        ensure_dir(extract_to)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

            self.logger.info(f"Extracted {len(zip_ref.namelist())} files")
            return extract_to

        except Exception as e:
            self.logger.error(f"Failed to extract ZIP: {e}")
            raise

    def list_html_files(self, extract_dir: str | Path) -> List[Path]:
        """
        List all HTML files in extracted directory

        Args:
            extract_dir: Path to extracted directory

        Returns:
            List of HTML file paths
        """
        extract_dir = Path(extract_dir)

        if not extract_dir.exists():
            raise FileNotFoundError(f"Extracted directory not found: {extract_dir}")

        html_files = []

        for ext in self.supported_extensions:
            html_files.extend(extract_dir.rglob(f"*{ext}"))

        self.logger.info(f"Found {len(html_files)} HTML files")
        return html_files

    def parse_html_file(self, html_path: str | Path) -> Dict[str, Any]:
        """
        Parse HTML file and extract text content

        Args:
            html_path: Path to HTML file

        Returns:
            Dictionary with extracted content
        """
        html_path = Path(html_path)

        try:
            with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, 'lxml')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Extract title
            title = ""
            if soup.title:
                title = soup.title.get_text().strip()

            # Extract main content
            # Try different content containers
            main_content = (
                soup.find('main') or
                soup.find('article') or
                soup.find('div', class_='content') or
                soup.find('div', class_='documentation') or
                soup.body
            )

            if main_content:
                # Get text content
                text = main_content.get_text(separator='\n', strip=True)

                # Clean up text
                lines = [line.strip() for line in text.split('\n')]
                lines = [line for line in lines if line]
                text = '\n'.join(lines)
            else:
                text = ""

            # Extract section headers
            sections = []
            for header in soup.find_all(['h1', 'h2', 'h3', 'h4']):
                section_text = header.get_text().strip()
                if section_text:
                    sections.append({
                        'level': header.name,
                        'title': section_text
                    })

            # Extract code snippets
            code_snippets = []
            for code in soup.find_all(['code', 'pre']):
                code_text = code.get_text().strip()
                if len(code_text) > 20:  # Only substantial snippets
                    code_snippets.append(code_text)

            return {
                'title': title,
                'text': text,
                'sections': sections,
                'code_snippets': code_snippets,
                'file_path': str(html_path.relative_to(html_path.parent.parent.parent)),
                'char_count': len(text)
            }

        except Exception as e:
            self.logger.warning(f"Failed to parse {html_path}: {e}")
            return {
                'title': '',
                'text': '',
                'sections': [],
                'code_snippets': [],
                'file_path': str(html_path),
                'char_count': 0
            }

    def detect_sections(self, parsed_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect documentation sections from parsed HTML files

        Args:
            parsed_docs: List of parsed HTML documents

        Returns:
            List of documentation sections
        """
        sections = []

        for doc in parsed_docs:
            if not doc['text']:
                continue

            # Use sections from HTML if available
            if doc['sections']:
                # Create sections based on HTML structure
                for section in doc['sections']:
                    sections.append({
                        'title': section['title'],
                        'level': section['level'],
                        'text': doc['text'],
                        'file_path': doc['file_path'],
                        'source_type': 'html'
                    })
            else:
                # Use title as section
                sections.append({
                    'title': doc['title'] if doc['title'] else doc['file_path'],
                    'level': 'h1',
                    'text': doc['text'],
                    'file_path': doc['file_path'],
                    'source_type': 'html'
                })

        self.logger.info(f"Detected {len(sections)} sections")
        return sections

    def load_zip_documentation(
        self,
        zip_path: str | Path,
        extract_to: str | Path,
        source_name: str
    ) -> List[Dict[str, Any]]:
        """
        Load and parse documentation from ZIP archive

        Args:
            zip_path: Path to ZIP file
            extract_to: Directory to extract to
            source_name: Name of the source

        Returns:
            List of documentation sections
        """
        zip_path = Path(zip_path)
        extract_to = Path(extract_to)

        self.logger.info(f"Loading documentation from {zip_path.name}")

        # Extract ZIP
        if not extract_to.exists():
            extract_dir = self.extract_zip(zip_path, extract_to)
        else:
            self.logger.info(f"Using existing extracted directory: {extract_to}")
            extract_dir = extract_to

        # List HTML files
        html_files = self.list_html_files(extract_dir)

        if not html_files:
            self.logger.warning(f"No HTML files found in {zip_path.name}")
            return []

        # Parse HTML files (limit to first 100 for testing)
        parsed_docs = []
        for i, html_file in enumerate(html_files[:100]):
            if i % 10 == 0:
                self.logger.info(f"Parsing {i}/{len(html_files)} files...")

            parsed = self.parse_html_file(html_file)
            if parsed['text']:
                parsed_docs.append(parsed)

        self.logger.info(f"Parsed {len(parsed_docs)} HTML files")

        # Detect sections
        sections = self.detect_sections(parsed_docs)

        # Add source name
        for section in sections:
            section['source_name'] = source_name
            section['source_type'] = 'html'

        self.logger.info(f"Loaded {len(sections)} sections from {zip_path.name}")
        return sections


def load_zip_with_docs(
    zip_path: str | Path,
    extract_to: str | Path,
    source_name: str
) -> List[Dict[str, Any]]:
    """
    Convenience function to load ZIP documentation

    Args:
        zip_path: Path to ZIP file
        extract_to: Directory to extract to
        source_name: Name of the source

    Returns:
        List of documentation sections
    """
    loader = ZIPLoader()
    return loader.load_zip_documentation(zip_path, extract_to, source_name)


if __name__ == "__main__":
    # Test ZIP loader
    import sys

    if len(sys.argv) < 2:
        print("Usage: python zip_loader.py <zip_path> <extract_to> <source_name>")
        sys.exit(1)

    zip_path = sys.argv[1]
    extract_to = sys.argv[2] if len(sys.argv) > 2 else "data/cache/sklearn_docs"
    source_name = sys.argv[3] if len(sys.argv) > 3 else "Scikit_learn_Docs"

    loader = ZIPLoader()
    sections = loader.load_zip_documentation(zip_path, extract_to, source_name)

    print(f"\nLoaded {len(sections)} sections:")
    for i, section in enumerate(sections[:5]):  # Show first 5
        print(f"\nSection {i + 1}: {section['title']}")
        print(f"  File: {section['file_path']}")
        print(f"  Text length: {section['char_count']} chars")
        print(f"  Preview: {section['text'][:200]}...")
