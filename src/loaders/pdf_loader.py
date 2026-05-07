"""
PDF Loader for extracting text and detecting chapters from PDF documents
Supports ML.pdf and DL.pdf with chapter detection
"""

import re
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader

from src.utils.logger import LoggerMixin


class PDFLoader(LoggerMixin):
    """
    Load and extract text from PDF files with chapter detection
    """

    def __init__(self):
        """Initialize PDF loader"""
        self.chapter_patterns = [
            r'^Chapter\s+\d+',           # "Chapter 1", "Chapter 2", etc.
            r'^CHAPTER\s+\d+',           # "CHAPTER 1", "CHAPTER 2", etc.
            r'^\d+\.\s+',                # "1. ", "2. ", etc.
            r'^Part\s+\d+',              # "Part 1", "Part 2", etc.
        ]

    def extract_text_from_pdf(self, pdf_path: str | Path) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with page-level metadata

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of pages with text and metadata
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self.logger.info(f"Extracting text from {pdf_path.name}")

        try:
            reader = PdfReader(str(pdf_path))
            pages_data = []

            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()

                    if text and text.strip():
                        pages_data.append({
                            "page_number": page_num + 1,
                            "text": text.strip(),
                            "char_count": len(text)
                        })

                except Exception as e:
                    self.logger.warning(f"Failed to extract page {page_num + 1}: {e}")
                    continue

            self.logger.info(f"Extracted {len(pages_data)} pages from {pdf_path.name}")
            return pages_data

        except Exception as e:
            self.logger.error(f"Failed to read PDF {pdf_path}: {e}")
            raise

    def detect_chapters_toc(self, reader: PdfReader) -> List[Dict[str, Any]]:
        """
        Try to detect chapters using PDF outline/TOC

        Args:
            reader: PyPDF PdfReader object

        Returns:
            List of chapters with page numbers
        """
        chapters = []

        try:
            outline = reader.outline

            if not outline:
                self.logger.debug("No PDF outline found")
                return chapters

            self.logger.info("Found PDF outline, extracting chapters...")

            def extract_from_outline(items, level=0):
                """Recursively extract chapter info from outline"""
                for item in items:
                    if isinstance(item, list):
                        extract_from_outline(item, level + 1)
                    else:
                        # item is a destination object
                        try:
                            title = item.title if hasattr(item, 'title') else str(item)
                            page_num = reader.get_destination_page_number(item) + 1

                            chapters.append({
                                "title": title,
                                "page_number": page_num,
                                "level": level
                            })
                        except:
                            continue

            extract_from_outline(outline)

            self.logger.info(f"Extracted {len(chapters)} chapters from outline")
            return chapters

        except Exception as e:
            self.logger.warning(f"Failed to extract TOC: {e}")
            return []

    def detect_chapters_regex(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect chapters using regex patterns on page text

        Args:
            pages_data: List of pages with text

        Returns:
            List of detected chapters
        """
        chapters = []

        for page in pages_data:
            text = page["text"]
            lines = text.split('\n')

            for line_num, line in enumerate(lines):
                line = line.strip()

                # Skip empty lines or very short lines
                if len(line) < 3:
                    continue

                # Try each pattern
                for pattern in self.chapter_patterns:
                    if re.match(pattern, line, re.IGNORECASE):
                        chapters.append({
                            "title": line,
                            "page_number": page["page_number"],
                            "line_number": line_num,
                            "detection_method": "regex"
                        })
                        break

        self.logger.info(f"Detected {len(chapters)} chapters using regex")
        return chapters

    def detect_chapters_font_size(self, pdf_path: str | Path) -> List[Dict[str, Any]]:
        """
        Detect chapters using font size analysis (future enhancement)

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of detected chapters
        """
        # This is a placeholder for future implementation
        # Would use pdfminer.six or similar to analyze font sizes
        self.logger.debug("Font size detection not implemented, using regex")
        return []

    def extract_chapters(
        self,
        pdf_path: str | Path,
        source_name: str
    ) -> List[Dict[str, Any]]:
        """
        Extract chapters from PDF with hierarchical structure

        Args:
            pdf_path: Path to PDF file
            source_name: Name of the source (e.g., "DL_Textbook")

        Returns:
            List of chapters with text content
        """
        pdf_path = Path(pdf_path)
        self.logger.info(f"Extracting chapters from {pdf_path.name}")

        # Extract all pages
        pages_data = self.extract_text_from_pdf(pdf_path)

        if not pages_data:
            self.logger.warning(f"No text extracted from {pdf_path.name}")
            return []

        # Try to detect chapters using TOC first
        try:
            reader = PdfReader(str(pdf_path))
            chapters_toc = self.detect_chapters_toc(reader)

            if chapters_toc:
                # Use TOC-detected chapters
                chapters = self._build_chapters_from_toc(chapters_toc, pages_data)
            else:
                # Fall back to regex detection
                chapters_regex = self.detect_chapters_regex(pages_data)
                chapters = self._build_chapters_from_regex(chapters_regex, pages_data)

        except Exception as e:
            self.logger.warning(f"TOC detection failed, using regex: {e}")
            chapters_regex = self.detect_chapters_regex(pages_data)
            chapters = self._build_chapters_from_regex(chapters_regex, pages_data)

        # If no chapters detected, create one big chapter
        if not chapters:
            self.logger.warning(f"No chapters detected in {pdf_path.name}, creating single chapter")
            chapters = [{
                "chapter_number": "1",
                "title": "Full Document",
                "page_start": 1,
                "page_end": len(pages_data),
                "text": "\n".join([p["text"] for p in pages_data]),
                "source_name": source_name
            }]

        # Add source name to all chapters
        for chapter in chapters:
            chapter["source_name"] = source_name
            chapter["source_type"] = "pdf"

        self.logger.info(f"Extracted {len(chapters)} chapters from {pdf_path.name}")
        return chapters

    def _build_chapters_from_toc(
        self,
        chapters_toc: List[Dict[str, Any]],
        pages_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build chapters from TOC information

        Args:
            chapters_toc: Chapters from TOC
            pages_data: All pages with text

        Returns:
            List of chapters with text content
        """
        chapters = []

        for i, toc_chapter in enumerate(chapters_toc):
            # Determine page range
            page_start = toc_chapter["page_number"]
            page_end = chapters_toc[i + 1]["page_number"] if i + 1 < len(chapters_toc) else len(pages_data)

            # Extract text for this chapter
            chapter_pages = [p for p in pages_data if page_start <= p["page_number"] < page_end]
            chapter_text = "\n".join([p["text"] for p in chapter_pages])

            chapters.append({
                "chapter_number": str(i + 1),
                "title": toc_chapter["title"],
                "page_start": page_start,
                "page_end": page_end - 1,
                "text": chapter_text,
                "detection_method": "toc"
            })

        return chapters

    def _build_chapters_from_regex(
        self,
        chapters_regex: List[Dict[str, Any]],
        pages_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build chapters from regex-detected headers

        Args:
            chapters_regex: Chapters detected by regex
            pages_data: All pages with text

        Returns:
            List of chapters with text content
        """
        chapters = []

        for i, regex_chapter in enumerate(chapters_regex):
            # Determine page range
            page_start = regex_chapter["page_number"]
            page_end = chapters_regex[i + 1]["page_number"] if i + 1 < len(chapters_regex) else len(pages_data)

            # Extract text for this chapter
            chapter_pages = [p for p in pages_data if page_start <= p["page_number"] < page_end]
            chapter_text = "\n".join([p["text"] for p in chapter_pages])

            chapters.append({
                "chapter_number": str(i + 1),
                "title": regex_chapter["title"],
                "page_start": page_start,
                "page_end": page_end - 1,
                "text": chapter_text,
                "detection_method": "regex"
            })

        return chapters


def load_pdf_with_chapters(pdf_path: str | Path, source_name: str) -> List[Dict[str, Any]]:
    """
    Convenience function to load PDF and extract chapters

    Args:
        pdf_path: Path to PDF file
        source_name: Name of the source

    Returns:
        List of chapters with text and metadata
    """
    loader = PDFLoader()
    return loader.extract_chapters(pdf_path, source_name)


if __name__ == "__main__":
    # Test PDF loader
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_loader.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    source_name = sys.argv[2] if len(sys.argv) > 2 else "Test_Source"

    loader = PDFLoader()
    chapters = loader.extract_chapters(pdf_path, source_name)

    print(f"\nExtracted {len(chapters)} chapters:")
    for i, chapter in enumerate(chapters[:5]):  # Show first 5
        print(f"\nChapter {i + 1}: {chapter['title']}")
        print(f"  Pages: {chapter['page_start']}-{chapter['page_end']}")
        print(f"  Text length: {len(chapter['text'])} chars")
        print(f"  Preview: {chapter['text'][:200]}...")
