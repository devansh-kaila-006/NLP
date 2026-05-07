"""
Web Loader for scraping and parsing HTML documentation from URLs
Supports PyTorch documentation from docs.pytorch.org
"""

import time
import re
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

from src.utils.logger import LoggerMixin
from src.utils.helpers import ensure_dir


class WebLoader(LoggerMixin):
    """
    Load and parse HTML documentation from web URLs
    """

    def __init__(self, delay: float = 1.0):
        """
        Initialize web loader

        Args:
            delay: Delay between requests in seconds (to be respectful)
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def scrape_url(self, url: str) -> str:
        """
        Scrape content from URL

        Args:
            url: URL to scrape

        Returns:
            HTML content
        """
        self.logger.info(f"Scraping {url}")

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Respectful delay
            time.sleep(self.delay)

            return response.text

        except Exception as e:
            self.logger.error(f"Failed to scrape {url}: {e}")
            return ""

    def parse_html_content(self, html: str, url: str) -> Dict[str, Any]:
        """
        Parse HTML content and extract documentation

        Args:
            html: HTML content
            url: Source URL

        Returns:
            Parsed content dictionary
        """
        soup = BeautifulSoup(html, 'lxml')

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()

        # Extract title
        title = ""
        if soup.title:
            title = soup.title.get_text().strip()

        # Extract main content
        main_content = (
            soup.find('main') or
            soup.find('article') or
            soup.find('div', class_='content') or
            soup.find('div', class_='document') or
            soup.find('div', class_='container') or
            soup.find('body')
        )

        if main_content:
            # Get text content
            text = main_content.get_text(separator='\n', strip=True)

            # Clean up text
            lines = [line.strip() for line in text.split('\n')]
            lines = [line for line in lines if line and len(line) > 3]
            text = '\n'.join(lines)
        else:
            text = ""

        # Extract section headers
        sections = []
        for header in main_content.find_all(['h1', 'h2', 'h3', 'h4']) if main_content else []:
            section_text = header.get_text().strip()
            if section_text and len(section_text) < 200:  # Reasonable header length
                sections.append({
                    'level': header.name,
                    'title': section_text
                })

        # Extract code snippets
        code_snippets = []
        for code in soup.find_all(['code', 'pre']):
            code_text = code.get_text().strip()
            if len(code_text) > 20 and len(code_text) < 2000:  # Reasonable snippet length
                code_snippets.append(code_text)

        return {
            'title': title,
            'text': text,
            'sections': sections,
            'code_snippets': code_snippets,
            'url': url,
            'char_count': len(text)
        }

    def find_documentation_links(self, html: str, base_url: str) -> List[str]:
        """
        Find internal documentation links

        Args:
            html: HTML content
            base_url: Base URL for resolving relative links

        Returns:
            List of documentation URLs
        """
        soup = BeautifulSoup(html, 'lxml')
        links = []

        base_domain = urlparse(base_url).netloc

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(base_url, href)

            # Only include links from the same domain
            if urlparse(full_url).netloc == base_domain:
                # Skip anchors, mailto, etc.
                if not full_url.startswith(('http://', 'https://')):
                    continue

                # Skip certain patterns
                if any(skip in full_url.lower() for skip in ['__', '.pdf', '.zip', 'github']):
                    continue

                links.append(full_url)

        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)

        return unique_links

    def load_documentation(
        self,
        start_url: str,
        cache_dir: str | Path,
        source_name: str,
        max_pages: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Load documentation from starting URL, following internal links

        Args:
            start_url: Starting URL
            cache_dir: Directory to cache HTML content
            max_pages: Maximum number of pages to scrape

        Returns:
            List of documentation sections
        """
        cache_dir = Path(cache_dir)
        ensure_dir(cache_dir)

        self.logger.info(f"Loading documentation from {start_url}")

        # Scrape start page
        html = self.scrape_url(start_url)

        if not html:
            self.logger.error(f"Failed to scrape start URL: {start_url}")
            return []

        # Find documentation links
        links = self.find_documentation_links(html, start_url)
        self.logger.info(f"Found {len(links)} documentation links")

        # Limit links
        links = links[:max_pages]

        # Scrape all pages
        parsed_docs = []
        for i, link in enumerate(links):
            self.logger.info(f"Scraping {i + 1}/{len(links)}: {link}")

            html = self.scrape_url(link)
            if html:
                parsed = self.parse_html_content(html, link)
                if parsed['text']:
                    parsed_docs.append(parsed)

        self.logger.info(f"Scraped {len(parsed_docs)} pages")

        # Convert to sections format
        sections = []
        for doc in parsed_docs:
            # Create sections from document
            if doc['sections']:
                for section in doc['sections'][:3]:  # Limit to top 3 sections per page
                    sections.append({
                        'title': section['title'],
                        'level': section['level'],
                        'text': doc['text'],
                        'url': doc['url'],
                        'source_name': source_name,
                        'source_type': 'html',
                        'char_count': len(doc['text'])
                    })
            else:
                sections.append({
                    'title': doc['title'] if doc['title'] else doc['url'],
                    'level': 'h1',
                    'text': doc['text'],
                    'url': doc['url'],
                    'source_name': source_name,
                    'source_type': 'html',
                    'char_count': len(doc['text'])
                })

        self.logger.info(f"Loaded {len(sections)} sections from {start_url}")
        return sections


def load_web_documentation(
    url: str,
    cache_dir: str | Path,
    source_name: str,
    max_pages: int = 50
) -> List[Dict[str, Any]]:
    """
    Convenience function to load web documentation

    Args:
        url: Starting URL
        cache_dir: Directory to cache content
        source_name: Name of the source
        max_pages: Maximum pages to scrape

    Returns:
        List of documentation sections
    """
    loader = WebLoader()
    return loader.load_documentation(url, cache_dir, source_name, max_pages)


if __name__ == "__main__":
    # Test web loader
    import sys

    if len(sys.argv) < 2:
        print("Usage: python web_loader.py <url> <cache_dir> <source_name>")
        sys.exit(1)

    url = sys.argv[1]
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else "data/cache/pytorch_docs"
    source_name = sys.argv[3] if len(sys.argv) > 3 else "PyTorch_Docs"

    loader = WebLoader()
    sections = loader.load_documentation(url, cache_dir, source_name, max_pages=10)

    print(f"\nLoaded {len(sections)} sections:")
    for i, section in enumerate(sections[:5]):  # Show first 5
        print(f"\nSection {i + 1}: {section['title']}")
        print(f"  URL: {section['url']}")
        print(f"  Text length: {section['char_count']} chars")
        print(f"  Preview: {section['text'][:200]}...")
