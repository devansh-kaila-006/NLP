"""
Aman.ai Primer Web Scraper
Scrapes and stores AI primer articles locally
"""

import time
import requests
from pathlib import Path
from typing import List, Dict
import re
from bs4 import BeautifulSoup


class AmanScraper:
    """Scraper for Aman.ai AI primers"""

    def __init__(self, output_dir: str = "data/aman_primers"):
        """
        Initialize scraper

        Args:
            output_dir: Directory to save scraped content
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://aman.ai"
        self.primers_url = f"{self.base_url}/primers/ai/"
        self.delay = 2  # Respectful delay between requests

    def get_article_links(self) -> List[Dict[str, str]]:
        """
        Get all article links from the primers page

        Returns:
            List of dicts with title, url, and category
        """
        articles = []

        try:
            response = requests.get(self.primers_url, timeout=20)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Get the main content
            content_div = soup.find('main') or soup.find('article') or soup.find('body')

            if not content_div:
                print("No content found")
                return []

            # Get text content and parse
            content_text = content_div.get_text(separator='\n', strip=True)

            # Known categories and patterns
            category_indicators = [
                'Model Architecture', 'Data Foundations', 'NLP/LLMs/Agents',
                'Vision', 'Speech', 'Multimodal AI/VLMs', 'Offline/Online Evaluation',
                'MLOps', 'On-Device AI', 'Project Planning, Scheduling, Execution',
                'Models', 'Miscellaneous', 'Hyperparameters', 'Practice', 'Overview'
            ]

            # Skip patterns (not articles)
            skip_patterns = [
                'Here\'s', 'Overview', 'Model Architecture', 'Data Foundations',
                'NLP/LLMs/Agents', 'Vision', 'Speech', 'Multimodal AI/VLMs',
                'Offline/Online Evaluation', 'MLOps', 'On-Device AI',
                'Project Planning, Scheduling, Execution', 'Models',
                'Miscellaneous', 'Hyperparameters', 'Practice'
            ]

            current_category = "General"
            lines = content_text.split('\n')

            for line in lines:
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Check if it's a category
                if line in category_indicators:
                    current_category = line
                    continue

                # Skip if it's a description or known non-article
                if line.startswith('Here') or any(skip in line for skip in skip_patterns):
                    continue

                # Check if it looks like an article title (no special chars, reasonable length)
                if len(line) > 3 and len(line) < 100 and not line.startswith('-'):
                    # Skip single words that might be noise
                    if len(line.split()) < 2:
                        continue

                    # Convert title to URL format
                    url_title = self._title_to_url(line)

                    articles.append({
                        'title': line,
                        'url': f"{self.base_url}/primers/ai/{url_title}",
                        'path': f"/primers/ai/{url_title}",
                        'category': current_category
                    })

            print(f"Found {len(articles)} articles")
            return articles

        except Exception as e:
            print(f"Error fetching article links: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _title_to_url(self, title: str) -> str:
        """Convert article title to URL format"""
        # Convert to lowercase
        url_title = title.lower()

        # Replace spaces and special characters with hyphens
        url_title = re.sub(r'[^\w\s-]', '', url_title)
        url_title = re.sub(r'[-\s]+', '-', url_title)

        return url_title

    def _categorize_article(self, path: str) -> str:
        """Categorize article based on URL path"""
        path_lower = path.lower()

        if any(x in path_lower for x in ['transformer', 'attention', 'bert', 'gpt', 'llama', 'llm']):
            return 'NLP/LLMs'
        elif any(x in path_lower for x in ['vision', 'image', 'vit', 'cnn', 'convolution']):
            return 'Vision'
        elif any(x in path_lower for x in ['training', 'gradient', 'optimization', 'backprop']):
            return 'Training'
        elif any(x in path_lower for x in ['data', 'sampling', 'preprocessing']):
            return 'Data'
        elif any(x in path_lower for x in ['eval', 'benchmark', 'metric']):
            return 'Evaluation'
        else:
            return 'General'

    def scrape_article(self, article: Dict[str, str]) -> Dict[str, str]:
        """
        Scrape a single article

        Args:
            article: Article dict with url and metadata

        Returns:
            Article dict with content added
        """
        try:
            print(f"Scraping: {article['title']}")
            response = requests.get(article['url'], timeout=20)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract main content
            # Usually in <main> or <article> or specific div classes
            content_div = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|post|article'))

            if not content_div:
                # Fallback to body
                content_div = soup.find('body')

            # Extract text content
            content = self._extract_text(content_div)

            # Clean and structure content
            article['content'] = content
            article['word_count'] = len(content.split())

            # Save to file
            self._save_article(article)

            print(f"  [OK] Saved ({article['word_count']} words)")
            time.sleep(self.delay)  # Respectful delay

            return article

        except Exception as e:
            print(f"  [ERROR] Error: {e}")
            article['error'] = str(e)
            return article

    def _extract_text(self, element) -> str:
        """Extract clean text from HTML element"""
        if not element:
            return ""

        # Remove script and style elements
        for script in element(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text
        text = element.get_text(separator='\n', strip=True)

        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()

        return text

    def _save_article(self, article: Dict[str, str]):
        """Save article to local file"""
        # Create filename from title
        safe_title = re.sub(r'[^\w\s-]', '', article['title'])
        safe_title = re.sub(r'[-\s]+', '-', safe_title)
        filename = f"{safe_title}.md"

        # Create category directory
        category_dir = self.output_dir / article['category']
        category_dir.mkdir(parents=True, exist_ok=True)

        filepath = category_dir / filename

        # Write markdown file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {article['title']}\n\n")
            f.write(f"**Source**: {article['url']}\n")
            f.write(f"**Category**: {article['category']}\n")
            f.write(f"**Word Count**: {article['word_count']}\n\n")
            f.write("---\n\n")
            f.write(article['content'])

    def scrape_all(self, max_articles: int = None) -> List[Dict[str, str]]:
        """
        Scrape all articles

        Args:
            max_articles: Maximum number of articles to scrape (None = all)

        Returns:
            List of scraped articles
        """
        print("="*80)
        print("AMAN.AI PRIMER SCRAPER")
        print("="*80)

        # Get article links
        articles = self.get_article_links()

        if not articles:
            print("No articles found!")
            return []

        # Limit if specified
        if max_articles:
            articles = articles[:max_articles]
            print(f"Limiting to {max_articles} articles")

        print(f"\nScraping {len(articles)} articles...")
        print("="*80)

        # Scrape articles
        scraped = []
        for i, article in enumerate(articles, 1):
            print(f"\n[{i}/{len(articles)}]", end=" ")
            result = self.scrape_article(article)
            scraped.append(result)

        # Summary
        successful = sum(1 for a in scraped if 'content' in a)
        failed = len(scraped) - successful

        print("\n" + "="*80)
        print("SCRAPING COMPLETE")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total: {len(scraped)}")
        print(f"  Output: {self.output_dir}")
        print("="*80)

        return scraped


def main():
    """Main scraping function"""
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Aman.ai AI primers")
    parser.add_argument('--max', type=int, default=None, help='Maximum articles to scrape')
    parser.add_argument('--output', type=str, default='data/aman_primers', help='Output directory')

    args = parser.parse_args()

    scraper = AmanScraper(output_dir=args.output)
    scraper.scrape_all(max_articles=args.max)


if __name__ == "__main__":
    main()
