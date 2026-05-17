"""
Add Aman.ai Primers to RAG System
One-stop script to scrape, process, and integrate Aman.ai content
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders.aman_scraper import AmanScraper
from src.processors.aman_processor import AmanProcessor


def main():
    """Complete pipeline: scrape and process Aman.ai articles"""

    print("="*80)
    print("ADD AMAN.AI PRIMERS TO RAG SYSTEM")
    print("="*80)

    # Step 1: Scrape articles
    print("\n[STEP 1/2] SCRAPING ARTICLES")
    print("-"*80)

    scraper = AmanScraper(output_dir="data/aman_primers")
    scraped = scraper.scrape_all()

    if not scraped:
        print("\n[ERROR] Scraping failed - no articles retrieved")
        return

    # Step 2: Process articles
    print("\n[STEP 2/2] PROCESSING ARTICLES")
    print("-"*80)

    processor = AmanProcessor(
        input_dir="data/aman_primers",
        output_dir="data/processed/aman_primers"
    )

    chunks = processor.process_all_articles()

    if not chunks:
        print("\n[ERROR] Processing failed - no chunks created")
        return

    # Success
    print("\n" + "="*80)
    print("[SUCCESS] AMAN.AI CONTENT SUCCESSFULLY ADDED")
    print("="*80)
    print(f"  Articles scraped: {len(scraped)}")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Index location: data/processed/aman_primers/aman_index.faiss")
    print("\n[OK] Ready to integrate into unified pipeline!")
    print("="*80)


if __name__ == "__main__":
    main()
