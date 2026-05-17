"""Test article extraction from Aman.ai"""

import requests
from bs4 import BeautifulSoup
import re

url = "https://aman.ai/primers/ai/"

response = requests.get(url, timeout=20)
soup = BeautifulSoup(response.content, 'html.parser')

# Get the main content
content_div = soup.find('main') or soup.find('article') or soup.find('body')

if not content_div:
    print("No content found")
else:
    # Get text content
    content_text = content_div.get_text(separator='\n', strip=True)

    # Save to file for inspection
    with open('aman_content.txt', 'w', encoding='utf-8') as f:
        f.write(content_text)

    print(f"Content length: {len(content_text)} characters")
    print(f"Lines: {len(content_text.split(chr(10)))} lines")

    # Look for patterns
    lines = content_text.split('\n')
    article_count = 0

    print("\nFirst 50 lines:")
    print("=" * 80)
    for i, line in enumerate(lines[:50]):
        line = line.strip()
        if line.startswith('##'):
            print(f"HEADER: {line}")
        elif line.startswith('- ') and len(line) > 5:
            article_count += 1
            print(f"ARTICLE {article_count}: {line}")

    print(f"\nTotal articles found in first 50 lines: {article_count}")
    print("Content saved to aman_content.txt")
