"""Debug Aman.ai scraper to see HTML structure"""

import requests
from bs4 import BeautifulSoup

url = "https://aman.ai/primers/ai/"

response = requests.get(url, timeout=20)
soup = BeautifulSoup(response.content, 'html.parser')

print("Title:", soup.title.string if soup.title else "No title")
print("\n" + "="*80)

# Find all links
print("ALL LINKS:")
print("="*80)

links_found = []
for link in soup.find_all('a', href=True):
    href = link['href']
    title = link.get_text(strip=True)

    if '/primers/ai/' in href:
        links_found.append((title, href))
        print(f"Title: '{title}'")
        print(f"URL: {href}")
        print("-" * 40)

print(f"\nTotal links with '/primers/ai/': {len(links_found)}")
