from tools.web_research import search_web, scrape_web


print("=== SEARCH ===")
results = search_web("site:firecrawl.dev Firecrawl", limit=3)

for result in results:
    print(f"{result.position}. {result.title}")
    print(f"   {result.url}")
    print(f"   {result.description}")
    print()


print("=== SCRAPE ===")
page = scrape_web("https://example.com")

print("Source:", page.source)
print("URL:", page.url)
print("Title:", page.title)
print("Content:", page.markdown[:1000])
