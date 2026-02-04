from src.scrapers.webgia_scraper import WebGiaScraper
import json
from datetime import datetime

# Helper to serialze datetime
def json_serial(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError ("Type not serializable")

def verify():
    print("Initializing WebGiaScraper...")
    scraper = WebGiaScraper()
    
    print("Scraping...")
    result = scraper.scrape()
    
    if result.success:
        print(f"✅ Success! Found {len(result.items)} items.")
        if len(result.items) > 0:
            print("Sample item:")
            print(json.dumps(result.items[0].__dict__, default=json_serial, indent=2, ensure_ascii=False))
            
            # Print first 5 items
            print("\nFirst 5 items:")
            for item in result.items[:5]:
                print(f"- {item.brand} ({item.location}): Buy {item.buy_price:,.0f} - Sell {item.sell_price:,.0f}")
    else:
        print(f"❌ Failed: {result.error}")

if __name__ == "__main__":
    verify()
