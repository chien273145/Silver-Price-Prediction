import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.scrapers.service import PriceService

async def main():
    print("Testing PriceService with New Scrapers...")
    service = PriceService()
    
    # Force fresh fetch
    service.cache = {}
    
    results = await service.fetch_all()
    
    print(f"\nTotal Items: {len(results['items'])}")
    print(f"Sources Status:")
    for src in results['sources']:
        print(f" - {src['name']}: {src['status']} ({src.get('count', 0)} items)")
        if 'error' in src:
            print(f"   Error: {src['error']}")
            
    print("\n--- Items ---")
    for item in results['items']:
        print(f"[{item.brand}] {item.product_type} - Buy: {item.buy_price:,.0f} | Sell: {item.sell_price:,.0f}")

if __name__ == "__main__":
    asyncio.run(main())
