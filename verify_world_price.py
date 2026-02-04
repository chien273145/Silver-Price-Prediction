import sys
import os
import asyncio
from datetime import datetime

# Adjust path to find src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.scrapers.world_price_scraper import WorldPriceScraper
import yfinance as yf

async def verify():
    print("Initializing WorldPriceScraper...")
    scraper = WorldPriceScraper()
    
    # Debug Raw Values logic from scraper
    print("\n--- DEBUGGING RAW VALUES ---")
    tickers = yf.Tickers("GC=F SI=F VND=X")
    
    try:
        gold_fast = tickers.tickers['GC=F'].fast_info.last_price
        silver_fast = tickers.tickers['SI=F'].fast_info.last_price
        vnd_fast = tickers.tickers['VND=X'].fast_info.last_price
        
        print(f"RAW fast_info - Gold (GC=F): {gold_fast}")
        print(f"RAW fast_info - Silver (SI=F): {silver_fast}")
        print(f"RAW fast_info - USDVND (VND=X): {vnd_fast}")
    except Exception as e:
        print(f"Fast info error: {e}")

    try:
        gold_hist = tickers.tickers['GC=F'].history(period="1d")['Close'].iloc[-1]
        print(f"RAW history - Gold (GC=F): {gold_hist}")
        silver_hist = tickers.tickers['SI=F'].history(period="1d")['Close'].iloc[-1]
        print(f"RAW history - Silver (SI=F): {silver_hist}")
        vnd_hist = tickers.tickers['VND=X'].history(period="1d")['Close'].iloc[-1]
        print(f"RAW history - USDVND (VND=X): {vnd_hist}")
    except Exception as e:
        print(f"History error: {e}")

    print("\n--- RUNNING SCRAPE ---")
    result = scraper.scrape()
    
    if result.success:
        print(f"Success! Source: {result.source}")
        for item in result.items:
            print(f"- {item.product_type}: {item.sell_price:,.0f} VND/Lượng (Brand: {item.brand})")
            print(f"  (Buy: {item.buy_price:,.0f})")
    else:
        print(f"Failed: {result.error}")

if __name__ == "__main__":
    asyncio.run(verify())
