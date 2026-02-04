"""Test each scraper individually"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.scrapers.phu_quy_scraper import PhuQuyScraper
from src.scrapers.btmc_scraper import BTMCScraper
from src.scrapers.ancarat_scraper import AncaratScraper

print("=== Testing PhuQuyScraper ===")
pq = PhuQuyScraper()
result = pq.scrape()
print(f"Success: {result.success}, Items: {len(result.items)}")
if result.error:
    print(f"Error: {result.error}")
for item in result.items:
    print(f"  [{item.brand}] {item.product_type}: Buy {item.buy_price:,.0f} | Sell {item.sell_price:,.0f}")

print("\n=== Testing BTMCScraper ===")
btmc = BTMCScraper()
result = btmc.scrape()
print(f"Success: {result.success}, Items: {len(result.items)}")
if result.error:
    print(f"Error: {result.error}")
for item in result.items:
    print(f"  [{item.brand}] {item.product_type}: Buy {item.buy_price:,.0f} | Sell {item.sell_price:,.0f}")

print("\n=== Testing AncaratScraper ===")
anc = AncaratScraper()
result = anc.scrape()
print(f"Success: {result.success}, Items: {len(result.items)}")
if result.error:
    print(f"Error: {result.error}")
for item in result.items:
    print(f"  [{item.brand}] {item.product_type}: Buy {item.buy_price:,.0f} | Sell {item.sell_price:,.0f}")
