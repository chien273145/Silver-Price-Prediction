from .base_scraper import BaseScraper
from .models import GoldPriceItem, ScraperResult
from bs4 import BeautifulSoup
import requests
from datetime import datetime
import re

class BTMCScraper(BaseScraper):
    def __init__(self):
        super().__init__("BTMC", "https://btmc.vn/")

    def parse(self, html_content):
        return []

    def scrape(self) -> ScraperResult:
        try:
            # BTMC often loads data dynamically or in a specific block. 
            # We try the main page or known sub-page.
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Alternative URL seen in search results or common patterns
            url = "https://btmc.vn/gia-vang-hom-nay.html" 
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                # Try root if specific page fails
                response = requests.get(self.base_url, headers=headers, timeout=10)

            soup = BeautifulSoup(response.content, 'html.parser')
            items = []

            # General logic: Find table rows with "Bạc" or "Ag"
            # BTMC tables often have clear classes like 'table-price'
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cols = row.find_all(['td', 'th'])
                    text_content = row.get_text().strip().lower()
                    
                    if 'bạc' in text_content or 'ag' in text_content:
                        # Likely a silver row
                        # Extraction logic depends on effective column layout
                        # Usually: Name | Buy | Sell
                        if len(cols) >= 3:
                            name = cols[0].get_text(strip=True)
                            
                            # Clean price strings
                            raw_buy = cols[1].get_text(strip=True).replace(',', '').replace('.', '')
                            raw_sell = cols[2].get_text(strip=True).replace(',', '').replace('.', '')
                            
                            # Regex to extract numbers
                            buy_match = re.search(r'\d+', raw_buy)
                            sell_match = re.search(r'\d+', raw_sell)
                            
                            if buy_match and sell_match:
                                buy_price = float(buy_match.group())
                                sell_price = float(sell_match.group())
                                
                                # BTMC prices might be in 'thousands' or 'millions'
                                # Heuristic: Silver price ~ 100k - 1M VND per tael/liang
                                # If price < 10000, likely x1000 required
                                if buy_price < 100000:
                                    buy_price *= 1000
                                    sell_price *= 1000
                                    
                                items.append(GoldPriceItem(
                                    brand="BTMC",
                                    product_type=name,
                                    buy_price=buy_price,
                                    sell_price=sell_price,
                                    updated_at=datetime.now(),
                                    location="Hà Nội"
                                ))

            if not items:
                # Fallback: Scrape from a known reliable aggregator via api/html if official fails?
                # For now, return empty lists so we don't break the app
                return ScraperResult(success=True, source=self.source_name, items=[])

            return ScraperResult(success=True, source=self.source_name, items=items)

        except Exception as e:
            return ScraperResult(success=False, source=self.source_name, error=str(e), items=[])
