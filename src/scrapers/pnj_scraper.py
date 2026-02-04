from typing import List
from datetime import datetime
import requests
import json
from .base_scraper import BaseScraper
from .models import GoldPriceItem

class PNJScraper(BaseScraper):
    """Scraper for PNJ Gold Prices using internal API."""
    
    def __init__(self):
        # PNJ often loads data from this JSON endpoint
        # Backup: https://www.pnj.com.vn/blog/gia-vang/ (HTML)
        super().__init__("PNJ", "https://cdn.pnj.io/images/giavang/net_gold.json")
        
    def fetch_page(self) -> str:
        # Override to fetch JSON directly
        try:
            response = requests.get(self.base_url, headers=self.headers, timeout=10)
            return response.text
        except Exception as e:
            print(f"Error fetching PNJ API: {e}")
            return None

    def parse(self, content: str) -> List[GoldPriceItem]:
        items = []
        try:
            data = json.loads(content)
            
            # PNJ JSON structure usually contains nested items
            # Example structure assumption
            for item in data:
                # Assuming item structure: {"name": "...", "buy": "...", "sell": "...", "update": "..."}
                # Need to adapt to actual PNJ structure
                
                name = item.get('name') or item.get('typeName')
                if not name: continue
                
                buy = item.get('buy')
                sell = item.get('sell')
                
                if not buy or not sell: continue
                
                # Clean prices (remove commas/dots)
                buy = float(str(buy).replace(',', '').replace('.', ''))
                sell = float(str(sell).replace(',', '').replace('.', ''))
                
                updated_at = datetime.now() # Use current time as fallback
                
                items.append(GoldPriceItem(
                    brand="PNJ",
                    product_type=name,
                    buy_price=buy, # PNJ JSON usually is in VND directly or '000? Need to check
                    sell_price=sell,
                    updated_at=updated_at,
                    location="Toàn quốc"
                ))
                    
        except Exception as e:
            print(f"Error parsing PNJ JSON: {e}")
            
        return items
