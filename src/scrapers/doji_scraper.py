from typing import List
from datetime import datetime
from bs4 import BeautifulSoup
from .base_scraper import BaseScraper
from .models import GoldPriceItem

class DOJIScraper(BaseScraper):
    """Scraper for DOJI Gold Prices."""
    
    def __init__(self):
        super().__init__("DOJI", "http://giavang.doji.vn/")
        
    def parse(self, html_content: str) -> List[GoldPriceItem]:
        items = []
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Find the main price table
            # Note: Selectors might need adjustment if DOJI changes their site
            # Looking for typical table structure
            rows = soup.select('table tr')
            
            updated_at = datetime.now() # Default fallback
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 4:
                    item_name = cols[0].get_text(strip=True)
                    if not item_name or "Loại" in item_name:
                        continue
                        
                    buy_price_text = cols[1].get_text(strip=True).replace('.', '').replace(',', '')
                    sell_price_text = cols[2].get_text(strip=True).replace('.', '').replace(',', '')
                    
                    try:
                        buy_price = float(buy_price_text) * 1000 # DOJI prices in '000 VND
                        sell_price = float(sell_price_text) * 1000
                        
                        items.append(GoldPriceItem(
                            brand="DOJI",
                            product_type=item_name,
                            buy_price=buy_price,
                            sell_price=sell_price,
                            updated_at=updated_at,
                            location="Hà Nội/HCM" # DOJI site usually shows mixed or tabbed info
                        ))
                    except ValueError:
                        continue
                        
        except Exception as e:
            print(f"Error parsing DOJI HTML: {e}")
            
        return items
