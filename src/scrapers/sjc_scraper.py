from typing import List
import xml.etree.ElementTree as ET
from datetime import datetime
from .base_scraper import BaseScraper
from .models import GoldPriceItem

class SJCScraper(BaseScraper):
    """Scraper for SJC Gold Prices using official XML feed."""
    
    def __init__(self):
        super().__init__("SJC", "https://sjc.com.vn/xml/tygiavang.xml")
        
    def parse(self, content: str) -> List[GoldPriceItem]:
        items = []
        try:
            root = ET.fromstring(content)
            
            # Extract timestamp from ratelist updated attribute
            updated_str = root.find('ratelist').get('updated')
            # Format usually: "02/02/2026 09:00:00 AM" or similar
            try:
                updated_at = datetime.strptime(updated_str, "%I:%M:%S %p %d/%m/%Y")
            except:
                try:
                    updated_at = datetime.strptime(updated_str, "%d/%m/%Y %I:%M:%S %p")
                except:
                    updated_at = datetime.now()
            
            for city in root.findall('.//city'):
                city_name = city.get('name')
                
                for item in city.findall('item'):
                    items.append(GoldPriceItem(
                        brand="SJC",
                        product_type=item.get('type') + " - " + city_name,
                        buy_price=float(item.get('buy').replace(',', '')) * 1000, # SJC prices are usually in '000 VND
                        sell_price=float(item.get('sell').replace(',', '')) * 1000,
                        updated_at=updated_at,
                        location=city_name
                    ))
                    
        except Exception as e:
            print(f"Error parsing SJC XML: {e}")
            
        return items
