from typing import List
from datetime import datetime
from bs4 import BeautifulSoup
import re
from .base_scraper import BaseScraper
from .models import GoldPriceItem

class WebGiaScraper(BaseScraper):
    """Scraper for WebGia.com (Aggregator)."""
    
    def __init__(self):
        super().__init__("WebGia", "https://webgia.com/gia-vang/")
        
    def decode_price(self, code: str) -> float:
        """Port of webgia's gm(r) javascript function."""
        if not code:
            return 0.0
        try:
            # Remove uppercase letters
            clean_code = re.sub(r'[A-Z]', '', code)
            
            # Hex decode pairs
            chars = []
            for i in range(0, len(clean_code) - 1, 2):
                hex_pair = clean_code[i:i+2]
                chars.append(chr(int(hex_pair, 16)))
            
            price_str = "".join(chars)
            # Remove dots and convert to float
            return float(price_str.replace('.', ''))
        except Exception:
            return 0.0

    def parse(self, html_content: str) -> List[GoldPriceItem]:
        items = []
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            table = soup.find('table', class_='table-radius')
            if not table:
                return []
            
            rows = table.select('tbody tr')
            current_location = "TP.HCM" # Default fallback
            
            for row in rows:
                # Handle rowspan for location
                location_th = row.find('th')
                if location_th and not location_th.get('colspan'): 
                    # If it's the first cell and a th, it's likely the location
                    # But need to check if it's the brand or location
                    # Structure: <tr><th>Location</th><td>Brand</td>...</tr>
                    # OR: <tr><td>Brand</td>...</tr> (if location rowspanned)
                    
                    text = location_th.get_text(strip=True)
                    # Simple heuristic: Locations are usually longer or specific names
                    # But easiest is just to check previous logic.
                    # Actually, the HTML shows <th rowspan="5">TP.Hồ Chí Minh</th>
                    if location_th.has_attr('rowspan') or location_th.find_next_sibling('td'):
                         current_location = text
                
                # Brand is usually in a <strong> tag inside a <td>
                brand_node = row.select_one('td a strong')
                if not brand_node:
                    continue
                    
                brand = brand_node.get_text(strip=True)
                
                # Prices can be in tds with 'nb' attribute OR plain text
                price_cells = row.select('td.text-right')
                if len(price_cells) >= 2:
                    # 1. Try 'nb' attribute (obfuscated)
                    buy_code = price_cells[0].get('nb')
                    sell_code = price_cells[1].get('nb')
                    
                    buy_price = 0.0
                    sell_price = 0.0
                    
                    if buy_code:
                        buy_price = self.decode_price(buy_code)
                    else:
                        # 2. Try plain text
                        try:
                            t = price_cells[0].get_text(strip=True).replace('.', '').replace(',', '')
                            if t and re.match(r'^\d+$', t):
                                buy_price = float(t)
                        except: pass

                    if sell_code:
                        sell_price = self.decode_price(sell_code)
                    else:
                        try:
                            t = price_cells[1].get_text(strip=True).replace('.', '').replace(',', '')
                            if t and re.match(r'^\d+$', t):
                                sell_price = float(t)
                        except: pass
                else:
                    continue

                # UNIT CORRECTION: Webgia uses "đồng / chỉ" (1/10 lượng)
                # If price is < 50,000,000, it means it's per CHI, so multiply by 10
                if 0 < buy_price < 50_000_000:
                    buy_price *= 10
                if 0 < sell_price < 50_000_000:
                    sell_price *= 10

                if buy_price > 0 and sell_price > 0:
                    items.append(GoldPriceItem(
                        brand=brand,
                        product_type=f"Vàng {brand} - {current_location}",
                        buy_price=buy_price,
                        sell_price=sell_price,
                        updated_at=datetime.now(),
                        location=current_location
                    ))
                    
        except Exception as e:
            print(f"Error parsing WebGia: {e}")
            
        return items
