from typing import List
from datetime import datetime
import yfinance as yf
from .base_scraper import BaseScraper
from .models import GoldPriceItem, ScraperResult

class WorldPriceScraper(BaseScraper):
    """Fetches World Gold & Silver prices and converts to VND."""
    
    def __init__(self):
        super().__init__("WorldPrice", "https://finance.yahoo.com")
        
    def fetch_page(self):
        # Not used, we use yfinance
        return None

    def parse(self, html_content):
        # Not used
        return []

    def scrape(self) -> ScraperResult:
        items = []
        try:
            # Fetch Data: GC=F (Gold Futures), SI=F (Silver Futures), VND=X (USDVND)
            # Spot tickers (XAU=X) can be flaky via yfinance sometimes regarding period
            tickers = yf.Tickers("GC=F SI=F VND=X")
            
            # fast_info might be engaging different endpoint, try info or history if fast_info fails
            # But fast_info.last_price is efficient. 
            
            # Using defaults 0.0 to prevent crash if one is missing
            # Using history() as fast_info can be unreliable/inconsistent units
            try:
                gold_usd = tickers.tickers['GC=F'].history(period="1d")['Close'].iloc[-1]
                silver_usd = tickers.tickers['SI=F'].history(period="1d")['Close'].iloc[-1]
                usd_vnd = tickers.tickers['VND=X'].history(period="1d")['Close'].iloc[-1]
            except Exception as e:
                return ScraperResult(success=False, source=self.source_name, error=f"YFinance History Error: {str(e)}", items=[])

            if not (gold_usd and silver_usd and usd_vnd):
                return ScraperResult(success=False, source=self.source_name, error="Failed to fetch yfinance data", items=[])

            # Constants
            # Futures are per Ounce.
            OZ_TO_TAEL = 1.205653  # 37.5g / 31.1035g
            
            # Gold Conversion
            gold_vnd_tael = gold_usd * usd_vnd * OZ_TO_TAEL
            
            # Silver Conversion
            silver_vnd_tael = silver_usd * usd_vnd * OZ_TO_TAEL
            
            # Create Items
            # World Gold
            items.append(GoldPriceItem(
                brand="WORLD", # Uppercase for style matching if consistent, or handled
                product_type="Vàng Thế Giới (Quy đổi)",
                buy_price=gold_vnd_tael,
                sell_price=gold_vnd_tael, 
                updated_at=datetime.now(),
                location="Thế Giới"
            ))
            
            # World Silver
            items.append(GoldPriceItem(
                brand="WORLD",
                product_type="Bạc Thế Giới (Quy đổi)",
                buy_price=silver_vnd_tael,
                sell_price=silver_vnd_tael,
                updated_at=datetime.now(),
                location="Thế Giới"
            ))
            
            return ScraperResult(success=True, source=self.source_name, items=items)
            
        except Exception as e:
            return ScraperResult(success=False, source=self.source_name, error=f"YFinance Error: {str(e)}", items=[])
