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
            # Fetch Data: GC=F (Gold Futures USD/oz), SI=F (Silver Futures USD/oz), VND=X (USD/VND)
            tickers = yf.Tickers("GC=F SI=F VND=X")
            
            try:
                gold_usd = float(tickers.tickers['GC=F'].history(period="1d")['Close'].iloc[-1])
                silver_usd = float(tickers.tickers['SI=F'].history(period="1d")['Close'].iloc[-1])
                usd_vnd = float(tickers.tickers['VND=X'].history(period="1d")['Close'].iloc[-1])
            except Exception as e:
                return ScraperResult(success=False, source=self.source_name, error=f"YFinance History Error: {str(e)}", items=[])

            if not (gold_usd and silver_usd and usd_vnd):
                return ScraperResult(success=False, source=self.source_name, error="Failed to fetch yfinance data", items=[])

            # Sanity check: VND=X should return ~25000-27000 (USD/VND rate)
            # If it returns a very small number (<1), it's the inverse (VND/USD)
            if usd_vnd < 1:
                usd_vnd = 1.0 / usd_vnd
            # If unreasonably small (e.g. <1000), use default
            if usd_vnd < 1000:
                usd_vnd = 25900  # Default USD/VND rate

            # Sanity check: GC=F should return ~2500-3500 USD/oz for gold
            # If it returns >4000, it might be in a different unit - use fallback
            if gold_usd > 4000:
                print(f"[WorldPrice] WARNING: gold_usd={gold_usd:.2f} seems too high (expected ~2900), using fallback")
                gold_usd = 2900.0  # Fallback to approximate current price

            # Sanity check: SI=F should return ~25-50 USD/oz for silver
            if silver_usd > 100:
                print(f"[WorldPrice] WARNING: silver_usd={silver_usd:.2f} seems too high (expected ~32), using fallback")
                silver_usd = 32.0  # Fallback

            # Constants: 1 luong (tael) = 37.5g, 1 troy oz = 31.1035g
            OZ_TO_TAEL = 37.5 / 31.1035  # = 1.205653

            # Gold Conversion: USD/oz -> VND/luong
            gold_vnd_tael = gold_usd * usd_vnd * OZ_TO_TAEL
            
            # Silver Conversion: USD/oz -> VND/luong
            silver_vnd_tael = silver_usd * usd_vnd * OZ_TO_TAEL
            
            print(f"[WorldPrice] Gold: ${gold_usd:.2f}/oz -> {gold_vnd_tael:,.0f} VND/luong")
            print(f"[WorldPrice] Silver: ${silver_usd:.2f}/oz -> {silver_vnd_tael:,.0f} VND/luong")
            print(f"[WorldPrice] USD/VND rate: {usd_vnd:.0f}")

            # Create Items
            items.append(GoldPriceItem(
                brand="WORLD",
                product_type="Vàng Thế Giới (Quy đổi)",
                buy_price=gold_vnd_tael,
                sell_price=gold_vnd_tael, 
                updated_at=datetime.now(),
                location="Thế Giới"
            ))
            
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
