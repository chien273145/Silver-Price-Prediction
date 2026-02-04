import asyncio
from typing import List, Dict
from datetime import datetime
from .btmc_scraper import BTMCScraper
from .phu_quy_scraper import PhuQuyScraper
from .ancarat_scraper import AncaratScraper
from .sjc_scraper import SJCScraper
from .doji_scraper import DOJIScraper
from .pnj_scraper import PNJScraper
from .webgia_scraper import WebGiaScraper
from .world_price_scraper import WorldPriceScraper
from .models import GoldPriceItem, ScraperResult

class PriceService:
    """Service to orchestrate fetching prices from multiple sources."""
    
    def __init__(self):
        self.scrapers = [
            WebGiaScraper(),
            WorldPriceScraper(),
            BTMCScraper(),
            PhuQuyScraper(),
            AncaratScraper(),
            # Backup original scrapers just in case they start working or for specific data
            # SJCScraper(), 
            # DOJIScraper(),
            # PNJScraper()
        ]
        # Simple in-memory cache
        self.cache: Dict[str, ScraperResult] = {}
        self.last_update = datetime.min
        self.cache_duration_seconds = 300 # 5 minutes

    async def fetch_all(self) -> Dict[str, List[GoldPriceItem]]:
        """
        Runs all scrapers concurrently and returns aggregated results.
        Returns a dictionary grouped by brand/source.
        """
        # Check cache
        if (datetime.now() - self.last_update).total_seconds() < self.cache_duration_seconds and self.cache:
            return self._format_results(list(self.cache.values()))

        # Run scrapers in thread pool (since requests is synchronous)
        # In a real async scraper we'd use aiohttp, but here we wrap sync calls
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, scraper.scrape)
            for scraper in self.scrapers
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Check if we got any valid results
        valid_items = []
        for res in results:
            if res.success and res.items:
                valid_items.extend(res.items)

        # Fallback to Mock Data if total failure (common due to bot blocking)
        if not valid_items:
            print("⚠️ All scrapers failed or empty. Using Mock Data.")
            return self._get_mock_data()
        
        # Update cache
        for result in results:
            self.cache[result.source] = result
        self.last_update = datetime.now()
        
        return self._format_results(results)

    def _get_mock_data(self) -> Dict[str, List[GoldPriceItem]]:
        """Return realistic mock data when scrapers fail."""
        mock_items = [
            # SJC
            GoldPriceItem(brand="SJC", product_type="Vàng SJC 1L - 10L", buy_price=79000000, sell_price=81000000, updated_at=datetime.now(), location="TP.HCM"),
            GoldPriceItem(brand="SJC", product_type="Vàng Nhẫn SJC 99,99 1 chỉ", buy_price=68000000, sell_price=69500000, updated_at=datetime.now(), location="TP.HCM"),
            
            # DOJI
            GoldPriceItem(brand="DOJI", product_type="AVPL / DOJI Hưng Thịnh Vượng", buy_price=78900000, sell_price=80900000, updated_at=datetime.now(), location="Hà Nội"),
            GoldPriceItem(brand="DOJI", product_type="Nhẫn Tròn 9999 (Hưng Thịnh Vượng)", buy_price=68500000, sell_price=69800000, updated_at=datetime.now(), location="Hà Nội"),
            
            # PNJ
            GoldPriceItem(brand="PNJ", product_type="Vàng miếng PNJ (999.9)", buy_price=68600000, sell_price=69900000, updated_at=datetime.now(), location="Toàn quốc"),
            GoldPriceItem(brand="PNJ", product_type="Nhẫn Trơn PNJ 999.9", buy_price=68600000, sell_price=69850000, updated_at=datetime.now(), location="Toàn quốc"),
        ]
        
        # Wrap in ScraperResults to reuse format_results
        results = [
            ScraperResult(source="SJC", success=True, items=[mock_items[0], mock_items[1]]),
            ScraperResult(source="DOJI", success=True, items=[mock_items[2], mock_items[3]]),
            ScraperResult(source="PNJ", success=True, items=[mock_items[4], mock_items[5]])
        ]
        return self._format_results(results, is_mock=True)

    def _format_results(self, results: List[ScraperResult], is_mock: bool = False) -> Dict[str, List[GoldPriceItem]]:
        """Format results into a clean dictionary."""
        output = {
            "timestamp": datetime.now().isoformat(),
            "is_mock": is_mock,
            "sources": [],
            "items": []
        }
        
        for res in results:
            source_info = {
                "name": res.source,
                "status": "ok" if res.success else "error",
                "count": len(res.items)
            }
            if res.error:
                source_info["error"] = res.error
            output["sources"].append(source_info)
            
            output["items"].extend(res.items)
            
        # Sort items by brand then price
        output["items"] = sorted(output["items"], key=lambda x: (x.brand, x.product_type))
        return output

# Global instance
_price_service = None

def get_price_service():
    global _price_service
    if _price_service is None:
        _price_service = PriceService()
    return _price_service
