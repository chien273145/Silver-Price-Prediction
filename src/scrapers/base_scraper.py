import abc
import requests
from typing import List, Optional
from datetime import datetime
from .models import ScraperResult, GoldPriceItem

class BaseScraper(abc.ABC):
    """Abstract base class for all gold price scrapers."""
    
    def __init__(self, source_name: str, base_url: str):
        self.source_name = source_name
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }

    def fetch_page(self) -> Optional[str]:
        """Fetches the HTML content of the page."""
        try:
            response = requests.get(self.base_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching {self.source_name}: {e}")
            return None

    @abc.abstractmethod
    def parse(self, html_content: str) -> List[GoldPriceItem]:
        """Parses the HTML and returns a list of GoldPriceItems."""
        pass

    def scrape(self) -> ScraperResult:
        """Main execution method."""
        try:
            html = self.fetch_page()
            if not html:
                return ScraperResult(source=self.source_name, success=False, error="Failed to fetch page")
            
            items = self.parse(html)
            return ScraperResult(source=self.source_name, success=True, items=items)
            
        except Exception as e:
            return ScraperResult(source=self.source_name, success=False, error=str(e))
