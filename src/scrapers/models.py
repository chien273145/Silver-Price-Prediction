from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class GoldPriceItem(BaseModel):
    """Represents a single gold/silver product price."""
    brand: str          # e.g., "SJC", "DOJI"
    product_type: str   # e.g., "Vàng miếng SJC", "Nhẫn tròn 9999"
    buy_price: float    # VND
    sell_price: float   # VND
    updated_at: datetime
    location: str = "Toàn quốc"
    
    # Computed fields
    spread: Optional[float] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.spread is None:
            self.spread = self.sell_price - self.buy_price

class ScraperResult(BaseModel):
    """Result from a scraper execution."""
    source: str
    success: bool
    items: List[GoldPriceItem] = []
    error: Optional[str] = None
    timestamp: datetime = datetime.now()
