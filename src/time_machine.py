"""
AI Time Machine - Portfolio Future Value Predictor

Predicts the future value of a user's portfolio based on AI price forecasts
for 7, 30, and 90 day horizons.

Architecture (following Backend Quality Guardian):
- Type hints on all functions
- Pydantic models for data validation
- Graceful error handling
- Business logic separate from API routes
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
import math


# ============================================
# Pydantic Models (Request/Response Validation)
# ============================================

class PortfolioItem(BaseModel):
    """Single item in user's portfolio."""
    id: str
    asset_type: str = Field(..., description="'gold' or 'silver'")
    brand: str
    quantity: float = Field(..., gt=0, description="Quantity in lượng/chỉ")
    buy_price: float = Field(..., gt=0, description="Original buy price per unit (VND)")
    buy_date: Optional[str] = None
    notes: Optional[str] = None


class PortfolioInput(BaseModel):
    """Input portfolio for time machine prediction."""
    items: List[PortfolioItem]
    current_gold_price: Optional[float] = None
    current_silver_price: Optional[float] = None


class FuturePrediction(BaseModel):
    """Single future prediction point."""
    days: int
    date: str
    predicted_value: float
    change_amount: float
    change_percent: float
    confidence_min: float
    confidence_max: float


class TimeMachineResult(BaseModel):
    """Complete time machine prediction result."""
    current_value: float
    total_invested: float
    current_profit: float
    current_profit_percent: float
    predictions: List[FuturePrediction]
    gold_trend: str  # "up", "down", "stable"
    silver_trend: str
    generated_at: str


# ============================================
# Constants (No magic numbers per skill)
# ============================================

PREDICTION_HORIZONS = [7, 30, 90]  # Days to predict

# Confidence interval multipliers (based on MAPE)
CONFIDENCE_7D = 0.034   # ~3.4% MAPE for 7-day
CONFIDENCE_30D = 0.08   # ~8% estimated for 30-day
CONFIDENCE_90D = 0.15   # ~15% estimated for 90-day

# Trend extrapolation factors for longer periods
TREND_DECAY_30D = 0.7   # Trend weakens over time
TREND_DECAY_90D = 0.5


# ============================================
# Core Calculator Class
# ============================================

class TimeMachineCalculator:
    """Calculates future portfolio value based on AI predictions."""
    
    def __init__(self) -> None:
        self.horizons = PREDICTION_HORIZONS
    
    def calculate(
        self,
        portfolio: PortfolioInput,
        ai_gold_predictions: Optional[List[Dict[str, Any]]] = None,
        ai_silver_predictions: Optional[List[Dict[str, Any]]] = None,
    ) -> TimeMachineResult:
        """
        Calculate future portfolio values.
        
        Args:
            portfolio: User's current portfolio items
            ai_gold_predictions: AI predictions for gold (7-day forecast)
            ai_silver_predictions: AI predictions for silver (7-day forecast)
        
        Returns:
            TimeMachineResult with current and future values
        """
        # Calculate current values
        current_gold_price = portfolio.current_gold_price or 0
        current_silver_price = portfolio.current_silver_price or 0
        
        total_invested = 0.0
        current_gold_value = 0.0
        current_silver_value = 0.0
        
        for item in portfolio.items:
            item_invested = item.quantity * item.buy_price
            total_invested += item_invested
            
            if item.asset_type.lower() == 'gold':
                current_gold_value += item.quantity * current_gold_price
            else:
                current_silver_value += item.quantity * current_silver_price
        
        current_value = current_gold_value + current_silver_value
        current_profit = current_value - total_invested
        current_profit_percent = (current_profit / total_invested * 100) if total_invested > 0 else 0
        
        # Extract trends from AI predictions
        gold_7d_change, gold_trend = self._extract_trend(ai_gold_predictions, current_gold_price)
        silver_7d_change, silver_trend = self._extract_trend(ai_silver_predictions, current_silver_price)
        
        # Calculate future predictions
        predictions: List[FuturePrediction] = []
        
        for days in self.horizons:
            pred = self._predict_value_at_days(
                days=days,
                current_gold_value=current_gold_value,
                current_silver_value=current_silver_value,
                gold_7d_change=gold_7d_change,
                silver_7d_change=silver_7d_change,
                current_value=current_value,
            )
            predictions.append(pred)
        
        return TimeMachineResult(
            current_value=round(current_value),
            total_invested=round(total_invested),
            current_profit=round(current_profit),
            current_profit_percent=round(current_profit_percent, 2),
            predictions=predictions,
            gold_trend=gold_trend,
            silver_trend=silver_trend,
            generated_at=datetime.now().isoformat(),
        )
    
    def _extract_trend(
        self, 
        predictions: Optional[List[Dict[str, Any]]], 
        current_price: float
    ) -> Tuple[float, str]:
        """Extract 7-day change percentage and trend direction from AI predictions."""
        if not predictions or len(predictions) < 7 or current_price <= 0:
            return 0.0, "stable"
        
        try:
            day7_price = predictions[6].get('price', current_price)
            change_pct = ((day7_price - current_price) / current_price) * 100
            
            if change_pct >= 1:
                trend = "up"
            elif change_pct <= -1:
                trend = "down"
            else:
                trend = "stable"
            
            return change_pct, trend
        except (IndexError, KeyError, TypeError):
            return 0.0, "stable"
    
    def _predict_value_at_days(
        self,
        days: int,
        current_gold_value: float,
        current_silver_value: float,
        gold_7d_change: float,
        silver_7d_change: float,
        current_value: float,
    ) -> FuturePrediction:
        """Calculate predicted portfolio value at given days horizon."""
        future_date = datetime.now() + timedelta(days=days)
        
        # Scale the 7-day trend to other periods with decay
        if days == 7:
            gold_change = gold_7d_change / 100
            silver_change = silver_7d_change / 100
            confidence = CONFIDENCE_7D
        elif days == 30:
            # Extrapolate with decay
            gold_change = (gold_7d_change / 100) * (30/7) * TREND_DECAY_30D
            silver_change = (silver_7d_change / 100) * (30/7) * TREND_DECAY_30D
            confidence = CONFIDENCE_30D
        else:  # 90 days
            gold_change = (gold_7d_change / 100) * (90/7) * TREND_DECAY_90D
            silver_change = (silver_7d_change / 100) * (90/7) * TREND_DECAY_90D
            confidence = CONFIDENCE_90D
        
        # Calculate future values
        future_gold = current_gold_value * (1 + gold_change)
        future_silver = current_silver_value * (1 + silver_change)
        predicted_value = future_gold + future_silver
        
        change_amount = predicted_value - current_value
        change_percent = (change_amount / current_value * 100) if current_value > 0 else 0
        
        # Confidence interval
        confidence_min = predicted_value * (1 - confidence)
        confidence_max = predicted_value * (1 + confidence)
        
        return FuturePrediction(
            days=days,
            date=future_date.strftime("%Y-%m-%d"),
            predicted_value=round(predicted_value),
            change_amount=round(change_amount),
            change_percent=round(change_percent, 2),
            confidence_min=round(confidence_min),
            confidence_max=round(confidence_max),
        )


# Singleton instance
time_machine = TimeMachineCalculator()


# ============================================
# Public API Function
# ============================================

def predict_portfolio_future(
    portfolio_items: List[Dict[str, Any]],
    current_gold_price: float,
    current_silver_price: float,
    gold_predictions: Optional[List[Dict[str, Any]]] = None,
    silver_predictions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Main entry point for Time Machine predictions.
    
    Args:
        portfolio_items: List of portfolio items from localStorage
        current_gold_price: Current gold price in VND/lượng
        current_silver_price: Current silver price in VND/lượng
        gold_predictions: AI gold price predictions (7-day)
        silver_predictions: AI silver price predictions (7-day)
    
    Returns:
        Dict with current values, future predictions, and confidence intervals
    """
    try:
        # Convert raw items to Pydantic models
        items = []
        for item in portfolio_items:
            items.append(PortfolioItem(
                id=str(item.get('id', '')),
                asset_type=item.get('asset_type', 'silver'),
                brand=item.get('brand', 'Unknown'),
                quantity=float(item.get('quantity', 0)),
                buy_price=float(item.get('buy_price', 0)),
                buy_date=item.get('buy_date'),
                notes=item.get('notes'),
            ))
        
        portfolio = PortfolioInput(
            items=items,
            current_gold_price=current_gold_price,
            current_silver_price=current_silver_price,
        )
        
        result = time_machine.calculate(
            portfolio=portfolio,
            ai_gold_predictions=gold_predictions,
            ai_silver_predictions=silver_predictions,
        )
        
        return result.model_dump()
        
    except Exception as e:
        print(f"Time Machine Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Return graceful fallback
        return {
            "error": str(e),
            "current_value": 0,
            "total_invested": 0,
            "current_profit": 0,
            "current_profit_percent": 0,
            "predictions": [],
            "gold_trend": "stable",
            "silver_trend": "stable",
            "generated_at": datetime.now().isoformat(),
        }
