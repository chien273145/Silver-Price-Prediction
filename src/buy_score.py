"""
AI Buy Score Calculator
Calculates a 0-100 score indicating whether it's a good time to buy gold/silver.

Scoring Factors:
- Spread Analysis (20 pts): Low spread = better time to buy
- AI Price Prediction (25 pts): If AI predicts price increase
- USD/VND Rate (15 pts): Weak USD = favorable for precious metals
- VIX Fear Index (15 pts): Higher fear = good for safe haven assets
- Price vs 7-day Average (15 pts): Below average = discount
- Time Factors (10 pts): Avoid high-spread periods like Thần Tài
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import math


class BuyScoreCalculator:
    """Calculates AI Buy Score for gold/silver purchases."""
    
    def __init__(self):
        # Average spread benchmarks (in VND per lượng)
        self.avg_spread_gold = 1_500_000  # 1.5 triệu típical spread SJC
        self.avg_spread_silver = 100_000   # 100k típical spread bạc Phú Quý
        
        # Special dates with typically high spreads (MM-DD format)
        self.high_spread_dates = [
            "02-10",  # Ngày vía Thần Tài (mùng 10 tháng Giêng âm lịch ~ around this date)
            "02-14",  # Valentine's
            "10-20",  # Phụ nữ Việt Nam
        ]
    
    def calculate(
        self,
        asset_type: str,  # "gold" or "silver"
        spread: Optional[float] = None,
        ai_prediction_change: Optional[float] = None,  # % change predicted
        usd_change: Optional[float] = None,  # USD % change
        vix_value: Optional[float] = None,
        current_price: Optional[float] = None,
        avg_7day_price: Optional[float] = None,
    ) -> Dict:
        """
        Calculate buy score based on all available factors.
        
        Returns:
            {
                "score": 72,
                "label": "Khá tốt",
                "color": "green",
                "factors": [
                    {"name": "Spread", "points": 15, "max": 20, "detail": "..."},
                    ...
                ],
                "recommendation": "Thời điểm khá tốt để mua..."
            }
        """
        factors = []
        total_score = 0
        
        # 1. Spread Analysis (20 pts max)
        spread_score, spread_detail = self._calc_spread_score(asset_type, spread)
        factors.append({
            "name": "Chênh lệch giá",
            "icon": "📊",
            "points": spread_score,
            "max": 20,
            "detail": spread_detail
        })
        total_score += spread_score
        
        # 2. AI Price Prediction (25 pts max)
        ai_score, ai_detail = self._calc_ai_prediction_score(ai_prediction_change)
        factors.append({
            "name": "AI Dự báo",
            "icon": "🤖",
            "points": ai_score,
            "max": 25,
            "detail": ai_detail
        })
        total_score += ai_score
        
        # 3. USD/VND Rate (15 pts max)
        usd_score, usd_detail = self._calc_usd_score(usd_change)
        factors.append({
            "name": "Tỷ giá USD",
            "icon": "💵",
            "points": usd_score,
            "max": 15,
            "detail": usd_detail
        })
        total_score += usd_score
        
        # 4. VIX Fear Index (15 pts max)
        vix_score, vix_detail = self._calc_vix_score(vix_value)
        factors.append({
            "name": "Chỉ số sợ hãi",
            "icon": "📈",
            "points": vix_score,
            "max": 15,
            "detail": vix_detail
        })
        total_score += vix_score
        
        # 5. Price vs 7-day Average (15 pts max)
        price_score, price_detail = self._calc_price_avg_score(current_price, avg_7day_price)
        factors.append({
            "name": "So với TB 7 ngày",
            "icon": "📉",
            "points": price_score,
            "max": 15,
            "detail": price_detail
        })
        total_score += price_score
        
        # 6. Time Factors (10 pts max)
        time_score, time_detail = self._calc_time_score()
        factors.append({
            "name": "Thời điểm",
            "icon": "📅",
            "points": time_score,
            "max": 10,
            "detail": time_detail
        })
        total_score += time_score
        
        # Get label and color
        label, color = self._get_label_and_color(total_score)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(asset_type, total_score, factors)
        
        return {
            "score": round(total_score),
            "label": label,
            "color": color,
            "asset_type": asset_type,
            "factors": factors,
            "recommendation": recommendation,
            "updated_at": datetime.now().isoformat()
        }
    
    def _calc_spread_score(self, asset_type: str, spread: Optional[float]) -> Tuple[float, str]:
        """Lower spread = higher score"""
        if spread is None:
            return 10, "Không có dữ liệu spread"
        
        avg_spread = self.avg_spread_gold if asset_type == "gold" else self.avg_spread_silver
        
        if spread <= avg_spread * 0.7:
            return 20, f"Spread rất thấp ({spread:,.0f}đ)"
        elif spread <= avg_spread * 0.9:
            return 16, f"Spread thấp hơn TB ({spread:,.0f}đ)"
        elif spread <= avg_spread * 1.1:
            return 12, f"Spread bình thường ({spread:,.0f}đ)"
        elif spread <= avg_spread * 1.3:
            return 6, f"Spread cao hơn TB ({spread:,.0f}đ)"
        else:
            return 2, f"Spread rất cao ({spread:,.0f}đ)"
    
    def _calc_ai_prediction_score(self, prediction_change: Optional[float]) -> Tuple[float, str]:
        """AI predicts price increase = higher score"""
        if prediction_change is None:
            return 12, "Không có dữ liệu dự báo"
        
        if prediction_change >= 3:
            return 25, f"AI dự báo tăng mạnh (+{prediction_change:.1f}%)"
        elif prediction_change >= 1:
            return 20, f"AI dự báo tăng (+{prediction_change:.1f}%)"
        elif prediction_change >= 0:
            return 15, f"AI dự báo ổn định ({prediction_change:+.1f}%)"
        elif prediction_change >= -1:
            return 10, f"AI dự báo giảm nhẹ ({prediction_change:.1f}%)"
        elif prediction_change >= -3:
            return 5, f"AI dự báo giảm ({prediction_change:.1f}%)"
        else:
            return 0, f"AI dự báo giảm mạnh ({prediction_change:.1f}%)"
    
    def _calc_usd_score(self, usd_change: Optional[float]) -> Tuple[float, str]:
        """Weak USD = higher score (good for gold/silver)"""
        if usd_change is None:
            return 8, "Không có dữ liệu USD"
        
        if usd_change <= -1.0:
            return 15, f"USD giảm mạnh ({usd_change:.2f}%) ✓"
        elif usd_change <= -0.3:
            return 12, f"USD giảm nhẹ ({usd_change:.2f}%) ✓"
        elif usd_change <= 0.3:
            return 8, f"USD ổn định ({usd_change:+.2f}%)"
        elif usd_change <= 1.0:
            return 4, f"USD tăng nhẹ (+{usd_change:.2f}%)"
        else:
            return 0, f"USD tăng mạnh (+{usd_change:.2f}%)"
    
    def _calc_vix_score(self, vix_value: Optional[float]) -> Tuple[float, str]:
        """Higher VIX = higher score (fear = flight to safety)"""
        if vix_value is None:
            return 8, "Không có dữ liệu VIX"
        
        if vix_value >= 30:
            return 15, f"Thị trường sợ hãi cao (VIX={vix_value:.1f}) ✓"
        elif vix_value >= 20:
            return 12, f"Tâm lý lo ngại (VIX={vix_value:.1f}) ✓"
        elif vix_value >= 15:
            return 8, f"Thị trường bình thường (VIX={vix_value:.1f})"
        elif vix_value >= 12:
            return 5, f"Thị trường lạc quan (VIX={vix_value:.1f})"
        else:
            return 2, f"Thị trường rất lạc quan (VIX={vix_value:.1f})"
    
    def _calc_price_avg_score(self, current: Optional[float], avg_7day: Optional[float]) -> Tuple[float, str]:
        """Below 7-day average = discount = higher score"""
        if current is None or avg_7day is None or avg_7day == 0:
            return 8, "Không có dữ liệu so sánh"
        
        diff_pct = ((current - avg_7day) / avg_7day) * 100
        
        if diff_pct <= -3:
            return 15, f"Giá thấp hơn TB 7 ngày {diff_pct:.1f}% ✓"
        elif diff_pct <= -1:
            return 12, f"Giá thấp hơn TB ({diff_pct:.1f}%) ✓"
        elif diff_pct <= 1:
            return 8, f"Giá gần bằng TB ({diff_pct:+.1f}%)"
        elif diff_pct <= 3:
            return 4, f"Giá cao hơn TB (+{diff_pct:.1f}%)"
        else:
            return 0, f"Giá cao hơn TB +{diff_pct:.1f}%"
    
    def _calc_time_score(self) -> Tuple[float, str]:
        """Check if it's a high-spread period"""
        today = datetime.now()
        today_str = today.strftime("%m-%d")
        
        # Check if near high-spread dates
        for date_str in self.high_spread_dates:
            date_month, date_day = map(int, date_str.split("-"))
            target_date = datetime(today.year, date_month, date_day)
            days_diff = abs((today - target_date).days)
            
            if days_diff <= 3:
                return 2, "Gần ngày Thần Tài/lễ (spread thường cao)"
            elif days_diff <= 7:
                return 6, "Cách ngày lễ 1 tuần"
        
        # Check weekend (markets closed, stale prices)
        if today.weekday() >= 5:
            return 6, "Cuối tuần (giá ít cập nhật)"
        
        return 10, "Thời điểm bình thường ✓"
    
    def _get_label_and_color(self, score: float) -> Tuple[str, str]:
        """Get label and color based on score"""
        if score >= 80:
            return "Rất tốt", "green"
        elif score >= 60:
            return "Khá tốt", "lightgreen"
        elif score >= 40:
            return "Trung bình", "yellow"
        else:
            return "Chưa nên", "red"
    
    def _generate_recommendation(self, asset_type: str, score: float, factors: List[Dict]) -> str:
        """Generate natural language recommendation"""
        asset_name = "vàng" if asset_type == "gold" else "bạc"
        
        if score >= 80:
            return f"Đây là thời điểm RẤT TỐT để mua {asset_name}. Nhiều yếu tố thuận lợi."
        elif score >= 60:
            return f"Thời điểm khá tốt để mua {asset_name}. Có thể cân nhắc mua."
        elif score >= 40:
            return f"Thời điểm trung bình. Có thể chờ thêm hoặc mua với số lượng nhỏ."
        else:
            return f"Chưa nên mua {asset_name} lúc này. Nên chờ điều kiện tốt hơn."


# Singleton instance
buy_score_calculator = BuyScoreCalculator()


def calculate_buy_score(
    asset_type: str = "silver",
    spread: float = None,
    ai_prediction_change: float = None,
    usd_change: float = None,
    vix_value: float = None,
    current_price: float = None,
    avg_7day_price: float = None,
) -> Dict:
    """
    Main function to calculate buy score.
    
    Args:
        asset_type: "gold" or "silver"
        spread: Current buy-sell spread in VND
        ai_prediction_change: AI predicted % change for next 7 days
        usd_change: USD/VND % change (negative = USD weaker)
        vix_value: Current VIX index value
        current_price: Current price in VND
        avg_7day_price: Average price over last 7 days
    
    Returns:
        Dict with score, label, factors, recommendation
    """
    return buy_score_calculator.calculate(
        asset_type=asset_type,
        spread=spread,
        ai_prediction_change=ai_prediction_change,
        usd_change=usd_change,
        vix_value=vix_value,
        current_price=current_price,
        avg_7day_price=avg_7day_price,
    )
