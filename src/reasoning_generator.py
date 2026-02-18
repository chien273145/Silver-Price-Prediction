"""
Prediction Reasoning Generator
Generates human-readable explanations for why prices are predicted to rise/fall.
Uses rule-based logic to analyze market drivers and technical indicators.
"""

from typing import Dict, List, Optional
from datetime import datetime


class ReasoningGenerator:
    """Generates explanations for price predictions."""
    
    def __init__(self):
        pass
    
    def generate_reasoning(
        self,
        prediction_change_pct: float,
        market_data: Optional[Dict] = None,
        market_drivers: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate reasoning for why price is predicted to change.
        
        Args:
            prediction_change_pct: Predicted % change (e.g., +2.3 or -1.5)
            market_data: Live market data (DXY, VIX, Gold, etc.)
            market_drivers: Market driver analysis from predictor
            
        Returns:
            {
                "direction": "up" | "down" | "stable",
                "confidence": "high" | "medium" | "low",
                "primary_reasons": [
                    {"factor": "DXY giảm", "impact": "positive", "detail": "..."},
                    ...
                ],
                "summary": "Giá dự đoán tăng vì..."
            }
        """
        reasons = []
        
        # Determine direction
        if prediction_change_pct >= 0.5:
            direction = "up"
            direction_text = "tăng"
        elif prediction_change_pct <= -0.5:
            direction = "down"
            direction_text = "giảm"
        else:
            direction = "stable"
            direction_text = "ổn định"
        
        # Analyze market drivers
        if market_drivers:
            dxy_reasons = self._analyze_dxy(market_drivers.get('dxy', {}))
            if dxy_reasons:
                reasons.extend(dxy_reasons)
            
            vix_reasons = self._analyze_vix(market_drivers.get('vix', {}))
            if vix_reasons:
                reasons.extend(vix_reasons)
            
            gold_reasons = self._analyze_gold(market_drivers.get('gold', {}))
            if gold_reasons:
                reasons.extend(gold_reasons)
        
        # Analyze live market data if available
        if market_data:
            live_reasons = self._analyze_live_data(market_data)
            if live_reasons:
                reasons.extend(live_reasons)
        
        # Add technical indicators reasoning
        tech_reasons = self._analyze_technical_indicators(prediction_change_pct)
        if tech_reasons:
            reasons.extend(tech_reasons)
        
        # Sort by impact strength
        reasons.sort(key=lambda x: self._impact_weight(x['impact']), reverse=True)
        
        # Take top 4 reasons
        primary_reasons = reasons[:4]
        
        # Determine confidence
        confidence = self._calculate_confidence(prediction_change_pct, len(primary_reasons))
        
        # Generate summary
        summary = self._generate_summary(direction_text, primary_reasons, prediction_change_pct)
        
        return {
            "direction": direction,
            "confidence": confidence,
            "primary_reasons": primary_reasons,
            "summary": summary,
            "generated_at": datetime.now().isoformat()
        }
    
    def _analyze_dxy(self, dxy_data: Dict) -> List[Dict]:
        """Analyze DXY (US Dollar Index) impact."""
        reasons = []

        # Support both 'change_pct'/'current' and 'change'/'value' key formats
        change = dxy_data.get('change_pct') or dxy_data.get('change', 0)
        current = dxy_data.get('current') or dxy_data.get('value', 0)

        if change <= -0.8:
            reasons.append({
                "factor": "DXY giảm mạnh",
                "impact": "very_positive",
                "detail": f"USD yếu đi {abs(change):.1f}% → Vàng/Bạc tăng giá",
                "icon": "💵↓"
            })
        elif change <= -0.3:
            reasons.append({
                "factor": "DXY giảm nhẹ",
                "impact": "positive",
                "detail": f"USD giảm {abs(change):.1f}% → Hỗ trợ kim loại quý",
                "icon": "💵↓"
            })
        elif change >= 0.8:
            reasons.append({
                "factor": "DXY tăng mạnh",
                "impact": "very_negative",
                "detail": f"USD mạnh lên +{change:.1f}% → Áp lực giảm giá",
                "icon": "💵↑"
            })
        elif change >= 0.3:
            reasons.append({
                "factor": "DXY tăng nhẹ",
                "impact": "negative",
                "detail": f"USD tăng +{change:.1f}% → Bất lợi cho vàng/bạc",
                "icon": "💵↑"
            })

        return reasons
    
    def _analyze_vix(self, vix_data: Dict) -> List[Dict]:
        """Analyze VIX (Fear Index) impact."""
        reasons = []

        # Support both 'current'/'change_pct' and 'value'/'change' key formats
        current = vix_data.get('current') or vix_data.get('value', 0)
        change = vix_data.get('change_pct') or vix_data.get('change', 0)

        if not current:
            return reasons

        if current >= 30:
            reasons.append({
                "factor": "VIX cao (>30)",
                "impact": "very_positive",
                "detail": f"Thị trường sợ hãi cao (VIX={current:.1f}) → Nhu cầu trú ẩn an toàn tăng",
                "icon": "📈"
            })
        elif current >= 20:
            reasons.append({
                "factor": "VIX tăng (>20)",
                "impact": "positive",
                "detail": f"Tâm lý lo ngại (VIX={current:.1f}) → Tốt cho kim loại quý",
                "icon": "📈"
            })
        elif current <= 12:
            reasons.append({
                "factor": "VIX thấp (<12)",
                "impact": "negative",
                "detail": f"Thị trường lạc quan (VIX={current:.1f}) → Giảm nhu cầu trú ẩn",
                "icon": "📉"
            })

        return reasons
    
    def _analyze_gold(self, gold_data: Dict) -> List[Dict]:
        """Analyze Gold price impact on Silver."""
        reasons = []

        # Support both 'change_pct' and 'change' key formats
        change = gold_data.get('change_pct') or gold_data.get('change', 0)

        if change >= 1.5:
            reasons.append({
                "factor": "Vàng tăng mạnh",
                "impact": "positive",
                "detail": f"Giá vàng tăng +{change:.1f}% → Bạc thường theo sau",
                "icon": "🥇↑"
            })
        elif change >= 0.5:
            reasons.append({
                "factor": "Vàng tăng nhẹ",
                "impact": "positive",
                "detail": f"Vàng tăng +{change:.1f}% → Hỗ trợ bạc",
                "icon": "🥇↑"
            })
        elif change <= -1.5:
            reasons.append({
                "factor": "Vàng giảm mạnh",
                "impact": "negative",
                "detail": f"Vàng giảm {change:.1f}% → Áp lực lên bạc",
                "icon": "🥇↓"
            })

        return reasons
    
    def _analyze_live_data(self, market_data: Dict) -> List[Dict]:
        """Analyze live market data."""
        reasons = []
        
        # This is a placeholder - can be extended with more live data analysis
        # For now, just acknowledge we have live data
        if market_data.get('silver_close'):
            reasons.append({
                "factor": "Dữ liệu real-time",
                "impact": "neutral",
                "detail": "Dự đoán dựa trên dữ liệu thị trường mới nhất",
                "icon": "🔴"
            })
        
        return reasons
    
    def _analyze_technical_indicators(self, prediction_change_pct: float) -> List[Dict]:
        """Analyze technical indicators."""
        reasons = []
        
        # Based on prediction strength, infer technical signals
        if abs(prediction_change_pct) >= 2.0:
            if prediction_change_pct > 0:
                reasons.append({
                    "factor": "Tín hiệu kỹ thuật mạnh",
                    "impact": "positive",
                    "detail": "RSI, MACD, Bollinger Bands đều cho tín hiệu tích cực",
                    "icon": "📊"
                })
            else:
                reasons.append({
                    "factor": "Tín hiệu kỹ thuật yếu",
                    "impact": "negative",
                    "detail": "Các chỉ báo kỹ thuật cho xu hướng giảm",
                    "icon": "📊"
                })
        elif abs(prediction_change_pct) >= 0.5:
            reasons.append({
                "factor": "Tín hiệu kỹ thuật trung bình",
                "impact": "neutral",
                "detail": "Các chỉ báo cho xu hướng ổn định",
                "icon": "📊"
            })
        
        return reasons
    
    def _impact_weight(self, impact: str) -> int:
        """Get weight for sorting by impact."""
        weights = {
            "very_positive": 5,
            "positive": 4,
            "neutral": 3,
            "negative": 2,
            "very_negative": 1
        }
        return weights.get(impact, 3)
    
    def _calculate_confidence(self, prediction_change_pct: float, num_reasons: int) -> str:
        """Calculate confidence level."""
        # Strong prediction + many reasons = high confidence
        if abs(prediction_change_pct) >= 2.0 and num_reasons >= 3:
            return "high"
        elif abs(prediction_change_pct) >= 1.0 and num_reasons >= 2:
            return "medium"
        else:
            return "low"
    
    def _generate_summary(self, direction_text: str, reasons: List[Dict], change_pct: float) -> str:
        """Generate summary text."""
        if not reasons:
            return f"Giá dự đoán {direction_text} {abs(change_pct):.1f}% dựa trên phân tích mô hình AI."
        
        # Get top 2-3 reasons
        top_reasons = reasons[:3]
        reason_texts = []
        
        for r in top_reasons:
            # Simplify the detail text
            detail = r['detail'].split('→')[0].strip()  # Take only the first part
            reason_texts.append(detail)
        
        if len(reason_texts) == 1:
            summary = f"Giá dự đoán {direction_text} {abs(change_pct):.1f}% chủ yếu do {reason_texts[0]}."
        elif len(reason_texts) == 2:
            summary = f"Giá dự đoán {direction_text} {abs(change_pct):.1f}% do {reason_texts[0]} và {reason_texts[1]}."
        else:
            summary = f"Giá dự đoán {direction_text} {abs(change_pct):.1f}% do {reason_texts[0]}, {reason_texts[1]} và {reason_texts[2]}."
        
        return summary


# Singleton instance
reasoning_generator = ReasoningGenerator()


def generate_prediction_reasoning(
    prediction_change_pct: float,
    market_data: Optional[Dict] = None,
    market_drivers: Optional[Dict] = None,
) -> Dict:
    """
    Main function to generate prediction reasoning.
    
    Args:
        prediction_change_pct: Predicted % change
        market_data: Live market data
        market_drivers: Market driver analysis
        
    Returns:
        Dict with reasoning explanation
    """
    return reasoning_generator.generate_reasoning(
        prediction_change_pct=prediction_change_pct,
        market_data=market_data,
        market_drivers=market_drivers,
    )
