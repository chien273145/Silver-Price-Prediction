"""
Action Recommendation Generator
Provides market condition analysis and educational guidance WITHOUT direct investment advice.
Complies with Vietnamese investment advisory regulations.
"""

from typing import Dict, Optional
from datetime import datetime


class ActionRecommendationGenerator:
    """
    Generates educational market analysis and conditions assessment.
    
    IMPORTANT: This does NOT provide investment advice. All outputs are for
    informational and educational purposes only.
    """
    
    # Legal disclaimer (always included)
    DISCLAIMER = (
        "⚠️ Thông tin này chỉ mang tính chất THAM KHẢO và GIÁO DỤC, "
        "KHÔNG phải lời khuyên đầu tư. Bạn nên tự nghiên cứu kỹ và "
        "tham khảo ý kiến chuyên gia tài chính có giấy phép trước khi "
        "đưa ra quyết định đầu tư."
    )
    
    def __init__(self):
        pass
    
    def generate_recommendation(
        self,
        buy_score: int,
        asset_type: str = "silver",
        prediction_trend: str = "up",
        volatility: str = "medium",
        user_goal: Optional[str] = None,  # "accumulate" or "trade"
    ) -> Dict:
        """
        Generate market condition assessment and educational guidance.
        
        Args:
            buy_score: AI Buy Score (0-100)
            asset_type: "gold" or "silver"
            prediction_trend: "up", "down", or "stable"
            volatility: "low", "medium", "high"
            user_goal: Optional user goal context
            
        Returns:
            {
                "market_condition": "favorable" | "neutral" | "unfavorable",
                "condition_label": "Điều kiện thuận lợi",
                "educational_points": [...],
                "risk_warnings": [...],
                "considerations": [...],
                "disclaimer": "..."
            }
        """
        asset_name = "vàng" if asset_type == "gold" else "bạc"
        
        # Determine market condition (NOT "should buy" or "should not buy")
        if buy_score >= 75:
            condition = "favorable"
            condition_label = "Điều kiện thuận lợi"
            condition_color = "green"
        elif buy_score >= 55:
            condition = "moderately_favorable"
            condition_label = "Điều kiện khá thuận lợi"
            condition_color = "lightgreen"
        elif buy_score >= 40:
            condition = "neutral"
            condition_label = "Điều kiện trung lập"
            condition_color = "yellow"
        else:
            condition = "unfavorable"
            condition_label = "Điều kiện chưa thuận lợi"
            condition_color = "orange"
        
        # Educational points (what the data shows)
        educational_points = self._get_educational_points(
            buy_score, asset_name, prediction_trend, volatility
        )
        
        # Risk warnings (always present)
        risk_warnings = self._get_risk_warnings(volatility, asset_name)
        
        # Considerations for decision-making
        considerations = self._get_considerations(
            condition, asset_name, user_goal, prediction_trend
        )
        
        # Strategy suggestions (educational, not directive)
        strategies = self._get_strategy_education(condition, volatility, user_goal)
        
        return {
            "market_condition": condition,
            "condition_label": condition_label,
            "condition_color": condition_color,
            "buy_score": buy_score,
            "educational_points": educational_points,
            "risk_warnings": risk_warnings,
            "considerations": considerations,
            "strategies": strategies,
            "disclaimer": self.DISCLAIMER,
            "generated_at": datetime.now().isoformat()
        }
    
    def _get_educational_points(
        self, buy_score: int, asset_name: str, trend: str, volatility: str
    ) -> list:
        """Educational information about current market conditions."""
        points = []
        
        # Score interpretation
        if buy_score >= 75:
            points.append({
                "icon": "📊",
                "title": "Điểm số thuận lợi",
                "content": f"AI Buy Score là {buy_score}/100, cho thấy nhiều yếu tố tích cực đang hội tụ."
            })
        elif buy_score >= 55:
            points.append({
                "icon": "📊",
                "title": "Điểm số khá tốt",
                "content": f"AI Buy Score là {buy_score}/100, một số yếu tố đang hỗ trợ {asset_name}."
            })
        elif buy_score >= 40:
            points.append({
                "icon": "📊",
                "title": "Điểm số trung lập",
                "content": f"AI Buy Score là {buy_score}/100, thị trường đang trong giai đoạn quan sát."
            })
        else:
            points.append({
                "icon": "📊",
                "title": "Điểm số thấp",
                "content": f"AI Buy Score là {buy_score}/100, nhiều yếu tố chưa hỗ trợ {asset_name}."
            })
        
        # Trend information
        if trend == "up":
            points.append({
                "icon": "📈",
                "title": "Xu hướng dự đoán",
                "content": f"Mô hình AI dự đoán giá {asset_name} có xu hướng tăng trong 7 ngày tới."
            })
        elif trend == "down":
            points.append({
                "icon": "📉",
                "title": "Xu hướng dự đoán",
                "content": f"Mô hình AI dự đoán giá {asset_name} có xu hướng giảm trong 7 ngày tới."
            })
        else:
            points.append({
                "icon": "➡️",
                "title": "Xu hướng dự đoán",
                "content": f"Mô hình AI dự đoán giá {asset_name} sẽ ổn định trong 7 ngày tới."
            })
        
        # Volatility information
        if volatility == "high":
            points.append({
                "icon": "⚡",
                "title": "Biến động cao",
                "content": "Thị trường đang có biến động mạnh, giá có thể thay đổi nhanh."
            })
        elif volatility == "low":
            points.append({
                "icon": "🔒",
                "title": "Biến động thấp",
                "content": "Thị trường tương đối ổn định, ít biến động bất thường."
            })
        
        return points
    
    def _get_risk_warnings(self, volatility: str, asset_name: str) -> list:
        """Always present risk warnings."""
        warnings = [
            {
                "icon": "⚠️",
                "title": "Rủi ro thị trường",
                "content": f"Giá {asset_name} có thể tăng hoặc giảm bất ngờ do nhiều yếu tố không lường trước được."
            },
            {
                "icon": "💸",
                "title": "Rủi ro tài chính",
                "content": "Chỉ đầu tư số tiền bạn có thể chấp nhận mất mà không ảnh hưởng đến cuộc sống."
            }
        ]
        
        if volatility == "high":
            warnings.append({
                "icon": "🌊",
                "title": "Biến động cao",
                "content": "Thị trường đang biến động mạnh, rủi ro tăng cao. Cân nhắc kỹ trước khi hành động."
            })
        
        return warnings
    
    def _get_considerations(
        self, condition: str, asset_name: str, user_goal: Optional[str], trend: str
    ) -> list:
        """Points to consider when making decisions."""
        considerations = []
        
        # General considerations
        considerations.append({
            "icon": "🎯",
            "title": "Xác định mục tiêu",
            "content": "Bạn mua để tích trữ dài hạn (>6 tháng) hay đầu cơ ngắn hạn? Mục tiêu khác nhau cần chiến lược khác nhau."
        })
        
        considerations.append({
            "icon": "💰",
            "title": "Ngân sách hợp lý",
            "content": "Chỉ sử dụng tiền nhàn rỗi, không vay nợ để đầu tư vào kim loại quý."
        })
        
        # Condition-specific considerations
        if condition == "favorable":
            considerations.append({
                "icon": "📅",
                "title": "Thời điểm mua",
                "content": "Nhiều yếu tố đang thuận lợi, nhưng hãy cân nhắc mua phân đợt để giảm rủi ro."
            })
        elif condition == "unfavorable":
            considerations.append({
                "icon": "⏳",
                "title": "Kiên nhẫn chờ đợi",
                "content": "Điều kiện chưa thuận lợi, có thể chờ thêm để tìm thời điểm tốt hơn."
            })
        
        # Trend-specific
        if trend == "down":
            considerations.append({
                "icon": "📉",
                "title": "Xu hướng giảm",
                "content": "Dự đoán giá giảm, nếu mua nên chuẩn bị tâm lý giữ dài hạn."
            })
        
        return considerations
    
    def _get_strategy_education(
        self, condition: str, volatility: str, user_goal: Optional[str]
    ) -> list:
        """Educational content about investment strategies (NOT advice)."""
        strategies = []
        
        # Dollar Cost Averaging (DCA)
        strategies.append({
            "name": "Mua phân đợt (DCA)",
            "description": "Chia nhỏ số tiền, mua định kỳ (ví dụ: mỗi tuần/tháng) để giảm rủi ro mua đỉnh.",
            "suitable_for": "Người tích trữ dài hạn, muốn giảm rủi ro biến động."
        })
        
        # Lump sum
        strategies.append({
            "name": "Mua một lần",
            "description": "Mua toàn bộ số lượng cùng lúc khi điều kiện rất thuận lợi.",
            "suitable_for": "Người có kinh nghiệm, tự tin về phân tích thị trường."
        })
        
        # Wait and see
        if condition in ["unfavorable", "neutral"]:
            strategies.append({
                "name": "Chờ đợi quan sát",
                "description": "Theo dõi thị trường, chờ điều kiện thuận lợi hơn trước khi hành động.",
                "suitable_for": "Người không vội, muốn tìm thời điểm tối ưu."
            })
        
        return strategies


# Singleton instance
action_recommendation_generator = ActionRecommendationGenerator()


def generate_action_recommendation(
    buy_score: int,
    asset_type: str = "silver",
    prediction_trend: str = "up",
    volatility: str = "medium",
    user_goal: Optional[str] = None,
) -> Dict:
    """
    Generate market condition assessment and educational guidance.
    
    IMPORTANT: This is NOT investment advice. For educational purposes only.
    
    Args:
        buy_score: AI Buy Score (0-100)
        asset_type: "gold" or "silver"
        prediction_trend: "up", "down", or "stable"
        volatility: "low", "medium", "high"
        user_goal: Optional user goal ("accumulate" or "trade")
        
    Returns:
        Dict with market condition assessment and educational content
    """
    return action_recommendation_generator.generate_recommendation(
        buy_score=buy_score,
        asset_type=asset_type,
        prediction_trend=prediction_trend,
        volatility=volatility,
        user_goal=user_goal,
    )
