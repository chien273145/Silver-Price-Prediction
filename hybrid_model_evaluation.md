# Đánh giá Mô hình Hybrid AI (Technical + LLM Sentiment)
## So sánh với State-of-the-Art trong dự báo giá Vàng/Bạc

---

## 1. Tổng quan: Ý tưởng của bạn đứng ở đâu?

### 1.1 Bản chất kiến trúc

Mô hình bạn đề xuất thuộc loại **"Hybrid Sentiment-Enhanced Forecasting"** — một hướng nghiên cứu đang rất hot. Công thức cốt lõi:

```
Giá_Cuối = P_base × (1 + α × S_score)
```

Ý tưởng này **KHÔNG mới** trong học thuật. Nó đã được nghiên cứu rộng rãi từ 2020 đến nay dưới nhiều biến thể. Tuy nhiên, **cách bạn áp dụng cho thị trường Việt Nam (SJC) với nguồn tin tiếng Việt** thì có tính mới và thực tiễn cao.

### 1.2 Bảng so sánh tổng thể

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BẢN ĐỒ CÁC MÔ HÌNH DỰ BÁO GIÁ VÀNG              │
├──────────────────┬──────────────┬────────────┬──────────────────────────┤
│ Cấp độ           │ Mô hình      │ Complexity │ Typical MAPE (1-day)    │
├──────────────────┼──────────────┼────────────┼──────────────────────────┤
│ 1. Baseline      │ ARIMA/GARCH  │ ★☆☆☆☆     │ 1.5 - 3.0%             │
│                  │ Random Walk  │ ★☆☆☆☆     │ 0.8 - 1.5%             │
│                  │ Ridge/SVR    │ ★★☆☆☆     │ 0.8 - 1.5%             │
├──────────────────┼──────────────┼────────────┼──────────────────────────┤
│ 2. Deep Learning │ LSTM         │ ★★★☆☆     │ 0.5 - 1.2%             │
│                  │ CNN-BiLSTM   │ ★★★★☆     │ 0.3 - 0.8%             │
│                  │ Transformer  │ ★★★★☆     │ 0.3 - 0.7%             │
├──────────────────┼──────────────┼────────────┼──────────────────────────┤
│ 3. Hybrid        │ LSTM+Sent.   │ ★★★★☆     │ 0.3 - 0.7%             │
│   (Sentiment)    │ Transf+FinBE │ ★★★★★     │ 0.2 - 0.5%             │
│                  │ LLM+XGBoost  │ ★★★★★     │ 0.3 - 0.6%             │
├──────────────────┼──────────────┼────────────┼──────────────────────────┤
│ 4. SOTA Fusion   │ LSTM-Transf  │ ★★★★★     │ 0.2 - 0.5%             │
│                  │ -XGBoost     │            │                         │
│                  │ w/ Multi-    │            │                         │
│                  │ source Sent. │            │                         │
├──────────────────┼──────────────┼────────────┼──────────────────────────┤
│ ★ Mô hình CỦA   │ Ridge+LLM    │ ★★★☆☆     │ 0.5 - 1.0% (ước tính)  │
│   BẠN            │ Sentiment    │            │                         │
└──────────────────┴──────────────┴────────────┴──────────────────────────┘
```

**Vị trí của bạn: Nằm giữa Level 2 và Level 3.** Ý tưởng đúng hướng (Level 3-4) nhưng implementation hiện tại (Ridge thay vì LSTM/Transformer) giới hạn performance ceiling.

---

## 2. Đánh giá chi tiết từng thành phần

### 2.1 "Bộ não trái" - Technical Model (Ridge/XGBoost)

**Điểm: 6/10**

| Tiêu chí | Đánh giá |
|---|---|
| Ridge Regression | Ổn định, ít overfit, nhưng chỉ linear — miss phi tuyến |
| XGBoost (nếu thêm) | Tốt hơn nhiều — nonlinear, feature interaction |
| So với SOTA | LSTM-Transformer-XGBoost fusion đang là SOTA (MDPI 2025, đạt R² > 0.99 trên gold futures Thượng Hải) |

**Nghiên cứu benchmark:**
- CNN-BiLSTM đạt MAE = 1, accuracy 97% trên gold price prediction (ResearchGate 2024)
- LSTM-Transformer-XGBoost 3-stage fusion trên Shanghai Gold Futures chọn 6 key drivers từ 36 biến (MDPI, May 2025)
- Framework hybrid (MDPI, June 2025) trên 11 năm data kết hợp financial + macro + sentiment — advanced supervised ML vượt trội econometric models

**Khoảng cách so với SOTA:** Ridge/XGBoost bị giới hạn bởi:
- Không capture temporal dependencies (sequence pattern)
- Không có attention mechanism cho long-range relationships
- Feature interaction chỉ ở mức thủ công

### 2.2 "Bộ não phải" - LLM Sentiment Analysis

**Điểm: 7.5/10 (ý tưởng) | 4/10 (implementation details)**

**Ý tưởng rất tốt vì:**
- Research mới nhất (2025) confirm: sentiment signals giảm forecasting errors đáng kể so với baseline LSTM
- FinBERT-based sentiment + LSTM đã demo ~184% net profit trong 1 tháng backtest trên Gold/USD (ScienceDirect, Sept 2025)
- LLM-generated alphas outperform human-defined alphas trong stock prediction (arXiv, Aug 2025)

**Nhưng bạn đang thiếu nhiều chi tiết quan trọng:**

#### Vấn đề 1: Timing Gap (Nghiêm trọng nhất)

```
Thực tế:  Tin tức → Thị trường phản ứng → Giá thay đổi
Thời gian:  0 phút    5-30 phút         đã xong

Mô hình:  Thu thập tin → LLM phân tích → Điều chỉnh dự báo
Thời gian:  +1-24 giờ     +30 giây         đã muộn
```

Ví dụ cú sập bạc về 70.2 mà bạn đề cập: khi báo chí đưa tin "bán tháo bạc", giá ĐÃ sập rồi. Tin tức là **lagging indicator**, không phải leading indicator. Thị trường di chuyển TRƯỚC khi tin tức được viết.

Ngoại lệ: Tin chính sách (Fed tăng lãi suất scheduled) có thể predict trước.

#### Vấn đề 2: LLM Output Variability

Research mới nhất (Frontiers in AI, July 2025) chỉ ra **Model Variability Problem (MVP)**: cùng 1 tin tức, LLM có thể cho score khác nhau mỗi lần chạy do stochastic inference, prompt sensitivity, và training data bias. Trong financial application, "even a few-tenth shift in polarity can move millions of dollars."

Cụ thể:
- Cùng headline "Fed holds rates steady" → LLM có thể cho +2 hoặc -1 tùy context window
- Temperature > 0 tạo randomness trong output
- Prompt phrasing thay đổi nhỏ → score thay đổi lớn

#### Vấn đề 3: Công thức tuyến tính quá đơn giản

```
Giá_Cuối = P_base × (1 + α × S_score)
```

Vấn đề:
- **α cố định (0.005)**: Tác động của tin tức thay đổi theo regime. Trong crisis, sentiment impact có thể 5-10x bình thường. Trong thị trường sideways, gần như 0.
- **Tuyến tính**: Sentiment +8 không có nghĩa là impact gấp đôi +4. Có saturation effect.
- **Thiếu decay**: Tin tức mất tác động theo thời gian (sentiment half-life ~1-3 ngày). Score hôm nay không nên ảnh hưởng dự báo ngày thứ 7 với cùng intensity.

#### Vấn đề 4: Nguồn tin tiếng Việt

- CafeF, SJC, PNJ thường đăng tin AFTER THE FACT (giá đã thay đổi rồi)
- Kitco, Bloomberg có tin nhanh hơn nhưng bằng tiếng Anh → LLM xử lý tốt hơn
- Không có equivalent của Bloomberg Terminal realtime cho thị trường VN

### 2.3 "The Fusion" - Kết hợp hai bộ não

**Điểm: 5/10**

Cách kết hợp multiplicative đơn giản `P × (1 + α × S)` là điểm yếu nhất. State-of-the-art dùng:

| Phương pháp | Mô tả | Complexity |
|---|---|---|
| **Linear (bạn)** | P × (1 + α × S) | ★☆☆☆☆ |
| Feature concatenation | Gộp sentiment score vào feature vector trước khi train | ★★☆☆☆ |
| Attention-based fusion | Model tự học khi nào nên "nghe" sentiment | ★★★★☆ |
| Multi-task learning | Predict price + predict sentiment đồng thời | ★★★★★ |
| Gated fusion | Gate mechanism quyết định weight của sentiment theo context | ★★★★★ |

---

## 3. So sánh trực tiếp với SOTA (2024-2025)

### 3.1 AchillesV11 (ScienceDirect, Sept 2025)
- **Kiến trúc**: LSTM + RSI/EMA + FinBERT sentiment
- **Target**: Gold/USD exchange rate, real-time
- **Kết quả**: ~184% net profit trong 1 tháng backtest
- **Ưu điểm so với mô hình của bạn**: LSTM capture temporal patterns, FinBERT fine-tuned cho finance

### 3.2 LSTM-Transformer-XGBoost Fusion (MDPI, May 2025)
- **Kiến trúc**: 3-stage — XGBoost feature selection → LSTM temporal memory → Transformer multi-head attention
- **Target**: Shanghai Gold Futures
- **Features**: 36 biến → chọn 6 key (NASDAQ, S&P 500, Silver, USD/CNY, Treasury yield, Coal ETF)
- **Ưu điểm**: Multi-scale feature fusion, tự chọn features quan trọng

### 3.3 Hybrid Framework (MDPI, June 2025)
- **Kiến trúc**: Econometric + ML + DL + Meta-model
- **Target**: Gold futures 2014-2024
- **Features**: Financial + macroeconomic + sentiment indicators
- **Kết quả**: Advanced supervised > traditional across diverse market conditions

### 3.4 LLM-Generated Formulaic Alpha (arXiv, Aug 2025)
- **Kiến trúc**: LLM generates alpha formulas → Transformer predicts
- **Insight chính**: LLM-generated alphas > human-defined alphas vì tự adapt, chống alpha decay
- **Ứng dụng**: LLM không chỉ cho sentiment score mà TẠO RA features/formulas mới

---

## 4. Đánh giá tổng thể mô hình Hybrid AI của bạn

### Bảng điểm

```
┌──────────────────────────────┬───────┬─────────────────────────────────────┐
│ Tiêu chí                     │ Điểm  │ Nhận xét                            │
├──────────────────────────────┼───────┼─────────────────────────────────────┤
│ Tính sáng tạo / Ý tưởng     │ 8/10  │ Đúng hướng SOTA, fit cho VN market  │
│ Kiến trúc Technical Model    │ 6/10  │ Ridge/XGB ok nhưng thua LSTM/Transf │
│ Kiến trúc Sentiment Model    │ 7/10  │ LLM sentiment hợp lý, cần FinBERT  │
│ Fusion Strategy              │ 4/10  │ Multiplicative quá đơn giản         │
│ Xử lý Timing/Latency        │ 3/10  │ Chưa address news lag problem       │
│ Tính khả thi triển khai      │ 7/10  │ Có thể MVP trong 2-4 tuần          │
│ Giá trị cho user (UX)        │ 9/10  │ UI explain rất tốt (hiếm thấy)     │
│ Tính mới cho thị trường VN   │ 8/10  │ Chưa ai làm tương tự ở VN          │
├──────────────────────────────┼───────┼─────────────────────────────────────┤
│ TỔNG                         │ 6.5/10│ Ý tưởng tốt, execution cần nâng cấp│
└──────────────────────────────┴───────┴─────────────────────────────────────┘
```

---

## 5. Về ví dụ "cú sập 70.2" — Phân tích thực tế

Bạn claim mô hình LLM sẽ detect tin xấu TRƯỚC cú sập. Đây là phân tích thực tế:

### Kịch bản thực tế (không như lý tưởng)

```
Timeline thực tế của một cú sập:

T-24h:  Thị trường bình thường. Không có tin đặc biệt.
        → LLM score: 0 (neutral)
        → Mô hình dự báo: 84-85 (sai)

T-2h:   Institutional selling bắt đầu (chỉ thấy trên order flow)
        → Chưa có tin tức nào
        → LLM score: 0 (vẫn neutral)

T-0:    Giá sập mạnh từ 84 → 75
        → Reuters/Bloomberg bắt đầu viết tin
        → Nhưng giá ĐÃ sập rồi

T+1h:   Tin tức: "Bạc bị bán tháo do dữ liệu việc làm Mỹ mạnh"
        → LLM score: -8
        → Mô hình điều chỉnh: 75 × (1 + 0.005 × (-8)) = 72
        → Nhưng giá đã ở 70.2 rồi — vẫn sai!

T+24h:  CafeF, SJC đưa tin (tiếng Việt)
        → LLM phân tích tin Việt
        → Nhưng đã quá muộn 1 ngày
```

### Khi nào LLM Sentiment THỰC SỰ hữu ích?

LLM sentiment **có giá trị thực sự** cho:
1. **Scheduled events**: FOMC meetings, Non-Farm Payrolls, CPI releases — LLM có thể phân tích expectation trước event
2. **Trending narratives**: "De-dollarization" trend, Central bank buying trend — slow-moving narratives kéo dài tuần/tháng
3. **Regime detection**: Phân biệt "risk-on" vs "risk-off" regime từ tổng hợp tin tức — không predict giá chính xác nhưng cho context đúng
4. **User explanation**: Giá trị lớn nhất là GIẢI THÍCH cho user tại sao giá di chuyển (đây là competitive advantage thực sự!)

---

## 6. Khuyến nghị cải thiện cụ thể

### 6.1 Quick Wins (1-2 tuần)

**A. Thay công thức fusion:**
```python
# ❌ CŨ: Multiplicative đơn giản
price_final = p_base * (1 + alpha * s_score)

# ✅ MỚI: Regime-adaptive với decay
def compute_adjusted_price(p_base, s_score, s_age_hours, volatility_regime):
    # Sentiment decay: mất 50% tác động mỗi 24h
    decay = 0.5 ** (s_age_hours / 24)
    effective_score = s_score * decay
    
    # Regime-adaptive alpha: cao hơn khi volatile
    alpha = 0.003 * (1 + volatility_regime)  # volatility_regime: 0=calm, 1=normal, 2=crisis
    
    # Asymmetric impact: tin xấu tác động mạnh hơn tin tốt
    if effective_score < 0:
        alpha *= 1.5  # Tin xấu impact 1.5x tin tốt
    
    # Saturation: ±10 score nhưng max impact ±5%
    capped_impact = np.tanh(alpha * effective_score) * 0.05
    
    return p_base * (1 + capped_impact)
```

**B. Dùng FinBERT thay vì general LLM:**
```python
# FinBERT: Pre-trained cho financial text, deterministic output
# pip install transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
# Output: {positive: 0.85, negative: 0.10, neutral: 0.05}
# Deterministic — không có variability problem
```

**C. Multi-source sentiment aggregation:**
```python
# Thay vì 1 LLM call → nhiều nguồn + aggregate
sentiment_sources = {
    'finbert_kitco': finbert_score(kitco_headlines),        # weight: 0.3
    'finbert_reuters': finbert_score(reuters_headlines),     # weight: 0.3
    'llm_vietnamese': llm_score(cafef_headlines),            # weight: 0.2
    'vix_implied': normalize(vix_level),                     # weight: 0.1
    'gpr_index': normalize(gpr_level),                       # weight: 0.1
}
# Weighted aggregate → more robust than single LLM
final_score = sum(score * weight for score, weight in sentiment_sources.items())
```

### 6.2 Medium-term (1-2 tháng)

**D. Nâng cấp Technical Model:**
```
Ridge → XGBoost → LSTM → (optional) Transformer

Cụ thể:
1. Giữ Ridge làm baseline/ensemble member
2. Thêm XGBoost cho nonlinear patterns
3. Thêm LSTM (2 layers, 64 units) cho temporal dependencies
4. Ensemble 3 models: w1*Ridge + w2*XGBoost + w3*LSTM
```

**E. Feature-level fusion thay vì score-level:**
```python
# ❌ Score-level: LLM → score → multiply with price
# ✅ Feature-level: LLM → embedding → concatenate with technical features → model

# Sentiment embedding (768-dim từ FinBERT) → PCA → 10 features
# Ghép vào technical features trước khi train LSTM
features = np.concatenate([
    technical_features,    # RSI, MACD, lags...
    sentiment_embedding,   # 10-dim PCA of FinBERT output
    macro_features,        # DXY, VIX, yield...
], axis=1)

model.fit(features, target_returns)
```

### 6.3 Long-term (3-6 tháng)

**F. Attention-based fusion (SOTA approach):**
```python
# Self-attention mechanism quyết định weight cho mỗi signal source
# Tự động học: khi nào nghe technical, khi nào nghe sentiment

class AttentionFusion(nn.Module):
    def __init__(self):
        self.technical_encoder = LSTM(...)
        self.sentiment_encoder = TransformerEncoder(...)
        self.cross_attention = MultiHeadAttention(...)
        self.output = Linear(...)
    
    def forward(self, technical_seq, sentiment_seq):
        tech_hidden = self.technical_encoder(technical_seq)
        sent_hidden = self.sentiment_encoder(sentiment_seq)
        
        # Cross-attention: technical attends to sentiment
        fused = self.cross_attention(
            query=tech_hidden,
            key=sent_hidden,
            value=sent_hidden
        )
        return self.output(fused)
```

---

## 7. Kết luận

### Bạn đang làm đúng:
- Hướng đi Hybrid AI đúng trend SOTA
- UI giải thích cho user là competitive advantage hiếm có
- Vietnam-specific (SJC, tin tiếng Việt, mùa vụ) là niche chưa ai khai thác

### Bạn cần cải thiện:
- **Fusion strategy** là điểm yếu nhất — cần nâng từ linear multiplication lên feature-level fusion
- **Timing problem** cần được address — sentiment là lagging, không leading
- **LLM variability** — chuyển sang FinBERT (deterministic) cho scoring, dùng LLM chỉ cho explanation text
- **Technical model** — Ridge là baseline tốt nhưng ceiling thấp, cần LSTM hoặc XGBoost

### Roadmap đề xuất:

```
Tháng 1: FinBERT + XGBoost + Regime-adaptive formula     → Beat baseline 5-10%
Tháng 2: LSTM technical model + Feature-level fusion      → Beat baseline 15-20%
Tháng 3: Multi-source sentiment + Attention fusion        → Approach SOTA
Tháng 4: Walk-forward validation + Production deployment  → Prove real-world value
```
