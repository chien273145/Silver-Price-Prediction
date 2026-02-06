# DuBaoVangBac.com - AI Gold & Silver Price Prediction

Hệ thống AI dự đoán giá **Vàng** và **Bạc** 7 ngày tới sử dụng **Ridge Regression** với dữ liệu real-time từ Yahoo Finance.

**Live**: [dubaovangbac.com](https://dubaovangbac.com)

## Tính Năng

- **3 Mô hình AI**: Silver (R²=0.96), World Gold (R²=0.91), Vietnam SJC Gold (Transfer Learning)
- **110+ Features**: RSI, MAs, Bollinger Bands, ATR, DXY, VIX, Oil, US10Y...
- **Bảng giá trong nước**: Scraping real-time từ BTMC, Phú Quý, Ancarat, WebGia
- **Portfolio Manager**: Sổ tài sản + AI Time Machine dự báo tương lai
- **Fear & Greed Index**: Chỉ số tâm lý thị trường
- **AI Buy Score**: Điểm mua/bán 0-100
- **Blog SEO**: 5 bài phân tích chuyên sâu
- **Auto Retrain**: GitHub Actions retrain mỗi ngày 00:00 VN

## Cấu Trúc

```
├── backend/
│   ├── app.py                  # FastAPI server (40+ endpoints)
│   └── realtime_data.py        # Yahoo Finance data fetcher
├── src/
│   ├── enhanced_predictor.py   # Silver model (Ridge Regression)
│   ├── gold_predictor.py       # World Gold model
│   ├── vietnam_gold_predictor.py # VN SJC Gold (Transfer Learning)
│   ├── buy_score.py            # AI Buy Score calculator
│   ├── time_machine.py         # Portfolio future prediction
│   ├── news_sentiment.py       # News sentiment analysis
│   └── scrapers/               # Vietnamese price scrapers
├── scripts/
│   └── daily_update.py         # Daily retraining pipeline
├── frontend/
│   ├── index.html, silver.html, gold.html  # Dashboards
│   ├── app.js, app-gold.js, app-silver.js  # Frontend logic
│   ├── styles.css              # UI styling
│   └── blog/                   # 5 SEO blog posts
├── models/                     # Trained ML models (.pkl)
├── dataset/                    # Training data (.csv)
├── .github/workflows/          # GitHub Actions CI/CD
├── requirements.txt
├── render.yaml                 # Render.com deployment config
└── Procfile
```

## Cài Đặt Local

```bash
pip install -r requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Truy cập: http://localhost:8000

## API Endpoints

| Endpoint | Mô tả |
|----------|-------|
| `/api/predict` | Dự đoán giá bạc 7 ngày |
| `/api/gold/predict` | Dự đoán giá vàng thế giới 7 ngày |
| `/api/gold-vn/predict` | Dự đoán giá vàng SJC Việt Nam |
| `/api/historical` | Dữ liệu lịch sử |
| `/api/prices/local` | Bảng giá vàng trong nước |
| `/api/market-analysis` | Phân tích thị trường AI |
| `/api/fear-greed` | Fear & Greed Index |
| `/api/buy-score` | AI Buy Score |
| `/api/news` | Tin tức thị trường |
| `/api/health` | Health check |
| `/docs` | Swagger API Documentation |

## Tech Stack

- **Backend**: Python 3.11, FastAPI, scikit-learn, yfinance
- **Frontend**: HTML/CSS/JS, Chart.js
- **ML**: Ridge Regression, Transfer Learning, PCA
- **Deploy**: Render.com (Free), GitHub Actions, UptimeRobot
- **SEO**: Sitemap, robots.txt, Open Graph, Structured Data

## License

MIT License
