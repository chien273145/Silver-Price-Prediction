# 🥈 Silver Price Prediction System

Hệ thống AI dự đoán giá bạc 7 ngày sử dụng **Machine Learning (Ridge Regression)** tích hợp Real-time Data từ Yahoo Finance.

## ✨ Tính năng

- 🤖 **AI Model**: Ridge Regression (R² = 0.96, MAPE = 3.37%) - Dự đoán cực nhanh và chính xác.
- 💱 **Hỗ trợ VND**: Tự động chuyển đổi từ USD sang VND (bao gồm Premium thị trường VN ~24%).
- 📊 **Dashboard đẹp**: Giao diện web hiện đại với biểu đồ tương tác Chart.js.
- 🔄 **Real-time**: Tự động lấy giá Spot Silver (XAG/USD) mới nhất từ Yahoo Finance.
- 🌐 **REST API**: FastAPI backend mạnh mẽ.
- ☁️ **Deploy Ready**: Sẵn sàng deploy miễn phí lên Render.com.

## 📁 Cấu trúc dự án

```
Predict_Gia_Bac/
├── dataset/                    # Dữ liệu giá bạc
├── src/                        # Source code AI
│   ├── unified_predictor.py   # Predictor (Ridge + Realtime)
│   ├── train_ridge.py         # Training script
├── backend/                    # FastAPI Backend
│   ├── app.py                 # API endpoints
│   └── realtime_data.py       # Yahoo Finance integration
├── frontend/                   # Web Dashboard
├── models/                     # Trained models
├── requirements.txt
├── render.yaml                # Render config
├── DEPLOY_GUIDE.md            # Hướng dẫn Deploy
└── push_to_github.bat         # Script push code tự động
```

## 🚀 Hướng dẫn cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy server (Local)

```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
Truy cập Dashboard: http://localhost:8000

## ☁️ Hướng dẫn Deploy (GitHub & Render)

### Bước 1: Push code lên GitHub
Nếu bạn chưa cài Git, hãy tải và cài đặt Git. Sau đó chạy file script tự động:

1. Chạy file `push_to_github.bat`
2. Đăng nhập GitHub khi được yêu cầu

### Bước 2: Deploy lên Render.com
Xem hướng dẫn chi tiết từng bước tại file [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md).

## 📡 API Endpoints

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/api/predict` | GET | Dự đoán giá 7 ngày (tự động fetch realtime) |
| `/api/historical` | GET | Dữ liệu lịch sử |
| `/api/realtime` | GET | Giá bạc & tỷ giá hiện tại |
| `/api/metrics` | GET | Độ chính xác mô hình |

## 🧠 Về mô hình AI

**Ridge Regression** được chọn thay thế LSTM vì các ưu điểm vượt trội:
- **Độ chính xác cao hơn**: R²=0.96 vs R²=0.56 (LSTM)
- **Tốc độ**: Train < 5 giây, Dự đoán < 0.1 giây
- **Features**: 110 chỉ số kỹ thuật (RSI, MAs, Bollinger Bands, Volatility...)

## 💱 Định giá Việt Nam

Hệ thống điều chỉnh giá theo thực tế thị trường Việt Nam:
```
Giá VND = Giá USD × 1.20565 × Tỷ giá × 1.24 (Vietnam Premium)
```
*Premium 24% phản ánh chi phí nhập khẩu, thuế và biên lợi nhuận tại Việt Nam.*

## 📝 License
MIT License
