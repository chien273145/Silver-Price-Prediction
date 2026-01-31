# 🥈 Hướng Dẫn Deploy Web Dự Đoán Giá Bạc Lên Render.com

## 📋 Yêu cầu
- Tài khoản GitHub (miễn phí)
- Tài khoản Render.com (miễn phí)

---

## 🚀 Bước 1: Đẩy code lên GitHub

### 1.1. Tạo repository mới trên GitHub
1. Truy cập https://github.com/new
2. Đặt tên: `silver-price-prediction`
3. Chọn **Private** hoặc **Public**
4. Nhấn **Create repository**

### 1.2. Push code từ máy tính

```bash
# Mở terminal trong thư mục dự án
cd C:\Users\admin\Predict_Gia_Bac

# Khởi tạo git
git init

# Thêm tất cả files
git add .

# Commit
git commit -m "Initial commit - Silver Price Prediction"

# Thêm remote (thay YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/silver-price-prediction.git

# Push lên GitHub
git branch -M main
git push -u origin main
```

---

## 🌐 Bước 2: Deploy lên Render.com

### 2.1. Đăng ký Render
1. Truy cập https://render.com
2. Nhấn **Get Started for Free**
3. Đăng nhập bằng **GitHub**

### 2.2. Tạo Web Service
1. Trong Dashboard, nhấn **New +** → **Web Service**
2. Chọn **Connect a repository**
3. Tìm và chọn repo `silver-price-prediction`
4. Cấu hình:
   - **Name**: `silver-price-prediction`
   - **Region**: `Singapore` (gần Việt Nam nhất)
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`

5. Nhấn **Create Web Service**

### 2.3. Chờ deploy
- Render sẽ tự động build và deploy
- Thời gian: ~3-5 phút
- Khi hoàn thành, bạn sẽ có URL: `https://silver-price-prediction.onrender.com`

---

## ✅ Bước 3: Kiểm tra

Truy cập các URL:
- **Dashboard**: `https://your-app.onrender.com`
- **API Dự đoán**: `https://your-app.onrender.com/api/predict`
- **API Docs**: `https://your-app.onrender.com/docs`

---

## 🔄 Cập nhật tự động

Mỗi khi bạn push code mới lên GitHub, Render sẽ tự động:
1. Pull code mới
2. Build lại
3. Deploy phiên bản mới

---

## 💡 Lưu ý quan trọng

### Free Tier Limitations
- **Spin down**: App sẽ "ngủ" sau 15 phút không hoạt động
- **Cold start**: Lần đầu truy cập sau khi "ngủ" sẽ mất ~30 giây
- **750 giờ/tháng**: Đủ dùng cho cả tháng

### Để giữ app luôn "thức"
Tạo cron job miễn phí tại https://cron-job.org:
1. Đăng ký tài khoản
2. Tạo cron job mới
3. URL: `https://your-app.onrender.com/api/health`
4. Schedule: Every 14 minutes

---

## 📊 Cấu trúc dự án

```
Predict_Gia_Bac/
├── backend/
│   ├── app.py           # FastAPI application
│   └── realtime_data.py # Data fetching
├── src/
│   ├── unified_predictor.py  # Ridge Regression predictor
│   └── train_ridge.py        # Training script
├── models/
│   ├── ridge_models.pkl      # Trained models
│   └── ridge_training_info.json
├── dataset/
│   └── dataset_silver.csv    # Historical data
├── frontend/
│   ├── index.html       # Dashboard
│   ├── styles.css       # Styles
│   └── app.js           # JavaScript
├── requirements.txt     # Dependencies
├── Procfile            # Render start command
├── render.yaml         # Render configuration
└── README.md
```

---

## 🎉 Hoàn thành!

Sau khi deploy, bạn có thể:
1. Chia sẻ link cho mọi người
2. Xem dự đoán giá bạc 7 ngày tới
3. Theo dõi giá thời gian thực

**Chi phí: $0/tháng** 💰
