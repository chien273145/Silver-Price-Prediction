# 🕐 Hướng Dẫn Thiết Lập Cron Job

Sử dụng [cron-job.org](https://cron-job.org) (miễn phí) để tự động cập nhật dữ liệu mỗi ngày.

---

## 📋 Bước 1: Đăng ký tài khoản

1. Truy cập https://cron-job.org
2. Click **Sign Up** → Điền thông tin và xác nhận email

---

## 🔧 Bước 2: Cron Job Cập Nhật Giá Bạc

| Field | Value |
|-------|-------|
| **Title** | Silver Price Daily Update |
| **URL** | `https://silver-price-prediction.onrender.com/api/update-daily` |
| **Schedule** | Every day at 8:00 AM |
| **Method** | POST |
| **Timezone** | Asia/Ho_Chi_Minh |

---

## 🌐 Bước 3: Cron Job Cập Nhật External Data (Gold/DXY/VIX)

| Field | Value |
|-------|-------|
| **Title** | External Data Update |
| **URL** | `https://silver-price-prediction.onrender.com/api/update-external` |
| **Schedule** | Every day at 8:05 AM |
| **Method** | POST |
| **Timezone** | Asia/Ho_Chi_Minh |

> ⚠️ Chạy sau 5 phút để đảm bảo giá bạc đã được cập nhật trước

---

## 🔄 Bước 4: Cron Job Giữ App Hoạt Động

| Field | Value |
|-------|-------|
| **Title** | Keep App Alive |
| **URL** | `https://silver-price-prediction.onrender.com/api/health` |
| **Schedule** | Every 14 minutes |
| **Method** | GET |

---

## 📊 API Endpoints

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/api/update-daily` | POST | Cập nhật giá bạc |
| `/api/update-external` | POST | Cập nhật Gold, DXY, VIX |
| `/api/data-status` | GET | Kiểm tra trạng thái |
| `/api/health` | GET | Health check |

---

## ✅ Thứ tự Cron Jobs (quan trọng)

```
8:00 AM  → update-daily (giá bạc)
8:05 AM  → update-external (Gold/DXY/VIX)
Every 14m → health (giữ app thức)
```

---

## 💡 Lưu ý

- **Free tier** cron-job.org: tối đa 10 cronjobs
- Thời gian tốt nhất: **8:00 AM GMT+7** (sau khi thị trường Mỹ đóng)
- Cuối tuần không có dữ liệu mới
