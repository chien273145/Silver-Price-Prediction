# 🕐 Hướng Dẫn Thiết Lập Cron Job Để Cập Nhật Dữ Liệu Hàng Ngày

Sử dụng [cron-job.org](https://cron-job.org) (miễn phí) để tự động gọi API cập nhật dữ liệu mỗi ngày.

---

## 📋 Bước 1: Đăng ký tài khoản

1. Truy cập https://cron-job.org
2. Click **Sign Up** / **Create Account**
3. Điền thông tin và xác nhận email

---

## 🔧 Bước 2: Tạo Cron Job Cập Nhật Dữ Liệu Hàng Ngày

1. Đăng nhập và vào **Dashboard**
2. Click **Create Cronjob**
3. Điền thông tin:

| Field | Value |
|-------|-------|
| **Title** | Silver Price Daily Update |
| **URL** | `https://silver-price-prediction.onrender.com/api/update-daily` |
| **Schedule** | Every day at 8:00 AM |
| **Request Method** | POST |
| **Timezone** | Asia/Ho_Chi_Minh (UTC+7) |

4. Click **Create**

---

## 🔄 Bước 3: Tạo Cron Job Giữ App Hoạt Động

Render Free Tier sẽ "ngủ" sau 15 phút không hoạt động. Để giữ app luôn "thức":

1. **Create Cronjob** với thông tin:

| Field | Value |
|-------|-------|
| **Title** | Keep App Alive |
| **URL** | `https://silver-price-prediction.onrender.com/api/health` |
| **Schedule** | Every 14 minutes |
| **Request Method** | GET |

2. Click **Create**

---

## ✅ Kiểm tra

- Sau khi thiết lập, các job sẽ tự động chạy theo schedule
- Có thể click **Execute Now** để test ngay
- Xem lịch sử thực thi trong tab **History**

---

## 📊 API Endpoints

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/api/update-daily` | POST | Cập nhật dữ liệu giá mới nhất |
| `/api/data-status` | GET | Kiểm tra trạng thái dataset |
| `/api/health` | GET | Health check (giữ app thức) |

---

## 💡 Lưu ý

- **Free tier** của cron-job.org cho phép tối đa 10 cronjobs
- Thời gian tốt nhất để cập nhật là **8:00 AM GMT+7** (sau khi thị trường Mỹ đóng cửa)
- Thị trường bạc không giao dịch vào cuối tuần, nên dữ liệu Thứ 7-CN sẽ giống nhau
