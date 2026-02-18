# Bug Fixes - Đánh Giá Website

## 1. Lọc tin tức theo chủ đề vàng/bạc
- [ ] Sửa `src/news_sentiment.py`: thêm keyword filter cho gold/silver
- [ ] Test lại `/api/news`

## 2. Giá vàng thế giới quy đổi sai (156tr vs ~75tr)
- [ ] Kiểm tra `backend/realtime_data.py` hoặc scraper
- [ ] Sửa công thức quy đổi oz → lượng

## 3. Buy Score trả về silver khi hỏi gold
- [ ] Kiểm tra `backend/app.py` endpoint `/api/buy-score`
- [ ] Sửa routing asset gold/silver

## 4. Sửa bug Reasoning "VIX=0.0"
- [ ] Kiểm tra `src/reasoning_generator.py`
- [ ] Sửa logic VIX display

## 5. Deploy
- [x] Git push sau khi sửa xong

## 6. Daily Data Scraping (Auto-Update)
- [x] Create `append_daily_prices.py` (Gold SJC + Silver Yahoo)
- [x] Fix Webgia SJC scraper (handle unit conversion issue)
- [x] Update GitHub Actions workflow (17:00 SA scrape job)
- [x] Verify local execution and data appending
