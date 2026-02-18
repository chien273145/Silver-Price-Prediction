"""
append_daily_prices.py
======================
Lấy giá vàng SJC và bạc Việt Nam cuối ngày, append vào dataset training.

Chạy bởi GitHub Actions lúc 10:00 UTC (17:00 SA Giờ VN) mỗi ngày.

Nguồn dữ liệu:
  - Vàng SJC: webgia.com (scrape HTML) + btmc.vn (fallback)
  - Bạc VN:   giá bạc thế giới (Yahoo SI=F) × tỷ giá USD/VND × hệ số quy đổi

Usage:
    python append_daily_prices.py           # Cả vàng lẫn bạc
    python append_daily_prices.py --gold    # Chỉ vàng
    python append_daily_prices.py --silver  # Chỉ bạc
"""

import os
import sys
import argparse
import re
import time
import traceback
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
sys.path.insert(0, BASE_DIR)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# GOLD SCRAPING — SJC price from webgia.com / btmc.vn
# ══════════════════════════════════════════════════════════════════════════════

def _decode_webgia_price(code: str) -> float:
    """Port of webgia's gm(r) JavaScript obfuscation function."""
    if not code:
        return 0.0
    try:
        clean = re.sub(r'[A-Z]', '', code)
        chars = [chr(int(clean[i:i+2], 16)) for i in range(0, len(clean) - 1, 2)]
        return float("".join(chars).replace(".", ""))
    except Exception:
        return 0.0


def scrape_sjc_from_webgia() -> dict | None:
    """Scrape SJC buy/sell price from webgia.com (handled unit: dong/chi -> *10 -> dong/luong)."""
    url = "https://webgia.com/gia-vang/"
    log(f"[Gold] Scraping webgia.com...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table", class_="table-radius")
        if not table:
            log("[Gold] webgia: table not found")
            return None

        # Helper to parse price input (decoded string or element text)
        def parse_raw_price(val: str | float) -> float:
            if isinstance(val, (int, float)):
                return float(val)
            text = str(val).replace(".", "").replace(",", "").strip()
            # If empty or not digits
            if not text or not re.match(r'^\d+$', text):
                return 0.0
            return float(text)

        for row in table.select("tbody tr"):
            brand_node = row.select_one("td a strong")
            location_node = row.find("th") # e.g. "TP.Hồ Chí Minh", "Miền Bắc"
            
            if not brand_node:
                continue
                
            brand = brand_node.get_text(strip=True).upper()
            if "SJC" not in brand:
                continue

            # Prioritize "Miền Bắc" or "TP.Hồ Chí Minh" rows which usually have standard SJC
            # But we can just take the first valid SJC row we find that matches range
            
            price_cells = row.select("td.text-right")
            if len(price_cells) >= 2:
                # 1. Try 'nb' attribute (obfuscated)
                buy_attr = price_cells[0].get("nb", "")
                sell_attr = price_cells[1].get("nb", "")
                
                buy = 0.0
                sell = 0.0
                
                if buy_attr:
                    buy = _decode_webgia_price(buy_attr)
                else:
                    # 2. Try plain text
                    buy = parse_raw_price(price_cells[0].get_text(strip=True))

                if sell_attr:
                    sell = _decode_webgia_price(sell_attr)
                else:
                    sell = parse_raw_price(price_cells[1].get_text(strip=True))
                
                # UNIT CORRECTION: Webgia uses "đồng / chỉ" (1/10 lượng)
                # If price is ~17M, it means 17M/chỉ -> 170M/lượng
                # Threshold: if < 50,000,000 -> multiply by 10
                if 0 < buy < 50_000_000:
                    buy *= 10
                if 0 < sell < 50_000_000:
                    sell *= 10
                    
                # Sanity check for 2026 prices (SJC ~100M - 300M)
                if buy > 100_000_000 and sell > 100_000_000:
                    log(f"[Gold] webgia SJC: buy={buy:,.0f}, sell={sell:,.0f}")
                    return {"buy_price": buy, "sell_price": sell, "source": "webgia"}

        log("[Gold] webgia: No valid SJC row found (checked plain text & obfuscated).")
        return None
    except Exception as e:
        log(f"[Gold] webgia error: {e}")
        return None


def scrape_sjc_from_btmc() -> dict | None:
    """Scrape SJC buy/sell price from btmc.vn as fallback."""
    url = "https://btmc.vn/gia-vang-hom-nay"
    log(f"[Gold] Scraping btmc.vn (fallback)...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # BTMC shows a table with gold prices
        for row in soup.select("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) < 3:
                continue
            label = cells[0].get_text(strip=True).upper()
            if "SJC" not in label:
                continue
            try:
                buy_text  = cells[1].get_text(strip=True).replace(",", "").replace(".", "")
                sell_text = cells[2].get_text(strip=True).replace(",", "").replace(".", "")
                buy  = float(buy_text)
                sell = float(sell_text)
                # BTMC shows in thousands (e.g. 178000 = 178,000,000 VND)
                if buy < 1_000_000:
                    buy  *= 1000
                    sell *= 1000
                if buy > 50_000_000 and sell > 50_000_000:
                    log(f"[Gold] btmc SJC: buy={buy:,.0f}, sell={sell:,.0f}")
                    return {"buy_price": buy, "sell_price": sell, "source": "btmc"}
            except Exception:
                continue

        log("[Gold] btmc: SJC row not found")
        return None
    except Exception as e:
        log(f"[Gold] btmc error: {e}")
        return None


def scrape_sjc_from_phu_quy() -> dict | None:
    """Scrape SJC price from phuquygold.com.vn as second fallback."""
    url = "https://www.phuquygold.com.vn/gia-vang-hom-nay"
    log(f"[Gold] Scraping phuquygold.com.vn (fallback 2)...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        for row in soup.select("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) < 3:
                continue
            label = cells[0].get_text(strip=True).upper()
            if "SJC" not in label:
                continue
            try:
                buy_text  = re.sub(r"[^\d]", "", cells[1].get_text(strip=True))
                sell_text = re.sub(r"[^\d]", "", cells[2].get_text(strip=True))
                buy  = float(buy_text)
                sell = float(sell_text)
                if buy < 1_000_000:
                    buy  *= 1000
                    sell *= 1000
                if buy > 50_000_000 and sell > 50_000_000:
                    log(f"[Gold] phuquy SJC: buy={buy:,.0f}, sell={sell:,.0f}")
                    return {"buy_price": buy, "sell_price": sell, "source": "phuquy"}
            except Exception:
                continue

        log("[Gold] phuquy: SJC row not found")
        return None
    except Exception as e:
        log(f"[Gold] phuquy error: {e}")
        return None


def get_sjc_price() -> dict | None:
    """Try all sources in order until one succeeds."""
    for scraper_fn in [scrape_sjc_from_webgia, scrape_sjc_from_btmc, scrape_sjc_from_phu_quy]:
        result = scraper_fn()
        if result:
            return result
        time.sleep(2)
    log("[Gold] All scrapers failed. Cannot get SJC price today.")
    return None


def append_gold_price() -> bool:
    """Fetch today's SJC price and append to gold_price_sjc_complete.csv."""
    path = os.path.join(DATASET_DIR, "gold_price_sjc_complete.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    last_date = df["date"].max()
    today = datetime.now().date()

    if last_date.date() >= today:
        log(f"[Gold] Already have data for {today}. Skipping.")
        return False

    # Check if market is open (Mon-Fri)
    if datetime.now().weekday() >= 5:
        log(f"[Gold] Weekend ({datetime.now().strftime('%A')}). Gold market closed. Skipping.")
        return False

    price_data = get_sjc_price()
    if not price_data:
        return False

    buy  = price_data["buy_price"]
    sell = price_data["sell_price"]

    # Sanity check: SJC gold should be between 50M and 300M VND/luong
    if not (50_000_000 < buy < 300_000_000 and 50_000_000 < sell < 300_000_000):
        log(f"[Gold] Price out of range: buy={buy}, sell={sell}. Skipping.")
        return False

    new_row = pd.DataFrame([{
        "date":       str(today),
        "buy_price":  int(buy),
        "sell_price": int(sell),
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(path, index=False)
    log(f"[Gold] Appended {today}: buy={buy:,.0f}, sell={sell:,.0f} (source: {price_data['source']})")
    log(f"[Gold] Dataset now has {len(df)} rows")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SILVER SCRAPING — World price × USD/VND rate
# ══════════════════════════════════════════════════════════════════════════════

def get_usd_vnd_rate() -> float:
    """Get current USD/VND exchange rate from multiple sources."""
    # Try Yahoo Finance first
    try:
        ticker = yf.Ticker("USDVND=X")
        hist = ticker.history(period="5d")
        if not hist.empty:
            rate = float(hist["Close"].iloc[-1])
            if 20000 < rate < 35000:
                log(f"[Silver] USD/VND from Yahoo: {rate:,.0f}")
                return rate
    except Exception as e:
        log(f"[Silver] Yahoo USD/VND error: {e}")

    # Try exchangerate-api
    try:
        resp = requests.get(
            "https://api.exchangerate-api.com/v4/latest/USD",
            timeout=10
        )
        data = resp.json()
        rate = data["rates"].get("VND", 0)
        if 20000 < rate < 35000:
            log(f"[Silver] USD/VND from exchangerate-api: {rate:,.0f}")
            return rate
    except Exception as e:
        log(f"[Silver] exchangerate-api error: {e}")

    # Fallback
    log("[Silver] Using fallback USD/VND: 25,900")
    return 25900.0


def append_silver_price() -> bool:
    """
    Fetch today's silver price from Yahoo Finance (SI=F) and append to dataset_enhanced.csv.
    Also fetches Gold, DXY, VIX for completeness.
    """
    path = os.path.join(DATASET_DIR, "dataset_enhanced.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    last_date = df["Date"].max()
    today = datetime.now().date()

    if last_date.date() >= today:
        log(f"[Silver] Already have data for {today}. Skipping.")
        return False

    # Check if market is open (Mon-Fri, US futures)
    if datetime.now().weekday() >= 5:
        log(f"[Silver] Weekend. Silver futures closed. Skipping.")
        return False

    log(f"[Silver] Fetching from Yahoo Finance (SI=F, GC=F, DX-Y.NYB, ^VIX)...")

    try:
        fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        fetch_end   = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        tickers = yf.download(
            ["SI=F", "GC=F", "DX-Y.NYB", "^VIX"],
            start=fetch_start,
            end=fetch_end,
            auto_adjust=True,
            progress=False,
        )

        if tickers.empty:
            log("[Silver] No data from Yahoo Finance.")
            return False

        if isinstance(tickers.columns, pd.MultiIndex):
            close = tickers["Close"]
            high  = tickers["High"]
            low   = tickers["Low"]
            open_ = tickers["Open"]
        else:
            close = tickers[["Close"]]
            high  = tickers[["High"]]
            low   = tickers[["Low"]]
            open_ = tickers[["Open"]]

        new_rows = []
        for date_idx in close.index:
            row_date = pd.Timestamp(date_idx).date()
            if pd.Timestamp(row_date) <= last_date:
                continue

            def get_val(df_col, ticker):
                try:
                    v = df_col[ticker].get(date_idx, np.nan)
                    return float(v) if not pd.isna(v) else np.nan
                except Exception:
                    return np.nan

            silver_c = get_val(close, "SI=F")
            silver_h = get_val(high,  "SI=F")
            silver_l = get_val(low,   "SI=F")
            silver_o = get_val(open_, "SI=F")
            gold_c   = get_val(close, "GC=F")
            dxy_c    = get_val(close, "DX-Y.NYB")
            vix_c    = get_val(close, "^VIX")

            if np.isnan(silver_c) or not (5 < silver_c < 100):
                log(f"[Silver] Skipping {row_date}: silver={silver_c}")
                continue
            if np.isnan(gold_c) or not (1000 < gold_c < 4000):
                log(f"[Silver] Skipping {row_date}: gold={gold_c}")
                continue

            last_dxy = float(df["DXY"].iloc[-1]) if "DXY" in df.columns else 104.0
            last_vix = float(df["VIX"].iloc[-1]) if "VIX" in df.columns else 20.0

            new_rows.append({
                "Date":         str(row_date),
                "Silver_Close": round(silver_c, 4),
                "Silver_Open":  round(silver_o if not np.isnan(silver_o) else silver_c, 4),
                "Silver_High":  round(silver_h if not np.isnan(silver_h) else silver_c, 4),
                "Silver_Low":   round(silver_l if not np.isnan(silver_l) else silver_c, 4),
                "Gold":         round(gold_c, 4),
                "DXY":          round(dxy_c if not np.isnan(dxy_c) else last_dxy, 4),
                "VIX":          round(vix_c if not np.isnan(vix_c) else last_vix, 4),
                "price":        round(silver_c, 4),
                "date":         str(row_date),
                "high":         round(silver_h if not np.isnan(silver_h) else silver_c, 4),
                "low":          round(silver_l if not np.isnan(silver_l) else silver_c, 4),
            })

        if not new_rows:
            log("[Silver] No valid new rows to add.")
            return False

        new_df = pd.DataFrame(new_rows)
        for col in df.columns:
            if col not in new_df.columns:
                new_df[col] = np.nan
        new_df = new_df.reindex(columns=df.columns)

        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(path, index=False)
        log(f"[Silver] Added {len(new_rows)} rows. Dataset now has {len(df)} rows")
        log(f"[Silver] Latest: Silver={new_rows[-1]['Silver_Close']:.3f} USD/oz, Gold={new_rows[-1]['Gold']:.2f} USD/oz")
        return True

    except Exception as e:
        log(f"[Silver] Error: {e}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append daily gold/silver prices to datasets")
    parser.add_argument("--gold",   action="store_true", help="Only update gold dataset")
    parser.add_argument("--silver", action="store_true", help="Only update silver dataset")
    args = parser.parse_args()

    # Default: update both
    do_gold   = args.gold   or (not args.gold and not args.silver)
    do_silver = args.silver or (not args.gold and not args.silver)

    log("=" * 60)
    log(f"Daily Price Append | gold={do_gold}, silver={do_silver}")
    log(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log("=" * 60)

    gold_updated   = False
    silver_updated = False

    if do_gold:
        try:
            gold_updated = append_gold_price()
        except Exception as e:
            log(f"[Gold] FATAL: {e}")
            traceback.print_exc()

    if do_silver:
        try:
            silver_updated = append_silver_price()
        except Exception as e:
            log(f"[Silver] FATAL: {e}")
            traceback.print_exc()

    log("=" * 60)
    log(f"Done | gold_updated={gold_updated}, silver_updated={silver_updated}")
    log("=" * 60)

    # Exit code: 0 = success (even if no new data), 1 = fatal error
    sys.exit(0)
