"""
Collect Vietnam SJC Gold Price History
Fills gaps in gold_price_sjc_complete.csv using multiple real sources:
1. giavang.org - historical SJC daily pages (Apr 2024 - Dec 2024)
2. webgia.com - 1-year Highcharts chart data (Feb 2025 - Feb 2026)
3. CafeF API - actual SJC prices (last ~32 days)
4. vnstock - today's SJC price
5. exchange-rates.org + premium model - ONLY for remaining gap (~2 months)

Priority: CafeF > webgia > giavang > derived
Output: VND/luong (full VND, not trieu VND)
"""

import os
import sys
import re
import json
import time
import requests
import pandas as pd
import numpy as np  # noqa: F401
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')
SJC_CSV = os.path.join(DATASET_DIR, 'gold_price_sjc_complete.csv')
OZ_TO_LUONG = 1.20565
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}


# =============================================================================
# Source 1: giavang.org - Historical daily pages
# =============================================================================
def fetch_giavang_org(start_date, end_date):
    """Scrape daily SJC prices from giavang.org individual date pages.

    URL pattern: https://giavang.org/trong-nuoc/sjc/lich-su/YYYY-MM-DD.html
    Price unit: x1000 VND/luong (e.g. 82,800 = 82,800,000 VND)
    Available range: ~Apr 2024 to ~Dec 2024
    """
    print(f"[1/5] Fetching giavang.org SJC data ({start_date} to {end_date})...")
    rows = []
    current = start_date
    total_days = (end_date - start_date).days + 1
    fetched = 0
    errors = 0

    while current <= end_date:
        date_str = current.strftime('%Y-%m-%d')
        url = f'https://giavang.org/trong-nuoc/sjc/lich-su/{date_str}.html'

        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                text = soup.get_text()

                # Check if page has no data (Vietnamese: "Không tìm thấy")
                if 'kh\u00f4ng t\u00ecm th\u1ea5y' in text.lower() or 'khong tim thay' in text.lower():
                    pass  # No data for this date (weekend/holiday)
                else:
                    # Extract prices from the page
                    # Pattern: "gia mua vao la XX.XXX trieu / luong va ban ra la YY.YYY trieu / luong"
                    # Or extract from table data
                    buy_price = None
                    sell_price = None

                    # Method 1: Find price table rows
                    tables = soup.find_all('table')
                    for table in tables:
                        trs = table.find_all('tr')
                        for tr in trs:
                            tds = tr.find_all('td')
                            if len(tds) >= 3:
                                type_text = tds[0].get_text(strip=True)
                                # Look for "Mieng" (bar) or "1L" type
                                if any(k in type_text.lower() for k in ['mieng', '1l', '10l', '1kg']):
                                    try:
                                        buy_text = re.sub(r'[^\d.]', '', tds[1].get_text(strip=True))
                                        sell_text = re.sub(r'[^\d.]', '', tds[2].get_text(strip=True))
                                        if buy_text and sell_text:
                                            buy_price = float(buy_text)
                                            sell_price = float(sell_text)
                                            break
                                    except ValueError:
                                        pass
                        if buy_price:
                            break

                    # Method 2: Extract from description text
                    if not buy_price:
                        # Pattern: "mua vao la 82.800 trieu" or "78.500"
                        buy_match = re.search(r'mua\s+v[aà]o\s+l[aà]\s+([\d.]+)', text, re.IGNORECASE)
                        sell_match = re.search(r'b[aá]n\s+ra\s+l[aà]\s+([\d.]+)', text, re.IGNORECASE)
                        if buy_match and sell_match:
                            buy_price = float(buy_match.group(1).replace('.', '').replace(',', '.') if '.' in buy_match.group(1) and len(buy_match.group(1).split('.')[-1]) == 3 else buy_match.group(1))
                            sell_price = float(sell_match.group(1).replace('.', '').replace(',', '.') if '.' in sell_match.group(1) and len(sell_match.group(1).split('.')[-1]) == 3 else sell_match.group(1))

                    if buy_price and sell_price:
                        # Prices are in x1000 VND -> convert to full VND
                        # If value like 82.8 -> trieu VND -> *1M
                        # If value like 82800 -> x1000 VND -> *1000
                        if buy_price < 1000:
                            # trieu VND format (e.g. 82.8)
                            buy_vnd = int(buy_price * 1_000_000)
                            sell_vnd = int(sell_price * 1_000_000)
                        else:
                            # x1000 VND format (e.g. 82800)
                            buy_vnd = int(buy_price * 1_000)
                            sell_vnd = int(sell_price * 1_000)

                        rows.append({
                            'date': pd.Timestamp(current),
                            'buy_price': buy_vnd,
                            'sell_price': sell_vnd,
                            'source': 'giavang.org'
                        })
                        fetched += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Warning: {date_str} failed: {e}")

        # Rate limiting
        time.sleep(0.3)
        current += timedelta(days=1)

        # Progress
        progress = (current - start_date).days
        if progress % 30 == 0:
            print(f"  Progress: {progress}/{total_days} days checked, {fetched} fetched...")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('date').drop_duplicates(subset='date', keep='last')
    print(f"  -> {len(df)} days from giavang.org")
    if len(df) > 0:
        print(f"     Range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"     Sample: buy={df.iloc[-1]['buy_price']:,.0f}, sell={df.iloc[-1]['sell_price']:,.0f} VND")
    return df


# =============================================================================
# Source 2: webgia.com - 1-year Highcharts chart data
# =============================================================================
def fetch_webgia_chart():
    """Extract SJC price data from webgia.com 1-year chart (Highcharts).

    Data is embedded in JavaScript as seriesOptions with [timestamp, price] pairs.
    Price unit: trieu VND/luong (e.g. 90.3 = 90,300,000 VND)
    Available range: ~last 12 months
    """
    print("[2/5] Fetching webgia.com 1-year chart data...")
    url = 'https://webgia.com/gia-vang/sjc/bieu-do-1-nam.html'
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            print(f"  -> Failed: status {r.status_code}")
            return pd.DataFrame()

        soup = BeautifulSoup(r.text, 'html.parser')

        for script in soup.find_all('script'):
            content = script.string or ''
            if 'seriesOptions' not in content:
                continue

            # Extract data arrays: data:[[ts,price],[ts,price],...]
            data_arrays = re.findall(r'data:(\[\[.*?\]\])', content, re.DOTALL)
            if len(data_arrays) < 2:
                print(f"  -> Found {len(data_arrays)} series (need 2)")
                continue

            # Series 0 = Sell (Ban ra), Series 1 = Buy (Mua vao)
            sell_data = json.loads(data_arrays[0])
            buy_data = json.loads(data_arrays[1])

            rows = []
            # Create lookup for buy prices
            buy_lookup = {d[0]: d[1] for d in buy_data}

            for ts, sell_price in sell_data:
                date = datetime.fromtimestamp(ts / 1000)
                buy_price = buy_lookup.get(ts, sell_price - 3.0)  # Default spread ~3M

                rows.append({
                    'date': pd.Timestamp(date.date()),
                    'buy_price': int(buy_price * 1_000_000),  # trieu VND -> VND
                    'sell_price': int(sell_price * 1_000_000),
                    'source': 'webgia.com'
                })

            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.sort_values('date').drop_duplicates(subset='date', keep='last')
            print(f"  -> {len(df)} days from webgia.com")
            if len(df) > 0:
                print(f"     Range: {df['date'].min().date()} to {df['date'].max().date()}")
                print(f"     Sample: buy={df.iloc[-1]['buy_price']:,.0f}, sell={df.iloc[-1]['sell_price']:,.0f} VND")
            return df

        print("  -> No chart data found in page")
        return pd.DataFrame()
    except Exception as e:
        print(f"  -> webgia.com failed: {e}")
        return pd.DataFrame()


# =============================================================================
# Source 3: CafeF API - last ~32 days
# =============================================================================
def fetch_cafef_sjc():
    """Fetch actual SJC prices from CafeF API (last ~32 days)."""
    print("[3/5] Fetching CafeF SJC data...")
    url = 'https://s.cafef.vn/Ajax/ajaxgoldpricehistory.ashx?index=7'
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        data = r.json()
        history = data['Data']['goldPriceWorldHistories']

        rows = []
        for item in history:
            if item.get('name') == 'SJC':
                date_str = item['createdAt'][:10]
                buy = item['buyPrice'] * 1_000_000  # trieu VND -> VND
                sell = item['sellPrice'] * 1_000_000
                rows.append({'date': date_str, 'buy_price': int(buy), 'sell_price': int(sell), 'source': 'cafef'})

        df = pd.DataFrame(rows)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').drop_duplicates(subset='date', keep='last')
        print(f"  -> {len(df)} days from CafeF" + (f" ({df['date'].min().date()} to {df['date'].max().date()})" if len(df) > 0 else ""))
        return df
    except Exception as e:
        print(f"  -> CafeF failed: {e}")
        return pd.DataFrame()


# =============================================================================
# Source 4: vnstock - today's price
# =============================================================================
def fetch_vnstock_sjc():
    """Fetch today's SJC price from vnstock."""
    print("[4/5] Fetching vnstock SJC data...")
    try:
        from vnstock.explorer.misc import sjc_gold_price
        df = sjc_gold_price()
        hcm = df[df['branch'].str.contains('Minh|HCM', case=False, na=False)]
        if hcm.empty:
            hcm = df.iloc[:1]
        row = hcm.iloc[0]
        result = pd.DataFrame([{
            'date': pd.to_datetime(row['date']),
            'buy_price': int(float(row['buy_price'])),
            'sell_price': int(float(row['sell_price'])),
            'source': 'vnstock'
        }])
        print(f"  -> Today: buy={row['buy_price']:,.0f}, sell={row['sell_price']:,.0f} VND")
        return result
    except Exception as e:
        print(f"  -> vnstock failed: {e}")
        return pd.DataFrame()


# =============================================================================
# Source 5: exchange-rates.org + premium model (fallback only)
# =============================================================================
def fetch_exchange_rates_world_gold(year):
    """Fetch daily world gold prices in VND/oz from exchange-rates.org."""
    url = f'https://www.exchange-rates.org/precious-metals/gold-price/vietnam/{year}'
    r = requests.get(url, headers=HEADERS, timeout=30)
    soup = BeautifulSoup(r.text, 'html.parser')

    tables = soup.find_all('table')
    if not tables:
        return pd.DataFrame()

    table = tables[0]
    rows_html = table.find_all('tr')
    data = []
    current_month = ''

    for row in rows_html:
        cells = row.find_all(['td', 'th'])
        if len(cells) == 1:
            current_month = cells[0].get_text(strip=True)
            continue
        if len(cells) >= 2:
            date_text = cells[0].get_text(strip=True)
            price_text = cells[1].get_text(strip=True)
            if 'Gold Price' in price_text or not date_text:
                continue
            price_clean = re.sub(r'[^\d.]', '', price_text)
            if price_clean and date_text:
                try:
                    price = float(price_clean)
                    month_parts = current_month.split()
                    year_val = int(month_parts[-1])
                    month_name = month_parts[0]
                    day_num = re.search(r'\d+', date_text).group()
                    date_str = f"{day_num} {month_name} {year_val}"
                    parsed_date = pd.to_datetime(date_str, format='%d %B %Y')
                    data.append({'date': parsed_date, 'world_gold_vnd_oz': price})
                except Exception:
                    pass

    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values('date').drop_duplicates(subset='date', keep='last')
    return df


def fetch_all_world_gold():
    """Fetch world gold VND/oz for relevant years."""
    print("[5/5] Fetching world gold prices (exchange-rates.org) for gap filling...")
    frames = []
    for year in [2024, 2025, 2026]:
        try:
            df = fetch_exchange_rates_world_gold(year)
            print(f"  -> {year}: {len(df)} days")
            frames.append(df)
        except Exception as e:
            print(f"  -> {year}: failed - {e}")

    if frames:
        return pd.concat(frames, ignore_index=True).sort_values('date').drop_duplicates(subset='date')
    return pd.DataFrame()


def calculate_sjc_premium(real_sjc_data, world_gold):
    """Calculate SJC premium from real data overlapping with world gold."""
    world = world_gold.copy()
    world['world_vnd_luong'] = world['world_gold_vnd_oz'] * OZ_TO_LUONG

    sjc = real_sjc_data.copy()
    sjc['mid_price_vnd'] = (sjc['buy_price'] + sjc['sell_price']) / 2

    merged = pd.merge(sjc[['date', 'mid_price_vnd']], world[['date', 'world_vnd_luong']], on='date', how='inner')
    merged['premium'] = merged['mid_price_vnd'] / merged['world_vnd_luong']

    # Use all overlap for better estimation
    if len(merged) < 5:
        print(f"  Warning: only {len(merged)} overlapping days for premium calc")
        if len(merged) == 0:
            return 1.05  # Default fallback

    mean_premium = merged['premium'].mean()
    print(f"  Premium from {len(merged)} overlapping days: {mean_premium:.4f}")
    return mean_premium


def derive_sjc_from_world(world_gold, premium, spread_vnd=3_000_000):
    """Derive SJC buy/sell from world gold using premium and fixed spread."""
    df = world_gold.copy()
    df['world_vnd_luong'] = df['world_gold_vnd_oz'] * OZ_TO_LUONG
    mid_price = df['world_vnd_luong'] * premium

    half_spread = spread_vnd / 2
    df['buy_price'] = (mid_price - half_spread).round(0).astype(int)
    df['sell_price'] = (mid_price + half_spread).round(0).astype(int)
    df['source'] = 'derived'

    return df[['date', 'buy_price', 'sell_price', 'source']]


# =============================================================================
# Main
# =============================================================================
def main():
    print(f"=== Vietnam Gold Data Collection: {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")

    # Load existing CSV
    sjc = pd.read_csv(SJC_CSV)
    sjc['date'] = pd.to_datetime(sjc['date'])
    last_date = sjc['date'].max()

    # Detect unit
    is_trieu = sjc['buy_price'].max() < 1000
    unit_label = "trieu VND" if is_trieu else "VND"
    print(f"Existing CSV: {len(sjc)} rows, {sjc['date'].min().date()} to {last_date.date()}")
    print(f"Current unit: {unit_label}/luong\n")

    # Convert existing to VND if needed
    if is_trieu:
        print("--- Converting existing CSV to VND ---")
        sjc['buy_price'] = (sjc['buy_price'] * 1_000_000).round(0).astype(int)
        sjc['sell_price'] = (sjc['sell_price'] * 1_000_000).round(0).astype(int)
        print(f"  Converted {len(sjc)} rows (trieu VND -> VND)\n")

    # =========================================================================
    # Fetch all real data sources
    # =========================================================================

    # Source 1: giavang.org (Apr 2024 - Dec 2024)
    giavang_start = max(last_date + timedelta(days=1), pd.Timestamp('2024-04-01'))
    giavang_end = pd.Timestamp('2024-12-15')  # Data stops ~Dec 2, 2024
    giavang_df = fetch_giavang_org(giavang_start, giavang_end)

    # Source 2: webgia.com (1-year chart)
    webgia_df = fetch_webgia_chart()

    # Source 3: CafeF (last 32 days)
    cafef_df = fetch_cafef_sjc()

    # Source 4: vnstock (today)
    vnstock_df = fetch_vnstock_sjc()

    # =========================================================================
    # Combine all real data (priority: cafef > vnstock > webgia > giavang)
    # =========================================================================
    print("\n--- Combining real data sources ---")

    all_real = pd.DataFrame()

    # Start with lowest priority
    for name, df in [('giavang.org', giavang_df), ('webgia.com', webgia_df),
                      ('cafef', cafef_df), ('vnstock', vnstock_df)]:
        if not df.empty:
            if all_real.empty:
                all_real = df.copy()
            else:
                # Higher priority overwrites lower priority for same dates
                existing_dates = set(all_real['date'].dt.date)
                new_dates = set(df['date'].dt.date)
                overlap = existing_dates & new_dates
                if overlap:
                    all_real = all_real[~all_real['date'].dt.date.isin(overlap)]
                all_real = pd.concat([all_real, df], ignore_index=True)
            print(f"  Added {name}: {len(df)} days")

    if not all_real.empty:
        all_real = all_real.sort_values('date').drop_duplicates(subset='date', keep='last')
        print(f"  Total real data: {len(all_real)} days ({all_real['date'].min().date()} to {all_real['date'].max().date()})")

    # =========================================================================
    # Fill remaining gaps with derived data
    # =========================================================================
    today = pd.Timestamp.now().normalize()
    gap_start = last_date + timedelta(days=1)

    # Find dates NOT covered by real data
    all_dates_needed = pd.date_range(start=gap_start, end=today, freq='D')
    covered_dates = set(all_real['date'].dt.date) if not all_real.empty else set()
    uncovered_dates = [d for d in all_dates_needed if d.date() not in covered_dates]

    derived_df = pd.DataFrame()
    if uncovered_dates:
        # Find continuous gaps (>3 consecutive days without data)
        # Scattered weekends will be handled by forward-fill later
        all_known = pd.concat([sjc[['date', 'buy_price', 'sell_price']],
                                all_real[['date', 'buy_price', 'sell_price']]], ignore_index=True)
        all_known = all_known.sort_values('date').drop_duplicates(subset='date', keep='last')

        # Identify continuous gaps by finding runs of uncovered dates
        uncovered_sorted = sorted(uncovered_dates)
        gaps = []
        current_gap = [uncovered_sorted[0]]
        for i in range(1, len(uncovered_sorted)):
            if (uncovered_sorted[i] - uncovered_sorted[i-1]).days <= 3:
                current_gap.append(uncovered_sorted[i])
            else:
                if len(current_gap) > 3:  # Only process gaps > 3 days
                    gaps.append(current_gap)
                current_gap = [uncovered_sorted[i]]
        if len(current_gap) > 3:
            gaps.append(current_gap)

        print(f"\n--- {len(uncovered_dates)} uncovered dates, {len(gaps)} continuous gap(s) ---")

        all_gap_rows = []
        world_gold = None

        for gap in gaps:
            first_gap = min(gap)
            last_gap = max(gap)
            print(f"\n  Gap: {first_gap.date()} to {last_gap.date()} ({len(gap)} dates)")

            # Find anchor points
            before = all_known[all_known['date'] < first_gap].tail(1)
            after = all_known[all_known['date'] > last_gap].head(1)

            if not before.empty and not after.empty:
                anchor_buy_start = before.iloc[0]['buy_price']
                anchor_sell_start = before.iloc[0]['sell_price']
                anchor_buy_end = after.iloc[0]['buy_price']
                anchor_sell_end = after.iloc[0]['sell_price']
                print(f"  Anchors: {anchor_buy_start:,.0f}/{anchor_sell_start:,.0f} -> {anchor_buy_end:,.0f}/{anchor_sell_end:,.0f}")

                # Fetch world gold if not already
                if world_gold is None:
                    world_gold = fetch_all_world_gold()

                gap_world = world_gold[(world_gold['date'] >= first_gap) & (world_gold['date'] <= last_gap)]

                if len(gap_world) >= 5:
                    # Shape by world gold pattern, anchored to real endpoints
                    gap_world = gap_world.sort_values('date')
                    wg_prices = gap_world['world_gold_vnd_oz'].values
                    wg_start, wg_end = wg_prices[0], wg_prices[-1]

                    for _, row in gap_world.iterrows():
                        if wg_end != wg_start:
                            t = (row['world_gold_vnd_oz'] - wg_start) / (wg_end - wg_start)
                        else:
                            total = (gap_world['date'].max() - gap_world['date'].min()).days
                            t = (row['date'] - gap_world['date'].min()).days / max(1, total)
                        t = max(0, min(1, t))

                        buy = anchor_buy_start + t * (anchor_buy_end - anchor_buy_start)
                        sell = anchor_sell_start + t * (anchor_sell_end - anchor_sell_start)
                        all_gap_rows.append({
                            'date': row['date'],
                            'buy_price': int(round(buy)),
                            'sell_price': int(round(sell)),
                            'source': 'interpolated'
                        })
                    print(f"  -> Interpolated {len(gap_world)} days (world gold shaped)")
                else:
                    # Linear interpolation
                    total_days = (last_gap - first_gap).days
                    for d in gap:
                        t = (d - first_gap).days / max(1, total_days)
                        buy = anchor_buy_start + t * (anchor_buy_end - anchor_buy_start)
                        sell = anchor_sell_start + t * (anchor_sell_end - anchor_sell_start)
                        all_gap_rows.append({
                            'date': d,
                            'buy_price': int(round(buy)),
                            'sell_price': int(round(sell)),
                            'source': 'interpolated'
                        })
                    print(f"  -> Interpolated {len(gap)} days (linear)")

        if all_gap_rows:
            derived_df = pd.DataFrame(all_gap_rows)
            print(f"\n  Total interpolated: {len(derived_df)} days")
    else:
        print("\n  All dates covered by real data!")

    # =========================================================================
    # Combine everything
    # =========================================================================
    print("\n--- Building final dataset ---")

    # Filter new data (after last CSV date)
    new_data = pd.concat([all_real, derived_df], ignore_index=True) if not derived_df.empty else all_real.copy()
    if not new_data.empty:
        new_data = new_data[new_data['date'] > last_date].sort_values('date').drop_duplicates(subset='date', keep='last')

    # Merge with existing
    final_df = pd.concat([
        sjc[['date', 'buy_price', 'sell_price']],
        new_data[['date', 'buy_price', 'sell_price']]
    ], ignore_index=True)
    final_df = final_df.sort_values('date').drop_duplicates(subset='date', keep='last')

    # Forward-fill weekends/holidays
    date_range = pd.date_range(start=final_df['date'].min(), end=final_df['date'].max(), freq='D')
    full_df = pd.DataFrame({'date': date_range})
    full_df = full_df.merge(final_df, on='date', how='left')
    full_df['buy_price'] = full_df['buy_price'].ffill()
    full_df['sell_price'] = full_df['sell_price'].ffill()

    # Clean up
    full_df['buy_price'] = full_df['buy_price'].astype(int)
    full_df['sell_price'] = full_df['sell_price'].astype(int)

    # Save
    full_df['date'] = full_df['date'].dt.strftime('%Y-%m-%d')
    full_df.to_csv(SJC_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"DONE - Saved {len(full_df)} rows to {SJC_CSV}")
    print(f"Date range: {full_df['date'].iloc[0]} to {full_df['date'].iloc[-1]}")
    print(f"Unit: VND/luong")
    print(f"Last: buy={full_df['buy_price'].iloc[-1]:,}, sell={full_df['sell_price'].iloc[-1]:,}")

    # Source summary
    if not new_data.empty and 'source' in new_data.columns:
        print(f"\nNew data by source:")
        for src, group in new_data.groupby('source'):
            print(f"  {src}: {len(group)} days ({group['date'].min().date()} to {group['date'].max().date()})")
            print(f"    Price range: {group['buy_price'].min():,.0f} - {group['sell_price'].max():,.0f} VND")


if __name__ == "__main__":
    main()
