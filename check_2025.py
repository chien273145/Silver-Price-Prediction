
import yfinance as yf
print("Fetching SI=F for Feb 2025...")
try:
    ticker = yf.Ticker("SI=F")
    hist = ticker.history(start="2025-02-01", end="2025-02-10")
    print(hist[['Close']])
except Exception as e:
    print(e)
