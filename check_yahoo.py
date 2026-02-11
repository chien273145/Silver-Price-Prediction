
import yfinance as yf
print("Fetching SI=F last 5 days...")
try:
    ticker = yf.Ticker("SI=F")
    hist = ticker.history(period="5d")
    print(hist[['Open', 'High', 'Low', 'Close', 'Volume']])
except Exception as e:
    print(e)
