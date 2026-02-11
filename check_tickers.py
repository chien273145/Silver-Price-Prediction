
import yfinance as yf
for sym in ["SLV", "XAG-USD"]:
    print(f"Checking {sym}...")
    try:
        t = yf.Ticker(sym)
        h = t.history(period="1d")
        if not h.empty:
            print(f"Price: {h['Close'].iloc[-1]}")
            print(f"Date: {h.index[-1]}")
        else:
            print("No data")
    except Exception as e:
        print(e)
    print("-" * 20)
