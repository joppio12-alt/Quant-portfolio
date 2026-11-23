import yfinance as yf

print("Testing data pull...")
data = yf.download("AAPL", period="5d")
print(data)
print("\nSuccess! Your environment can pull market data.")