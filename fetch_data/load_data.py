import os
import yfinance as yf
import pandas as pd

Project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Default_save_path = os.path.join(Project_root, "data", "raw", "btc_raw.csv")

def fetch_btcdata(period="2y", save_path=Default_save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print("Fetching BTC daily data...")
    btc = yf.Ticker("BTC-USD")
    df = btc.history(period=period)
    
    
    df.columns = [col.lower() for col in df.columns]
    
    
    df.to_csv(save_path)
    print(f"Data saved successfully to {save_path}! Shape: {df.shape}")
    return df

if __name__ == "__main__":
    data = fetch_btcdata()
    print(data.tail())