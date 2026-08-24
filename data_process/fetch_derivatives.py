import os
import requests
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DERIVATIVES_OUT_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "btc_derivatives.csv")

def fetch_binance_funding_rates(symbol="BTCUSDT", limit=500):

    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}
    
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data)
    df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
    df['fundingRate'] = df['fundingRate'].astype(float)
    df = df.rename(columns={'fundingTime': 'Date', 'fundingRate': 'FundingRate'})
    return df[['Date', 'FundingRate']]

def fetch_binance_open_interest_hist(symbol="BTCUSDT", period="1d", limit=500):

    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": period, "limit": limit}
    
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['sumOpenInterestValue'] = df['sumOpenInterestValue'].astype(float)
    df = df.rename(columns={'timestamp': 'Date', 'sumOpenInterestValue': 'OpenInterestUSD'})
    return df[['Date', 'OpenInterestUSD']]

def update_derivatives_dataset():
    print("Fetching Binance Futures funding rates and open interest...")
    
    funding_df = fetch_binance_funding_rates()

    funding_df['Date'] = funding_df['Date'].dt.floor('D')
    daily_funding = funding_df.groupby('Date')['FundingRate'].mean().reset_index()

    oi_df = fetch_binance_open_interest_hist()
    oi_df['Date'] = oi_df['Date'].dt.floor('D')


    merged = pd.merge(daily_funding, oi_df, on='Date', how='inner')
    
    merged['OI_Change_Pct'] = merged['OpenInterestUSD'].pct_change()
    merged.dropna(inplace=True)

    # Strip timezone before saving to CSV or returning
    funding_df['Date'] = pd.to_datetime(funding_df['Date']).dt.tz_localize(None)
    oi_df['Date'] = pd.to_datetime(oi_df['Date']).dt.tz_localize(None)

    os.makedirs(os.path.dirname(DERIVATIVES_OUT_PATH), exist_ok=True)
    merged.to_csv(DERIVATIVES_OUT_PATH, index=False)
    print(f"Saved derivatives features to {DERIVATIVES_OUT_PATH}")
    return merged

if __name__ == "__main__":
    update_derivatives_dataset()