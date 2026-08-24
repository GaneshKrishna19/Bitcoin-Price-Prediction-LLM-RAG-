import os
import sys
import pandas as pd
import pandas_ta as ta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "btc_raw.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "btc_features.csv")
DERIVATIVES_PATH = os.path.join(PROJECT_ROOT, "fetch_data")

sys.path.insert(0, DERIVATIVES_PATH)  
from fetch_derivatives import update_derivatives_dataset

def add_features(input_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH):

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw data not found at {input_path}. Run data_loader.py first.")
        
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).floor('D')

    # ----------------------------------------------------
    # 1. Technical Indicators (pandas-ta)
    # ----------------------------------------------------
    df = df.drop(columns=['dividends', 'stock splits'], errors='ignore')
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['sma_20'] = ta.sma(df['close'], length=20)
    df['ema_50'] = ta.ema(df['close'], length=50)
    
    macd = ta.macd(df['close'])
    if macd is not None:
        df['macd'] = macd.iloc[:, 0]        
        df['macd_signal'] = macd.iloc[:, 2] 
        
    bbands = ta.bbands(df['close'], length=20)
    if bbands is not None:
        df['bb_upper'] = bbands.iloc[:, 2]
        df['bb_lower'] = bbands.iloc[:, 0]

    df['daily_return'] = df['close'].pct_change()
    df['return_lag1'] = df['daily_return'].shift(1)
    df['return_lag2'] = df['daily_return'].shift(2)
    df['volume_lag1'] = df['volume'].shift(1)

    # ----------------------------------------------------
    # 2. Fetch & Merge Derivatives Features (Binance API)
    # ----------------------------------------------------
    print("Updating and merging derivatives data...")
    try:
        deriv_df = update_derivatives_dataset()
        
        # 1. Standardize column names to lowercase
        df.columns = df.columns.str.lower()
        deriv_df.columns = deriv_df.columns.str.lower()

        # 2. Extract clean YYYY-MM-DD date strings (stripping time & timezones)
        df['date_key'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.strftime('%Y-%m-%d')
        deriv_df['date_key'] = pd.to_datetime(deriv_df['date']).dt.tz_localize(None).dt.strftime('%Y-%m-%d')

        # 3. Merge derivatives data on daily key
        df = df.merge(deriv_df[['date_key', 'fundingrate', 'oi_change_pct']], on='date_key', how='left')
        df = df.drop(columns=['date_key'])

        # 4. Rename back to standard feature names
        df = df.rename(columns={'fundingrate': 'FundingRate', 'oi_change_pct': 'OI_Change_Pct'})

        # 5. Fill missing historical values
        df['FundingRate'] = df['FundingRate'].ffill().fillna(0)
        df['OI_Change_Pct'] = df['OI_Change_Pct'].ffill().fillna(0)

        print(f"Successfully merged derivatives. Non-zero rows: {(df['FundingRate'] != 0).sum()} / {len(df)}")

    except Exception as e:
        print(f"Warning: Could not fetch derivatives data ({e}). Defaulting columns to 0.")
        df['FundingRate'] = 0.0
        df['OI_Change_Pct'] = 0.0
    # ----------------------------------------------------
    # 3. Target Definition & Cleaning
    # ----------------------------------------------------
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Drop NaNs resulting from indicator windows and lags
    df_clean = df.dropna().copy()

    # Save processed feature set
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path)
    
    print("\nFeature engineering completed successfully!")
    print(f"Processed dataset saved to: {output_path}")
    print(f"Dataset shape: {df_clean.shape}")
    print(f"Included features: {list(df_clean.columns)}")
    print(f"\nTarget Distribution (0 = Down, 1 = Up):\n{df_clean['target'].value_counts(normalize=True)}")
    
    return df_clean


if __name__ == "__main__":
    add_features()