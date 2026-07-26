import os
import pandas as pd
import pandas_ta as ta


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "btc_raw.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "btc_features.csv")

def add_features(input_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH):

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw data not found at {input_path}. Run data_loader.py first.")
        
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    

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

    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    df_clean = df.dropna().copy()


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path)
    
    print(f"Feature engineering completed successfully!")
    print(f"Processed dataset saved to: {output_path}")
    print(f"Dataset shape: {df_clean.shape}")
    print(f"Target Distribution (0 = Down, 1 = Up):\n{df_clean['target'].value_counts(normalize=True)}")
    
    return df_clean

if __name__ == "__main__":
    add_features()