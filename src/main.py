import os
import joblib
import pandas as pd
from llm_sentiment import analyze_sentiment_with_llm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgboost_btc.joblib")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "btc_features.csv")

def get_latest_technical_signal():
    """
    Load latest feature data and pre-trained XGBoost model to generate numeric prediction.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print("Model or feature data not found. Using fallback neutral technical signal.")
        return 0.5  # Neutral probability

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    
    # Get the most recent row of indicators
    latest_features = df.drop(columns=['Date', 'Target'], errors='ignore').iloc[[-1]]
    
    # Probability of price going UP
    prob_up = model.predict_proba(latest_features)[0][1]
    return prob_up

def run_hybrid_prediction():
    print("=" * 50)
    print("🚀 RUNNING HYBRID BITCOIN PRICE FORECASTER")
    print("=" * 50)

    # 1. Get Technical Signal (XGBoost)
    print("\n1. Calculating Quantitative Signal (XGBoost)...")
    prob_up = get_latest_technical_signal()
    print(f"   -> Model Probability of UP move: {prob_up * 100:.2f}%")

    # 2. Get Qualitative Signal (Groq LLM + RAG)
    print("\n2. Extracting Market Sentiment (Groq RAG Pipeline)...")
    sentiment_data = analyze_sentiment_with_llm()
    sentiment_label = sentiment_data.get("sentiment", "NEUTRAL").upper()

    # Convert sentiment to numeric multiplier
    sentiment_weights = {"BULLISH": 0.15, "NEUTRAL": 0.0, "BEARISH": -0.15}
    sentiment_adjustment = sentiment_weights.get(sentiment_label, 0.0)

    # 3. Hybrid Synthesis
    final_score = prob_up + sentiment_adjustment
    # Clamp score between 0.0 and 1.0
    final_score = max(0.0, min(1.0, final_score))

    # 4. Generate Recommendation
    if final_score >= 0.60:
        decision = "STRONG BUY 🟢"
    elif final_score >= 0.52:
        decision = "WEAK BUY 🟡"
    elif final_score <= 0.40:
        decision = "STRONG SELL 🔴"
    elif final_score <= 0.48:
        decision = "WEAK SELL 🟠"
    else:
        decision = "HOLD ⚪"

    # 5. Output Summary Report
    print("\n" + "=" * 50)
    print("📌 HYBRID FORECAST SUMMARY")
    print("=" * 50)
    print(f"• Technical Signal (XGBoost Score) : {prob_up:.2f}")
    print(f"• News Sentiment Label (LLM)      : {sentiment_label}")
    print(f"• Combined Hybrid Score            : {final_score:.2f}")
    print(f"• Final Trade Signal               : {decision}")
    print("-" * 50)
    print(f"Summary: {sentiment_data.get('summary', 'N/A')}")
    print("=" * 50)

if __name__ == "__main__":
    run_hybrid_prediction()