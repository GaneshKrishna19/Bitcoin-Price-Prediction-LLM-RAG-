import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, f1_score


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "btc_features.csv")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "xgboost_model.json")

def train_numeric_model():
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA_PATH}. Run feature_engineering.py first.")
    
    df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
    
    ignore_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    X = df[feature_cols]
    y = df['target']
    
    print(f"Features used ({len(feature_cols)}): {feature_cols}\n")

    tscv = TimeSeriesSplit(n_splits=5)
    
    print("--- Walk-Forward Cross Validation ---")
    fold = 1
    scores = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        

        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        scores.append(acc)
        print(f"Fold {fold} Accuracy: {acc:.2%}")
        fold += 1

    print(f"\nAverage Cross-Validation Accuracy: {sum(scores)/len(scores):.2%}\n")


    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    final_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric='logloss'
    )
    final_model.fit(X_train, y_train)
    

    y_pred = final_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    
    print("--- Final Model Holdout Test Results ---")
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    final_model.save_model(MODEL_SAVE_PATH)
    print(f"Model successfully saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_numeric_model()