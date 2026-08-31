import streamlit as st
import os
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from groq import Groq
from rag_pipeline import build_and_query_vectorstore
from main import run_hybrid_prediction, get_latest_technical_signal
from llm_sentiment import analyze_sentiment_with_llm
from dotenv import load_dotenv
import io
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta
import joblib
import xgboost as xgb

load_dotenv()

st.set_page_config(
    page_title="₿ Bitcoin Analyst Pro",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #f7931a 0%, #ff9f0a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
    }
    .metric-card {
        background: #1e1e2e;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #2d2d44;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        color: #888;
        font-size: 0.85rem;
    }
    .signal-buy { color: #00d4aa; }
    .signal-sell { color: #ff4b4b; }
    .signal-hold { color: #f7931a; }
    .news-card {
        background: #1e1e2e;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #2d2d44;
        margin-bottom: 0.75rem;
    }
    .news-title {
        font-weight: 600;
        color: #f7931a;
    }
    .news-meta {
        color: #888;
        font-size: 0.8rem;
    }
    .stChatMessage {
        background: #1e1e2e !important;
    }
    .chat-header {
        background: #1e1e2e;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #2d2d44;
        margin-bottom: 1rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-live { background: rgba(0, 212, 170, 0.2); color: #00d4aa; }
    .status-stale { background: rgba(255, 75, 75, 0.2); color: #ff4b4b; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: #f7931a !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgboost_model.json")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "btc_features.csv")

@st.cache_resource
def get_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

@st.cache_data(ttl=300)
def load_price_data():
    df = pd.read_csv(FEATURES_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data(ttl=300)
def get_latest_analysis():
    sentiment = analyze_sentiment_with_llm()
    news = build_and_query_vectorstore(top_k=5)
    f = io.StringIO()
    with redirect_stdout(f):
        run_hybrid_prediction()
    prediction_output = f.getvalue()
    prob_up = get_latest_technical_signal()
    return {
        "sentiment": sentiment,
        "news": news,
        "prediction_output": prediction_output,
        "prob_up": prob_up
    }

def parse_prediction_output(output):
    lines = output.strip().split('\n')
    data = {}
    for line in lines:
        if ':' in line and any(k in line for k in ['Technical Signal', 'News Sentiment', 'Combined Hybrid', 'Final Trade']):
            key, val = line.split(':', 1)
            data[key.strip().replace('• ', '').strip()] = val.strip()
    return data

def create_price_chart(df, days=90):
    recent = df.tail(days).copy()
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.55, 0.2, 0.25],
        subplot_titles=('BTC Price & Moving Averages', 'Volume', 'RSI & MACD')
    )
    
    fig.add_trace(go.Candlestick(
        x=recent['Date'], open=recent['open'], high=recent['high'],
        low=recent['low'], close=recent['close'],
        name='BTC', increasing_line_color='#00d4aa', decreasing_line_color='#ff4b4b'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=recent['Date'], y=recent['sma_20'], name='SMA 20',
        line=dict(color='#f7931a', width=1.5)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=recent['Date'], y=recent['ema_50'], name='EMA 50',
        line=dict(color='#5865f2', width=1.5)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=recent['Date'], y=recent['bb_upper'], name='BB Upper',
        line=dict(color='rgba(255,255,255,0.3)', width=1)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=recent['Date'], y=recent['bb_lower'], name='BB Lower',
        line=dict(color='rgba(255,255,255,0.3)', width=1),
        fill='tonexty', fillcolor='rgba(247,147,26,0.1)'
    ), row=1, col=1)
    
    colors = ['#00d4aa' if c >= o else '#ff4b4b' for c, o in zip(recent['close'], recent['open'])]
    fig.add_trace(go.Bar(
        x=recent['Date'], y=recent['volume'], name='Volume',
        marker_color=colors, opacity=0.7
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=recent['Date'], y=recent['rsi'], name='RSI',
        line=dict(color='#f7931a', width=1.5)
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ff4b4b", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00d4aa", row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=recent['Date'], y=recent['macd'], name='MACD',
        line=dict(color='#5865f2', width=1.5)
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=recent['Date'], y=recent['macd_signal'], name='Signal',
        line=dict(color='#ff4b4b', width=1.5)
    ), row=3, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        height=700,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        xaxis_rangeslider_visible=False
    )
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI / MACD", row=3, col=1)
    
    return fig

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>₿ Bitcoin Market Analyst Pro</h1>
        <p>Hybrid XGBoost Quantitative + LLM Sentiment Analysis • Real-time Market Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar(analysis):
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", type="primary", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with col2:
            auto_refresh = st.checkbox("Auto (5m)", value=False)
        
        st.divider()
        
        st.markdown("### 📊 Current Signal")
        prob_up = analysis['prob_up']
        sentiment = analysis['sentiment'].get('sentiment', 'NEUTRAL')
        
        signal_class = "signal-buy" if prob_up > 0.52 else ("signal-sell" if prob_up < 0.48 else "signal-hold")
        signal_text = "BULLISH" if prob_up > 0.52 else ("BEARISH" if prob_up < 0.48 else "NEUTRAL")
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">XGBoost Probability (UP)</div>
            <div class="metric-value {signal_class}">{prob_up:.1%}</div>
            <div class="metric-label">Technical Signal: <span class="{signal_class}">{signal_text}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        sent_color = {"BULLISH": "#00d4aa", "BEARISH": "#ff4b4b", "NEUTRAL": "#f7931a"}.get(sentiment, "#888")
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 0.75rem;">
            <div class="metric-label">News Sentiment (LLM)</div>
            <div class="metric-value" style="color: {sent_color};">{sentiment}</div>
            <div class="metric-label">{analysis['sentiment'].get('summary', '')[:80]}...</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown("### 📈 Data Sources")
        st.markdown("""
        - **Price**: Yahoo Finance (BTC-USD)
        - **Derivatives**: Binance API (Funding Rate, OI)
        - **News**: CoinTelegraph, CoinDesk RSS
        - **LLM**: Groq qwen/qwen3.6-27b
        - **Embeddings**: all-MiniLM-L6-v2
        """)
        
        st.divider()
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
        st.caption("⚠️ Not financial advice")

def render_dashboard(analysis, df):
    st.subheader("📊 Market Dashboard")
    
    pred_data = parse_prediction_output(analysis['prediction_output'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        prob = float(pred_data.get('Technical Signal (XGBoost Score)', '0.5'))
        st.metric("XGBoost Score", f"{prob:.2f}", delta=f"{(prob-0.5)*100:+.1f}% vs neutral")
    
    with col2:
        hybrid = float(pred_data.get('Combined Hybrid Score', '0.5'))
        st.metric("Hybrid Score", f"{hybrid:.2f}", delta=f"{(hybrid-0.5)*100:+.1f}% vs neutral")
    
    with col3:
        decision = pred_data.get('Final Trade Signal', 'HOLD ⚪')
        st.metric("Trade Signal", decision.split(' ')[0] + ' ' + decision.split(' ')[1])
    
    with col4:
        latest_price = df['close'].iloc[-1]
        change = df['daily_return'].iloc[-1] * 100
        st.metric("BTC Price", f"${latest_price:,.0f}", delta=f"{change:+.2f}%")
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["📈 Price Chart", "📰 News & Sentiment", "🤖 Model Details"])
    
    with tab1:
        days = st.select_slider("Timeframe", [30, 60, 90, 180, 365], value=90)
        fig = create_price_chart(df, days)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("#### 📰 Recent News (RAG Retrieved)")
            for idx, article in enumerate(analysis['news']):
                st.markdown(f"""
                <div class="news-card">
                    <div class="news-title">{idx+1}. {article['title']}</div>
                    <div class="news-meta">{article['published']} • <a href="{article['url']}" target="_blank" style="color: #f7931a;">Source</a></div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_right:
            st.markdown("#### 🧠 Sentiment Analysis")
            sent = analysis['sentiment']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Overall Sentiment</div>
                <div class="metric-value" style="color: {'#00d4aa' if sent.get('sentiment')=='BULLISH' else '#ff4b4b' if sent.get('sentiment')=='BEARISH' else '#f7931a'};">
                    {sent.get('sentiment', 'NEUTRAL')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"**Summary:** {sent.get('summary', 'N/A')}")
            st.markdown("**Key Factors:**")
            for factor in sent.get('key_factors', []):
                st.markdown(f"- {factor}")
            st.markdown("**Sources:**")
            for src in sent.get('cited_sources', []):
                st.markdown(f"- {src}")
    
    with tab3:
        st.markdown("#### 🔍 Full Prediction Output")
        st.code(analysis['prediction_output'], language='text')
        
        st.markdown("#### 📋 Feature Importance (XGBoost)")
        if os.path.exists(MODEL_PATH):
            model = xgb.Booster()
            model.load_model(MODEL_PATH)
            feature_names = [c for c in df.columns if c not in ['Date', 'target', 'Target']]
            if hasattr(model, 'feature_importances_'):
                importances = pd.DataFrame({
                    'feature': feature_names[:len(model.feature_importances_)],
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(15)
                
                fig = px.bar(importances, x='importance', y='feature', orientation='h',
                           template='plotly_dark', height=400)
                fig.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
                st.plotly_chart(fig, width='stretch')

def render_chat(analysis):
    st.divider()
    st.markdown("""
    <div class="chat-header">
        <h3 style="margin: 0;">💬 Ask the Analyst</h3>
        <p style="margin: 0.5rem 0 0 0; color: #888;">Query the hybrid model, request explanations, or discuss market outlook</p>
    </div>
    """, unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "₿"):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask about Bitcoin outlook, methodology, news impact, technical levels..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        
        client = get_groq_client()
        
        system_prompt = (
            "You are a Bitcoin market analyst assistant with access to:\n"
            "1. Hybrid quantitative (XGBoost) + sentiment (LLM+RAG) analysis\n"
            "2. Recent news articles via vector search\n"
            "3. Latest model prediction output\n\n"
            f"Current Market Context:\n"
            f"Latest Prediction:\n{analysis['prediction_output']}\n\n"
            f"Sentiment Analysis: {json.dumps(analysis['sentiment'], indent=2)}\n\n"
            "Recent News:\n" +
            "\n".join([f"- {a['title']} ({a['published']})" for a in analysis['news']]) +
            "\n\nAnswer questions about Bitcoin outlook, explain methodology, "
            "discuss news impact, or provide educational context. "
            "Be concise, informative, and always clarify this is not financial advice."
        )
        
        messages = [
            {"role": "system", "content": system_prompt}
        ] + st.session_state.messages[-10:]
        
        with st.chat_message("assistant", avatar="₿"):
            with st.spinner("Analyzing..."):
                response = client.chat.completions.create(
                    model="qwen/qwen3.6-27b",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=600
                )
                reply = response.choices[0].message.content
                st.markdown(reply)
        
        st.session_state.messages.append({"role": "assistant", "content": reply})

def main():
    render_header()
    
    with st.spinner("Loading market analysis..."):
        analysis = get_latest_analysis()
        df = load_price_data()
    
    render_sidebar(analysis)
    render_dashboard(analysis, df)
    render_chat(analysis)

if __name__ == "__main__":
    main()