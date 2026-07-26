import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Add src to system path to import modules cleanly
sys.path.append(os.path.abspath("src"))

from main import get_latest_technical_signal
from llm_sentiment import analyze_sentiment_with_llm

st.set_page_config(
    page_title="BTC Hybrid Forecaster",
    page_icon="🪙",
    layout="wide"
)

# Application Header
st.title("🪙 Bitcoin Hybrid Price Forecasting Dashboard")
st.markdown("Quantitative XGBoost Indicators + LLM RAG News Sentiment Analysis")

st.divider()

# Sidebar Control
st.sidebar.header("Controls & Parameters")
top_k_news = st.sidebar.slider("News Articles to Retrieve (RAG)", min_value=1, max_value=10, value=5)
run_button = st.sidebar.button("Run Hybrid Analysis", type="primary", use_container_width=True)

# Main Dashboard View
if run_button:
    with st.spinner("Processing technical indicators & querying Groq LLM RAG pipeline..."):
        # 1. Fetch Signals
        prob_up = get_latest_technical_signal()
        sentiment_data = analyze_sentiment_with_llm()
        sentiment_label = sentiment_data.get("sentiment", "NEUTRAL").upper()
        
        # 2. Hybrid Decision Logic
        sentiment_weights = {"BULLISH": 0.15, "NEUTRAL": 0.0, "BEARISH": -0.15}
        adjustment = sentiment_weights.get(sentiment_label, 0.0)
        final_score = max(0.0, min(1.0, prob_up + adjustment))

        if final_score >= 0.60:
            signal, color = "STRONG BUY 🟢", "#22c55e"
        elif final_score >= 0.52:
            signal, color = "WEAK BUY 🟡", "#eab308"
        elif final_score <= 0.40:
            signal, color = "STRONG SELL 🔴", "#ef4444"
        elif final_score <= 0.48:
            signal, color = "WEAK SELL 🟠", "#f97316"
        else:
            signal, color = "HOLD ⚪", "#6b7280"

    # Layout Row 1: KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Technical Prob (UP)", f"{prob_up * 100:.1f}%")
    col2.metric("News Sentiment", sentiment_label)
    col3.metric("Hybrid Score", f"{final_score:.2f}")
    col4.markdown(f"<h3 style='color: {color}; margin: 0;'>{signal}</h3>", unsafe_allow_html=True)

    st.divider()

    # Layout Row 2: Gauge Visual & LLM Summary
    col_chart, col_summary = st.columns([1, 1])

    with col_chart:
        st.subheader("Hybrid Confidence Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=final_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Bullish Confidence (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40], 'color': "#fca5a5"},
                    {'range': [40, 52], 'color': "#fef08a"},
                    {'range': [52, 100], 'color': "#86efac"}
                ]
            }
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_summary:
        st.subheader("Market Sentiment Synthesis")
        st.info(sentiment_data.get("summary", "No summary generated."))
        
        st.write("**Key Drivers:**")
        for factor in sentiment_data.get("key_factors", []):
            st.markdown(f"- {factor}")

    # Layout Row 3: Retrieved RAG News Sources
    st.divider()
    st.subheader("📰 Cited News Sources (Vector DB Retrieval)")
    sources = sentiment_data.get("cited_sources", [])
    if sources:
        for idx, src in enumerate(sources, 1):
            st.markdown(f"**{idx}.** {src}")
    else:
        st.caption("No explicit sources returned by the LLM.")

else:
    st.info("👈 Click **'Run Hybrid Analysis'** in the sidebar to run predictions!")