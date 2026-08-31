import streamlit as st
import os
import json
from groq import Groq
from rag_pipeline import build_and_query_vectorstore
from main import run_hybrid_prediction
from llm_sentiment import analyze_sentiment_with_llm
from dotenv import load_dotenv
import io
import sys
from contextlib import redirect_stdout

load_dotenv()

st.set_page_config(page_title="₿ Bitcoin Analyst", page_icon="₿", layout="wide")

@st.cache_resource
def get_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

@st.cache_data(ttl=300)
def get_latest_analysis():
    """Run full analysis pipeline and return structured results."""
    # Get sentiment
    sentiment = analyze_sentiment_with_llm()
    
    # Get news context
    news = build_and_query_vectorstore(top_k=3)
    
    # Get prediction output
    f = io.StringIO()
    with redirect_stdout(f):
        run_hybrid_prediction()
    prediction_output = f.getvalue()
    
    return {
        "sentiment": sentiment,
        "news": news,
        "prediction_output": prediction_output
    }

def build_system_prompt(analysis):
    prompt = (
        "You are a Bitcoin market analyst assistant with access to:\n"
        "1. Hybrid quantitative (XGBoost) + sentiment (LLM+RAG) analysis\n"
        "2. Recent news articles via vector search\n"
        "3. Latest model prediction output\n\n"
        "Current Market Context:\n"
    )
    
    if analysis["prediction_output"]:
        prompt += f"Latest Prediction:\n{analysis['prediction_output']}\n\n"
    
    if analysis["sentiment"]:
        prompt += f"Sentiment Analysis: {json.dumps(analysis['sentiment'], indent=2)}\n\n"
    
    if analysis["news"]:
        prompt += "Recent News:\n"
        for article in analysis["news"]:
            prompt += f"- {article['title']} ({article['published']})\n"
        prompt += "\n"
    
    prompt += (
        "Answer questions about Bitcoin outlook, explain methodology, "
        "discuss news impact, or provide educational context. "
        "Be concise, informative, and always clarify this is not financial advice."
    )
    return prompt

def main():
    st.title("₿ Bitcoin Market Analyst")
    st.caption("Hybrid XGBoost + LLM Sentiment Analysis Chatbot")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        if st.button("🔄 Refresh Analysis", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        st.markdown("**Data Sources:**")
        st.markdown("- BTC Price: Yahoo Finance")
        st.markdown("- Derivatives: Binance API")
        st.markdown("- News: RSS (CoinTelegraph, CoinDesk, etc.)")
        st.markdown("- LLM: Groq qwen/qwen3.6-27b")
    
    # Load analysis
    with st.spinner("Loading latest market analysis..."):
        analysis = get_latest_analysis()
    
    # Display current prediction summary
    with st.expander("📊 Current Prediction Summary", expanded=True):
        st.code(analysis["prediction_output"], language="text")
    
    # Chat interface
    st.divider()
    st.subheader("💬 Ask the Analyst")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about Bitcoin outlook, methodology, news impact..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        client = get_groq_client()
        messages = [
            {"role": "system", "content": build_system_prompt(analysis)}
        ] + st.session_state.messages[-10:]
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = client.chat.completions.create(
                    model="qwen/qwen3.6-27b",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=500
                )
                reply = response.choices[0].message.content
                st.markdown(reply)
        
        st.session_state.messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()