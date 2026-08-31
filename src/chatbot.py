import os
import json
from groq import Groq
from rag_pipeline import build_and_query_vectorstore
from main import run_hybrid_prediction
from dotenv import load_dotenv

load_dotenv()

class BitcoinChatbot:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.conversation_history = []
        self.latest_prediction = None
        self.latest_sentiment = None
        self.news_context = None

    def get_latest_analysis(self):
        """Run hybrid prediction and capture results for chat context."""
        print("\n🔄 Fetching latest market analysis...")
        self.latest_sentiment = analyze_sentiment_with_llm()
        self.news_context = build_and_query_vectorstore(top_k=3)
        
        # Run prediction but capture the output
        import io
        import sys
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            run_hybrid_prediction()
        output = f.getvalue()
        
        # Extract key metrics from output
        self.latest_prediction = output
        return output

    def build_system_prompt(self):
        prompt = (
            "You are a Bitcoin market analyst assistant. You have access to:\n"
            "1. A hybrid quantitative + sentiment analysis system\n"
            "2. Recent news articles via RAG retrieval\n"
            "3. The latest model prediction output\n\n"
            "Current Market Context:\n"
        )
        
        if self.latest_prediction:
            prompt += f"Latest Prediction Output:\n{self.latest_prediction}\n\n"
        
        if self.latest_sentiment:
            prompt += f"LLM Sentiment Analysis: {json.dumps(self.latest_sentiment, indent=2)}\n\n"
        
        if self.news_context:
            prompt += "Recent Relevant News:\n"
            for article in self.news_context:
                prompt += f"- {article['title']} ({article['published']})\n"
            prompt += "\n"
        
        prompt += (
            "Answer user questions about Bitcoin market outlook, "
            "explain the prediction methodology, discuss news impact, "
            "or provide educational context. Be concise but informative. "
            "Always clarify this is not financial advice."
        )
        return prompt

    def chat(self, user_message):
        self.conversation_history.append({"role": "user", "content": user_message})
        
        messages = [
            {"role": "system", "content": self.build_system_prompt()}
        ] + self.conversation_history[-10:]  # Keep last 10 exchanges
        
        response = self.client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )
        
        reply = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    def run_cli(self):
        print("=" * 60)
        print("₿ BITCOIN MARKET ANALYST CHATBOT")
        print("=" * 60)
        print("Type 'refresh' to update analysis, 'quit' to exit\n")
        
        # Initial analysis
        self.get_latest_analysis()
        print("✅ Initial analysis loaded. Ask me anything!\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ('quit', 'exit', 'q'):
                    print("Goodbye! 👋")
                    break
                
                if user_input.lower() == 'refresh':
                    self.get_latest_analysis()
                    print("✅ Analysis refreshed!\n")
                    continue
                
                if not user_input:
                    continue
                
                print("🤖 Analyst: ", end="", flush=True)
                reply = self.chat(user_input)
                print(reply + "\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}\n")


# Reuse existing sentiment function
def analyze_sentiment_with_llm(query="What is the latest Bitcoin market sentiment and news?"):
    from llm_sentiment import analyze_sentiment_with_llm as _analyze
    return _analyze(query)


if __name__ == "__main__":
    bot = BitcoinChatbot()
    bot.run_cli()