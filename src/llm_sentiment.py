import os
import json
from groq import Groq
from rag_pipeline import build_and_query_vectorstore
from dotenv import load_dotenv

load_dotenv()

def analyze_sentiment_with_llm(query="What is the latest Bitcoin market sentiment and news?"):

    retrieved_articles = build_and_query_vectorstore(query_text=query, top_k=5)
    
    if not retrieved_articles:
        return {"sentiment": "NEUTRAL", "reasoning": "No news articles found."}


    context_str = ""
    for idx, article in enumerate(retrieved_articles, 1):
        context_str += f"[{idx}] Title: {article['title']}\n    Content: {article['text']}\n    URL: {article['url']}\n\n"


    system_prompt = (
        "You are an expert cryptocurrency market analyst. Analyze the provided news articles "
        "and determine the current overall market sentiment for Bitcoin.\n"
        "Return your response ONLY as valid JSON matching this schema:\n"
        "{\n"
        '  "sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",\n'
        '  "summary": "Brief 1-2 sentence overview of current market narrative",\n'
        '  "key_factors": ["Reason 1", "Reason 2", "Reason 3"],\n'
        '  "cited_sources": ["Title of Source 1", "Title of Source 2"]\n'
        "}"
    )

    user_prompt = f"Retrieved News Context:\n{context_str}\nProvide the market sentiment JSON analysis."


    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.2
    )

    result_json = json.loads(response.choices[0].message.content)
    
    print("\n--- LLM Sentiment Analysis Output ---")
    print(json.dumps(result_json, indent=2))
    
    return result_json

if __name__ == "__main__":
    analyze_sentiment_with_llm()