import os
import json
from groq import Groq
from rag_pipeline import build_and_query_vectorstore
from dotenv import load_dotenv

load_dotenv()

def analyze_sentiment_with_llm(query="What is the latest Bitcoin market sentiment and news?"):

    retrieved_articles = build_and_query_vectorstore(query_text=query, top_k=3)
    
    if not retrieved_articles:
        return {"sentiment": "NEUTRAL", "reasoning": "No news articles found."}


    context_str = ""
    for idx, article in enumerate(retrieved_articles, 1):
        title = article['title']
        content = article['text'][:200] + "..." if len(article['text']) > 200 else article['text']
        context_str += f"[{idx}] {title}\n{content}\n\n"


    system_prompt = (
        "You are an expert cryptocurrency market analyst. Analyze the provided news articles "
        "and determine the current overall market sentiment for Bitcoin.\n"
        "Return ONLY a valid JSON object with exactly these fields:\n"
        '{"sentiment": "BULLISH|BEARISH|NEUTRAL", "summary": "string", "key_factors": ["string"], "cited_sources": ["string"]}\n'
        "No extra text, no markdown, no explanation. Only the JSON object."
    )

    user_prompt = f"News articles:\n{context_str}\n\nOutput JSON only:"


    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    response = client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=2000
    )

    content = response.choices[0].message.content
    
    try:
        result_json = json.loads(content)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result_json = json.loads(json_match.group())
        else:
            raise ValueError(f"Failed to parse JSON from response: {content}")
    
    print("\n--- LLM Sentiment Analysis Output ---")
    print(json.dumps(result_json, indent=2))
    
    return result_json

if __name__ == "__main__":
    analyze_sentiment_with_llm()