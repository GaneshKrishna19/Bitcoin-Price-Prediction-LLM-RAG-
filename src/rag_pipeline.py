import os
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "btc_news.csv")
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")

def build_and_query_vectorstore(query_text="What is the latest Bitcoin market sentiment and news?", top_k=5):
 
    if not os.path.exists(NEWS_DATA_PATH):
        raise FileNotFoundError(f"News dataset missing at {NEWS_DATA_PATH}. Run news_ingestion.py first.")


    df = pd.read_csv(NEWS_DATA_PATH)
    if df.empty:
        print("No news items found in dataset.")
        return []

  
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    

    collection = client.get_or_create_collection(
        name="btc_news_sentiment",
        embedding_function=embedding_fn
    )

    documents = df['text'].tolist()
    metadatas = [
        {"title": str(row['title']), "url": str(row['url']), "published": str(row['published'])} 
        for _, row in df.iterrows()
    ]
    ids = [f"news_{i}" for i in range(len(df))]

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Stored/updated {len(documents)} articles in ChromaDB at: {CHROMA_DB_PATH}")


    print(f"\nQuerying vector DB with: '{query_text}'")
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k
    )

   
    retrieved_chunks = []
    print("\n--- Top Retrieved Relevant Headlines ---")
    for idx, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        chunk_info = {
            "rank": idx + 1,
            "text": doc,
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "published": meta.get("published", "")
        }
        retrieved_chunks.append(chunk_info)
        print(f"[{idx+1}] {meta.get('title')}")
        print(f"    URL: {meta.get('url')}\n")

    return retrieved_chunks

if __name__ == "__main__":
    build_and_query_vectorstore()