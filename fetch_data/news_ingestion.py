import os
import feedparser
import pandas as pd
from bs4 import BeautifulSoup

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_SAVE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "btc_news.csv")


RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cryptoslate.com/feed/",
    "https://news.bitcoin.com/feed/",
    "https://decrypt.co/feed"
]

def clean_html(raw_html):

    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text()

def fetch_latest_crypto_news(feeds=RSS_FEEDS, save_path=NEWS_SAVE_PATH):
   
    articles = []
    
    print("Fetching crypto news from RSS feeds...")
    for feed_url in feeds:
        parsed_feed = feedparser.parse(feed_url)
        print(f"Feed: {feed_url} | Found {len(parsed_feed.entries)} items")
        
        for entry in parsed_feed.entries:
            title = entry.get("title", "")
            summary = clean_html(entry.get("summary", entry.get("description", "")))
            published = entry.get("published", entry.get("updated", ""))
            link = entry.get("link", "")
            
            full_text = f"Title: {title}. Summary: {summary}"
            
            articles.append({
                "title": title,
                "text": full_text,
                "published": published,
                "url": link
            })
            
    df = pd.DataFrame(articles)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} news articles to: {save_path}")
    return df

if __name__ == "__main__":
    fetch_latest_crypto_news()