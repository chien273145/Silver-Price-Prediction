"""
News Sentiment Analyzer for Gold & Silver
Fetches news from RSS feeds and calculates sentiment scores based on financial keywords.
"""
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import re
import html
import time
import random

class NewsFetcher:
    def __init__(self):
        self.sources = [
            {
                "name": "Kitco Gold",
                "url": "https://www.kitco.com/rss/category/gold.xml",
                "tag": "gold"
            },
            {
                "name": "Kitco Silver",
                "url": "https://www.kitco.com/rss/category/silver.xml",
                "tag": "silver"
            },
            {
                "name": "Investing.com Gold",
                "url": "https://www.investing.com/rss/news_288.rss",
                "tag": "gold"
            }
        ]
        # User agent to avoid 403s
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def fetch_feeds(self):
        all_news = []
        for source in self.sources:
            try:
                print(f"Fetching {source['name']}...", end=" ")
                response = requests.get(source['url'], headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    # Clean potential encoding issues
                    content = response.content
                    
                    try:
                        root = ET.fromstring(content)
                        
                        # Handle different RSS versions (usually channel/item)
                        items = root.findall('.//item')
                        print(f"Found {len(items)} items")
                        
                        for item in items[:5]: # Limit to 5 per source to keep it fast
                            title = item.find('title').text if item.find('title') is not None else ""
                            link = item.find('link').text if item.find('link') is not None else ""
                            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
                            description = item.find('description').text if item.find('description') is not None else ""
                            
                            # Clean HTML from description
                            description = self._clean_html(description)
                            
                            if title:
                                all_news.append({
                                    "title": title,
                                    "link": link,
                                    "date": pub_date,
                                    "summary": description,
                                    "source": source["name"],
                                    "tag": source["tag"]
                                })
                    except ET.ParseError:
                        print("XML Parse Error")
                else:
                    print(f"Failed ({response.status_code})")
                    
            except Exception as e:
                print(f"Error: {e}")
                
        return all_news
    
    def _clean_html(self, raw_html):
        if not raw_html:
            return ""
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        return html.unescape(cleantext).strip()

class SentimentAnalyzer:
    def __init__(self):
        # Financial Sentiment Dictionary
        self.positive_words = {
            'surge', 'jump', 'rally', 'soar', 'gain', 'rise', 'climb', 'bull', 'bullish', 
            'record', 'high', 'profit', 'support', 'boost', 'up', 'growth', 'strong', 
            'recover', 'breakout', 'optimism', 'buy'
        }
        self.negative_words = {
            'drop', 'fall', 'slump', 'plunge', 'loss', 'decline', 'bear', 'bearish', 
            'low', 'weak', 'resistance', 'risk', 'fear', 'crash', 'down', 'pressure', 
            'uncertainty', 'sell', 'inflation', 'recession', 'correction'
        }
        
    def analyze_text(self, text):
        if not text:
            return 0
            
        text = text.lower()
        words = re.findall(r'\w+', text)
        
        score = 0
        for word in words:
            if word in self.positive_words:
                score += 1
            elif word in self.negative_words:
                score -= 1
                
        # Normalize score between -1 and 1 (approx)
        # Assuming typical short headline/summary has max ~3-5 emotional words
        normalized = max(min(score / 3.0, 1.0), -1.0)
        return normalized

    def analyze_news(self, news_list):
        analyzed_news = []
        total_score = 0
        count = 0
        
        for item in news_list:
            # Analyze combined title and summary
            full_text = f"{item['title']} {item['summary']}"
            score = self.analyze_text(full_text)
            
            sentiment_label = "Neutral"
            if score >= 0.3: sentiment_label = "Positive"
            elif score <= -0.3: sentiment_label = "Negative"
            
            # Map -1 to 1 range -> 0 to 100 scale for UI
            # 0 (Negative) -> 50 (Neutral) -> 100 (Positive)
            ui_score = int((score + 1) * 50)
            
            item['sentiment_score'] = ui_score
            item['sentiment_label'] = sentiment_label
            analyzed_news.append(item)
            
            total_score += ui_score
            count += 1
            
        avg_score = int(total_score / count) if count > 0 else 50
        
        return {
            "overall_sentiment": avg_score,
            "overall_label": self._get_label(avg_score),
            "news": analyzed_news
        }
        
    def _get_label(self, score):
        if score >= 60: return "Bullish"
        if score <= 40: return "Bearish"
        return "Neutral"

# Simple test
if __name__ == "__main__":
    print("Testing News Sentiment...")
    fetcher = NewsFetcher()
    news = fetcher.fetch_feeds()
    
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_news(news)
    
    print(f"\nOverall Sentiment: {result['overall_sentiment']} ({result['overall_label']})")
    print(f"Analyzed {len(result['news'])} articles.")
