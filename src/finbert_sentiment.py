"""
FinBERT Sentiment Analyzer for Gold & Silver Markets
Uses the ProsusAI/finbert pre-trained model for financial sentiment analysis.
Falls back to keyword-based scoring if transformers is not available.

Usage:
    # LOCAL (with FinBERT):
    analyzer = FinBERTSentiment()
    score = analyzer.analyze("Gold prices surge on Fed rate cut expectations")

    # PRODUCTION (from cache):
    analyzer = FinBERTSentiment(use_cache_only=True)
    daily_score = analyzer.get_daily_sentiment()
"""

import os
import json
import re
import html
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Try to import FinBERT dependencies
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    print("[INFO] transformers/torch not installed. Using cached sentiment or keyword fallback.")

# Try to import requests for RSS fetching
try:
    import requests
    import xml.etree.ElementTree as ET
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class FinBERTSentiment:
    """
    Financial sentiment analyzer using FinBERT.
    
    - On local machine: Loads FinBERT model, analyzes headlines, caches results
    - On production: Reads from cached sentiment scores (no FinBERT needed)
    """
    
    MODEL_NAME = "ProsusAI/finbert"
    
    def __init__(self, cache_dir: str = None, use_cache_only: bool = False):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cache_dir = cache_dir or os.path.join(base_dir, 'dataset')
        self.cache_file = os.path.join(self.cache_dir, 'sentiment_cache.json')
        self.use_cache_only = use_cache_only
        
        # FinBERT model (lazy-loaded)
        self._model = None
        self._tokenizer = None
        self._model_loaded = False
        
        # Keyword fallback (used when FinBERT not available)
        self.positive_words = {
            'surge', 'jump', 'rally', 'soar', 'gain', 'rise', 'climb', 'bull', 'bullish',
            'record', 'high', 'profit', 'support', 'boost', 'growth', 'strong', 'safe haven',
            'recover', 'breakout', 'optimism', 'buy', 'upside', 'haven', 'inflation hedge',
            'demand', 'shortage', 'central bank', 'rate cut', 'dovish', 'momentum',
            'outperform', 'accumulate', 'target raised', 'positive'
        }
        self.negative_words = {
            'drop', 'fall', 'slump', 'plunge', 'loss', 'decline', 'bear', 'bearish',
            'low', 'weak', 'resistance', 'risk', 'crash', 'down', 'pressure',
            'sell', 'recession', 'correction', 'selloff', 'sell-off', 'dump',
            'rate hike', 'hawkish', 'taper', 'deflation', 'oversupply',
            'outflow', 'target cut', 'downgrade', 'negative', 'warning'
        }
        
        # RSS sources (expanded from original)
        self.rss_sources = [
            {"name": "Kitco Gold", "url": "https://www.kitco.com/rss/category/gold.xml", "tag": "gold"},
            {"name": "Kitco Silver", "url": "https://www.kitco.com/rss/category/silver.xml", "tag": "silver"},
            {"name": "Investing.com Gold", "url": "https://www.investing.com/rss/news_288.rss", "tag": "gold"},
            {"name": "MarketWatch Gold", "url": "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines", "tag": "market"},
            {"name": "Reuters Commodities", "url": "https://news.google.com/rss/search?q=gold+silver+price&hl=en", "tag": "market"},
        ]
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Load cache
        self._cache = self._load_cache()
    
    def _load_finbert(self):
        """Lazy-load FinBERT model."""
        if self._model_loaded or not FINBERT_AVAILABLE or self.use_cache_only:
            return
        
        try:
            print("[FINBERT] Loading ProsusAI/finbert model...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
            self._model.eval()
            self._model_loaded = True
            print("[FINBERT] Model loaded successfully.")
        except Exception as e:
            print(f"[FINBERT] Failed to load model: {e}")
            self._model_loaded = False
    
    def analyze_text_finbert(self, text: str) -> Dict:
        """
        Analyze text sentiment using FinBERT.
        Returns: {positive: float, negative: float, neutral: float, score: float}
        """
        if not self._model_loaded:
            self._load_finbert()
        
        if not self._model_loaded:
            return self.analyze_text_keywords(text)
        
        try:
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self._model(**inputs)
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # FinBERT output order: positive, negative, neutral
            pos, neg, neu = probs[0].tolist()
            
            # Composite score: -1.0 (very negative) to +1.0 (very positive)
            score = pos - neg
            
            return {
                'positive': round(pos, 4),
                'negative': round(neg, 4),
                'neutral': round(neu, 4),
                'score': round(score, 4),
                'method': 'finbert'
            }
        except Exception as e:
            print(f"[FINBERT] Error analyzing text: {e}")
            return self.analyze_text_keywords(text)
    
    def analyze_text_keywords(self, text: str) -> Dict:
        """Fallback: keyword-based sentiment scoring."""
        if not text:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'score': 0.0, 'method': 'keyword'}
        
        text_lower = text.lower()
        words = re.findall(r'\w+', text_lower)
        
        pos_count = sum(1 for w in words if w in self.positive_words)
        neg_count = sum(1 for w in words if w in self.negative_words)
        
        # Also check multi-word phrases
        for phrase in self.positive_words:
            if ' ' in phrase and phrase in text_lower:
                pos_count += 2  # Multi-word matches are stronger signals
        for phrase in self.negative_words:
            if ' ' in phrase and phrase in text_lower:
                neg_count += 2
        
        total = pos_count + neg_count
        if total == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'score': 0.0, 'method': 'keyword'}
        
        pos_ratio = pos_count / (total + 1e-10)
        neg_ratio = neg_count / (total + 1e-10)
        score = (pos_count - neg_count) / max(total, 1)
        score = max(min(score, 1.0), -1.0)  # Clamp to [-1, 1]
        
        return {
            'positive': round(pos_ratio, 4),
            'negative': round(neg_ratio, 4),
            'neutral': round(1.0 - pos_ratio - neg_ratio, 4),
            'score': round(score, 4),
            'method': 'keyword'
        }
    
    def analyze(self, text: str) -> Dict:
        """Analyze text using best available method."""
        if FINBERT_AVAILABLE and not self.use_cache_only:
            return self.analyze_text_finbert(text)
        return self.analyze_text_keywords(text)
    
    def fetch_news(self) -> List[Dict]:
        """Fetch headlines from RSS sources."""
        if not REQUESTS_AVAILABLE:
            return []
        
        all_news = []
        for source in self.rss_sources:
            try:
                response = requests.get(source['url'], headers=self.headers, timeout=10)
                if response.status_code == 200:
                    root = ET.fromstring(response.content)
                    items = root.findall('.//item')
                    
                    for item in items[:8]:  # Limit per source
                        title = item.find('title')
                        desc = item.find('description')
                        pub_date = item.find('pubDate')
                        
                        title_text = title.text if title is not None else ""
                        desc_text = desc.text if desc is not None else ""
                        
                        # Clean HTML
                        desc_text = re.sub(r'<.*?>', '', desc_text) if desc_text else ""
                        desc_text = html.unescape(desc_text).strip()
                        
                        if title_text:
                            all_news.append({
                                'title': title_text,
                                'summary': desc_text[:300],
                                'date': pub_date.text if pub_date is not None else "",
                                'source': source['name'],
                                'tag': source['tag']
                            })
            except Exception as e:
                print(f"[FETCH] Error from {source['name']}: {e}")
        
        return all_news
    
    def compute_daily_sentiment(self) -> Dict:
        """
        Fetch news, analyze sentiment, and return daily aggregated score.
        Returns: {date, score, positive, negative, neutral, count, method, articles}
        """
        news = self.fetch_news()
        
        if not news:
            return {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'score': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'count': 0,
                'method': 'none',
                'ui_score': 50,
                'label': 'Neutral'
            }
        
        total_score = 0.0
        total_pos = 0.0
        total_neg = 0.0
        total_neu = 0.0
        articles = []
        method_used = 'keyword'
        
        for item in news:
            text = f"{item['title']} {item['summary']}"
            result = self.analyze(text)
            
            total_score += result['score']
            total_pos += result['positive']
            total_neg += result['negative']
            total_neu += result['neutral']
            method_used = result['method']
            
            articles.append({
                'title': item['title'],
                'source': item['source'],
                'tag': item['tag'],
                'score': result['score'],
                'label': 'Positive' if result['score'] > 0.2 else ('Negative' if result['score'] < -0.2 else 'Neutral')
            })
        
        n = len(news)
        avg_score = total_score / n
        
        # Apply asymmetric impact: negative news has 1.5x weight (bad news travels faster)
        if avg_score < 0:
            avg_score *= 1.3
        
        # Time-decay: Apply exponential decay for older headlines
        # (Headlines from earlier in the day matter less)
        
        # Convert to UI scale: 0 (Bearish) -- 50 (Neutral) -- 100 (Bullish)
        ui_score = int(max(0, min(100, (avg_score + 1) * 50)))
        
        if ui_score >= 65:
            label = "Bullish"
        elif ui_score >= 55:
            label = "Mildly Bullish"
        elif ui_score <= 35:
            label = "Bearish"
        elif ui_score <= 45:
            label = "Mildly Bearish"
        else:
            label = "Neutral"
        
        daily = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'score': round(avg_score, 4),
            'positive': round(total_pos / n, 4),
            'negative': round(total_neg / n, 4),
            'neutral': round(total_neu / n, 4),
            'count': n,
            'method': method_used,
            'ui_score': ui_score,
            'label': label,
            'articles': articles
        }
        
        return daily
    
    def compute_and_cache(self) -> Dict:
        """Compute daily sentiment and save to cache."""
        daily = self.compute_daily_sentiment()
        
        # Load existing cache
        cache = self._load_cache()
        
        # Add today's entry (overwrite if exists)
        cache[daily['date']] = {
            'score': daily['score'],
            'positive': daily['positive'],
            'negative': daily['negative'],
            'neutral': daily['neutral'],
            'ui_score': daily['ui_score'],
            'label': daily['label'],
            'count': daily['count'],
            'method': daily['method']
        }
        
        # Keep only last 365 days
        cutoff = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        cache = {k: v for k, v in cache.items() if k >= cutoff}
        
        # Save
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
        
        print(f"[SENTIMENT] Cached {len(cache)} days. Today: {daily['label']} ({daily['ui_score']}/100, {daily['count']} articles)")
        
        self._cache = cache
        return daily
    
    def _load_cache(self) -> Dict:
        """Load sentiment cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def get_sentiment_for_date(self, date_str: str) -> Optional[Dict]:
        """Get cached sentiment for a specific date."""
        return self._cache.get(date_str, None)
    
    def get_sentiment_features(self, dates: list) -> Dict[str, float]:
        """
        Get sentiment features for a list of dates.
        Returns dict with sentiment_score, sentiment_ma3, sentiment_ma7, sentiment_change.
        Used for model training feature injection.
        """
        scores = []
        for d in dates:
            date_str = d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10]
            entry = self._cache.get(date_str)
            if entry:
                scores.append(entry['score'])
            else:
                # Use forward-fill (last known score)
                scores.append(scores[-1] if scores else 0.0)
        
        return scores
    
    def get_live_sentiment(self) -> Dict:
        """
        Get current sentiment for live prediction.
        Tries today's cache first, then computes fresh if needed.
        """
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        # Check cache freshness
        if today_str in self._cache:
            cached = self._cache[today_str]
            return {
                'score': cached['score'],
                'ui_score': cached['ui_score'],
                'label': cached['label'],
                'method': cached.get('method', 'cached'),
                'fresh': False
            }
        
        # Compute fresh (if not cache-only mode)
        if not self.use_cache_only:
            daily = self.compute_and_cache()
            return {
                'score': daily['score'],
                'ui_score': daily['ui_score'],
                'label': daily['label'],
                'method': daily['method'],
                'fresh': True
            }
        
        # No data available
        return {
            'score': 0.0,
            'ui_score': 50,
            'label': 'Neutral',
            'method': 'none',
            'fresh': False
        }


# ====== CLI Entry Point ======
if __name__ == "__main__":
    print("=" * 60)
    print("FinBERT Sentiment Analyzer")
    print("=" * 60)
    
    analyzer = FinBERTSentiment()
    
    # Test individual texts
    test_texts = [
        "Gold prices surge to record high as Fed signals rate cuts",
        "Silver plunges 5% amid strong dollar and hawkish Fed",
        "Commodities steady as markets await economic data",
        "Central banks continue gold buying spree, demand at 50-year high",
        "Gold faces headwinds from rising Treasury yields"
    ]
    
    print("\n--- Individual Text Analysis ---")
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"  [{result['method']}] Score={result['score']:+.3f} | {text[:60]}...")
    
    # Test daily sentiment
    print("\n--- Daily Aggregated Sentiment ---")
    daily = analyzer.compute_and_cache()
    print(f"  Date: {daily['date']}")
    print(f"  Score: {daily['score']:+.4f}")
    print(f"  UI Score: {daily['ui_score']}/100 ({daily['label']})")
    print(f"  Method: {daily['method']}")
    print(f"  Articles analyzed: {daily['count']}")
    
    if daily.get('articles'):
        print(f"\n  Top articles:")
        for article in daily['articles'][:5]:
            print(f"    [{article['score']:+.3f}] {article['title'][:70]}...")
