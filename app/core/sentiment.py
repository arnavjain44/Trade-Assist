"""
Core Sentiment Analysis Module

Updated for Phase 4:
Integrates real FinBERT sentiment analyzer (ProsusAI/finbert) and real news fetching.
Maintains backward compatibility for API endpoints.
"""

from typing import List, Dict, Any, Optional
import datetime
import pytz
from app.schemas.responses import SentimentItem
from app.ml.news_processor import (
    LocalNewsCacheProvider,
    FinBERTSentimentEngine,
    NewsArticle,
    IST,
)


class NewsSentimentAnalyzer:
    """Fetches real news headlines and calculates sentiment score per stock using FinBERT."""

    def __init__(self):
        self.news_provider = LocalNewsCacheProvider()
        self.finbert_engine = FinBERTSentimentEngine()

    def analyze_headlines(self, symbol: str) -> List[SentimentItem]:
        """Fetches and scores real news headlines for a given stock symbol using FinBERT."""
        clean_symbol = symbol.upper().strip()
        articles = self.news_provider.fetch_news_for_symbol(clean_symbol)

        if not articles:
            return []

        items = []
        for art in articles[:5]:  # Return top 5 recent articles
            sent = self.finbert_engine.analyze_text(art.headline)
            score = sent["sentiment_score"]
            if score >= 0.05:
                label = "POSITIVE"
            elif score <= -0.05:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"

            date_str = art.pub_timestamp_ist.strftime("%Y-%m-%d %H:%M IST")
            items.append(
                SentimentItem(
                    headline=art.headline,
                    sentiment=label,
                    score=round(score, 2),
                    date=date_str,
                )
            )

        return items

    def calculate_aggregated_sentiment(self, sentiments: List[SentimentItem]) -> float:
        """Averages headline sentiment scores into a single numeric feature (-1.0 to +1.0)."""
        if not sentiments:
            return 0.0
        total_score = sum(s.score for s in sentiments)
        return round(total_score / len(sentiments), 2)

    def get_phase5_news_features(
        self, symbol: str, decision_timestamp_ist: Optional[datetime.datetime] = None
    ) -> Dict[str, Any]:
        """Extracts the exact Phase 5 news features:
        - sentiment_score: mean(pos - neg) rounded to 4 decimals, or float('nan') if no eligible articles
        - has_news: bool (True if eligible articles exist, False otherwise)
        - number_of_articles: int (count of eligible articles)

        Causal guarantee: Only articles with pub_timestamp_ist < decision_timestamp_ist are used.
        """
        clean_symbol = symbol.upper().strip()
        articles = self.news_provider.fetch_news_for_symbol(clean_symbol)

        if decision_timestamp_ist is None:
            decision_timestamp_ist = datetime.datetime.now(IST)
        elif decision_timestamp_ist.tzinfo is None:
            decision_timestamp_ist = IST.localize(decision_timestamp_ist)
        else:
            decision_timestamp_ist = decision_timestamp_ist.astimezone(IST)

        eligible = [a for a in articles if a.pub_timestamp_ist < decision_timestamp_ist]

        if not eligible:
            return {
                "sentiment_score": float("nan"),
                "has_news": False,
                "number_of_articles": 0,
            }

        scores = []
        for art in eligible:
            sent = self.finbert_engine.analyze_text(art.headline)
            scores.append(sent["sentiment_score"])

        return {
            "sentiment_score": round(sum(scores) / len(scores), 4),
            "has_news": True,
            "number_of_articles": len(eligible),
        }


sentiment_analyzer = NewsSentimentAnalyzer()
