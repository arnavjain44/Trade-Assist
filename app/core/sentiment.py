import random
from typing import List, Dict, Any
from app.schemas.responses import SentimentItem

class NewsSentimentAnalyzer:
    """Fetches news headlines and calculates sentiment score per stock (FinBERT / VADER)."""

    def __init__(self):
        # Try loading VADER sentiment analyzer if available
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
        except ImportError:
            self.vader = None

    def analyze_headlines(self, symbol: str) -> List[SentimentItem]:
        """Fetches and scores recent news headlines for a given stock symbol."""
        # Simulated news fetching from NewsAPI or stock-data provider
        clean_symbol = symbol.split('.')[0]
        sample_headlines = [
            f"{clean_symbol} reports strong quarterly revenue growth and expansion plans.",
            f"Analysts issue positive outlook for {clean_symbol} following strategic partnership.",
            f"Market volatility impacts short-term momentum for {clean_symbol}.",
            f"{clean_symbol} increases institutional investments in core business sectors.",
            f"Regulatory scrutiny noted across sector, {clean_symbol} maintaining steady operations."
        ]

        items = []
        for i, text in enumerate(sample_headlines):
            if self.vader:
                scores = self.vader.polarity_scores(text)
                compound = scores['compound']
                if compound >= 0.05:
                    label = "POSITIVE"
                elif compound <= -0.05:
                    label = "NEGATIVE"
                else:
                    label = "NEUTRAL"
            else:
                # Rule-based fallback sentiment scoring
                words = text.lower().split()
                pos_words = {"strong", "positive", "growth", "expansion", "steady", "increases"}
                neg_words = {"volatility", "impacts", "scrutiny", "decline", "fall"}
                pos_cnt = sum(1 for w in words if w in pos_words)
                neg_cnt = sum(1 for w in words if w in neg_words)
                if pos_cnt > neg_cnt:
                    label = "POSITIVE"
                    compound = 0.65
                elif neg_cnt > pos_cnt:
                    label = "NEGATIVE"
                    compound = -0.50
                else:
                    label = "NEUTRAL"
                    compound = 0.0

            items.append(SentimentItem(
                headline=text,
                sentiment=label,
                score=round(compound, 2),
                date="Today"
            ))
        return items

    def calculate_aggregated_sentiment(self, sentiments: List[SentimentItem]) -> float:
        """Averages headline sentiment scores into a single daily numeric feature (-1.0 to +1.0)."""
        if not sentiments:
            return 0.0
        total_score = sum(s.score for s in sentiments)
        return round(total_score / len(sentiments), 2)

sentiment_analyzer = NewsSentimentAnalyzer()
