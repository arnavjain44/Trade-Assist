"""
Phase 4 Real News Processor & FinBERT Sentiment Engine

Provides:
1. NewsProvider interface + YFinanceNewsProvider + LocalNewsCacheProvider.
2. FinBERTSentimentEngine (HuggingFace ProsusAI/finbert model).
3. HistoricalNewsAggregator enforcing strict temporal leakage prevention:
   news_timestamp < candle_timestamp.
   Missing news explicitly returns has_news = False and sentiment fields = NaN.
"""

from abc import ABC, abstractmethod
import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.utc


class NewsArticle:
    """Dataclass representing a real news article with timezone-aware publication timestamp."""

    def __init__(
        self,
        article_id: str,
        symbol: str,
        headline: str,
        summary: str,
        pub_timestamp_ist: datetime.datetime,
        provider: str = "",
        url: str = "",
    ):
        self.article_id = article_id
        self.symbol = symbol.upper().strip()
        self.headline = headline.strip()
        self.summary = summary.strip()
        if pub_timestamp_ist.tzinfo is None:
            self.pub_timestamp_ist = IST.localize(pub_timestamp_ist)
        else:
            self.pub_timestamp_ist = pub_timestamp_ist.astimezone(IST)
        self.provider = provider
        self.url = url

    def to_dict(self) -> Dict[str, Any]:
        return {
            "article_id": self.article_id,
            "symbol": self.symbol,
            "headline": self.headline,
            "summary": self.summary,
            "pub_timestamp_ist": self.pub_timestamp_ist.isoformat(),
            "provider": self.provider,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NewsArticle":
        pub_dt = datetime.datetime.fromisoformat(data["pub_timestamp_ist"])
        return cls(
            article_id=data["article_id"],
            symbol=data["symbol"],
            headline=data["headline"],
            summary=data.get("summary", ""),
            pub_timestamp_ist=pub_dt,
            provider=data.get("provider", ""),
            url=data.get("url", ""),
        )


class NewsProvider(ABC):
    """Abstract Base Class for News Providers."""

    @abstractmethod
    def fetch_news_for_symbol(self, symbol: str) -> List[NewsArticle]:
        """Fetch news articles for a specific stock symbol."""
        pass


class YFinanceNewsProvider(NewsProvider):
    """Fetches real news articles using yfinance API."""

    def fetch_news_for_symbol(self, symbol: str) -> List[NewsArticle]:
        import yfinance as yf

        articles: List[NewsArticle] = []
        try:
            ticker = yf.Ticker(symbol)
            raw_news = ticker.news
            if not raw_news:
                return []

            for item in raw_news:
                content = item.get("content", {})
                if not content:
                    continue

                title = content.get("title", "").strip()
                if not title:
                    continue

                summary = content.get("summary", "").strip()
                pub_date_str = content.get("pubDate") or content.get("displayTime")

                if pub_date_str:
                    try:
                        dt_utc = datetime.datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                        dt_ist = dt_utc.astimezone(IST)
                    except Exception:
                        dt_ist = datetime.datetime.now(IST)
                else:
                    pub_time_sec = item.get("providerPublishTime")
                    if pub_time_sec:
                        dt_ist = datetime.datetime.fromtimestamp(pub_time_sec, tz=UTC).astimezone(IST)
                    else:
                        dt_ist = datetime.datetime.now(IST)

                article_id = item.get("id") or hashlib.md5(f"{symbol}_{title}_{dt_ist.isoformat()}".encode()).hexdigest()
                provider_name = content.get("provider", {}).get("displayName", "")
                click_url = content.get("clickThroughUrl", {}).get("url", "")

                articles.append(
                    NewsArticle(
                        article_id=article_id,
                        symbol=symbol,
                        headline=title,
                        summary=summary,
                        pub_timestamp_ist=dt_ist,
                        provider=provider_name,
                        url=click_url,
                    )
                )
        except Exception as exc:
            logger.warning("YFinanceNewsProvider error fetching news for %s: %s", symbol, exc)

        return articles


class LocalNewsCacheProvider(NewsProvider):
    """File-backed persistent news provider for reproducibility and offline execution."""

    def __init__(self, cache_file: str = "data/processed/raw_news_cache.json", fallback_provider: Optional[NewsProvider] = None):
        self.cache_file = Path(cache_file)
        self.fallback_provider = fallback_provider or YFinanceNewsProvider()
        self._cache: Dict[str, List[NewsArticle]] = {}
        self._load_cache()

    def _load_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    for sym, art_list in raw_data.items():
                        self._cache[sym] = [NewsArticle.from_dict(d) for d in art_list]
            except Exception as exc:
                logger.warning("Error loading news cache from %s: %s", self.cache_file, exc)

    def _save_cache(self):
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            serializable = {
                sym: [a.to_dict() for a in articles]
                for sym, articles in self._cache.items()
            }
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
        except Exception as exc:
            logger.warning("Error saving news cache to %s: %s", self.cache_file, exc)

    def fetch_news_for_symbol(self, symbol: str) -> List[NewsArticle]:
        clean_sym = symbol.upper().strip()
        if clean_sym in self._cache and len(self._cache[clean_sym]) > 0:
            return self._cache[clean_sym]

        articles = self.fallback_provider.fetch_news_for_symbol(clean_sym)
        if articles:
            self._cache[clean_sym] = articles
            self._save_cache()
        return articles


class FinBERTSentimentEngine:
    """Uses Hugging Face ProsusAI/finbert pipeline to compute real probabilities & sentiment score."""

    def __init__(self, cache_file: str = "data/processed/finbert_sentiment_cache.json"):
        self.cache_file = Path(cache_file)
        self._cache: Dict[str, Dict[str, float]] = {}
        self._pipeline = None
        self._model_loaded = False
        self._load_cache()

    def _load_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception as exc:
                logger.warning("Error loading sentiment cache: %s", exc)

    def _save_cache(self):
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as exc:
            logger.warning("Error saving sentiment cache: %s", exc)

    def _ensure_pipeline(self):
        if not self._model_loaded:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

                model_name = "ProsusAI/finbert"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self._pipeline = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    return_all_scores=True,
                )
                self._model_loaded = True
                logger.info("FinBERTSentimentEngine loaded model '%s' successfully.", model_name)
            except Exception as exc:
                logger.error("Failed to load FinBERT model: %s", exc)
                raise RuntimeError(f"FinBERT model initialization failed: {exc}")

    def analyze_text(self, text: str) -> Dict[str, float]:
        text_clean = text.strip()
        if not text_clean:
            return {
                "positive_probability": float("nan"),
                "negative_probability": float("nan"),
                "neutral_probability": float("nan"),
                "sentiment_score": float("nan"),
            }

        text_hash = hashlib.md5(text_clean.encode("utf-8")).hexdigest()
        if text_hash in self._cache:
            return self._cache[text_hash]

        self._ensure_pipeline()
        raw_res = self._pipeline(text_clean[:512])

        if isinstance(raw_res, list) and len(raw_res) > 0:
            if isinstance(raw_res[0], list):
                items = raw_res[0]
            else:
                items = raw_res
        else:
            items = []

        prob_map = {item["label"].lower(): float(item["score"]) for item in items if isinstance(item, dict)}
        pos = prob_map.get("positive", 0.0)
        neg = prob_map.get("negative", 0.0)
        neu = prob_map.get("neutral", 0.0)

        sentiment_score = round(pos - neg, 4)

        result_dict = {
            "positive_probability": round(pos, 4),
            "negative_probability": round(neg, 4),
            "neutral_probability": round(neu, 4),
            "sentiment_score": sentiment_score,
        }

        self._cache[text_hash] = result_dict
        self._save_cache()
        return result_dict


class HistoricalNewsAggregator:
    """Aggregates news for trading candles strictly enforcing news_timestamp < candle_timestamp."""

    @staticmethod
    def aggregate_news_for_candle(
        articles_with_sentiment: List[Tuple[NewsArticle, Dict[str, float]]],
        candle_timestamp_ist: datetime.datetime,
    ) -> Dict[str, Any]:
        """Filters articles where pub_timestamp_ist < candle_timestamp_ist.

        Returns aggregated dictionary. When no news exists, sentiment numeric fields are explicitly NaN.
        """
        if candle_timestamp_ist.tzinfo is None:
            candle_dt_ist = IST.localize(candle_timestamp_ist)
        else:
            candle_dt_ist = candle_timestamp_ist.astimezone(IST)

        eligible: List[Tuple[NewsArticle, Dict[str, float]]] = []
        for art, sent in articles_with_sentiment:
            # STRICT TEMPORAL RULE: news_timestamp < candle_timestamp
            if art.pub_timestamp_ist < candle_dt_ist:
                eligible.append((art, sent))

        if not eligible:
            return {
                "number_of_articles": 0,
                "mean_sentiment": float("nan"),
                "positive_probability_mean": float("nan"),
                "negative_probability_mean": float("nan"),
                "neutral_probability_mean": float("nan"),
                "latest_news_timestamp": None,
                "has_news": False,
            }

        n = len(eligible)
        mean_sentiment = sum(sent["sentiment_score"] for _, sent in eligible) / n
        pos_mean = sum(sent["positive_probability"] for _, sent in eligible) / n
        neg_mean = sum(sent["negative_probability"] for _, sent in eligible) / n
        neu_mean = sum(sent["neutral_probability"] for _, sent in eligible) / n
        latest_ts = max(art.pub_timestamp_ist for art, _ in eligible).isoformat()

        return {
            "number_of_articles": n,
            "mean_sentiment": round(mean_sentiment, 4),
            "positive_probability_mean": round(pos_mean, 4),
            "negative_probability_mean": round(neg_mean, 4),
            "neutral_probability_mean": round(neu_mean, 4),
            "latest_news_timestamp": latest_ts,
            "has_news": True,
        }
