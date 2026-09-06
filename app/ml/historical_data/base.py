"""
Base Provider Interface for Historical Market Data Acquisition (Phase 5.4)
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd


class BaseHistoricalProvider(ABC):
    """
    Abstract Base Class for historical market data providers.
    Enforces a provider-agnostic contract for fetching, paginating, and normalizing data.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique identifier string for this provider (e.g. 'kite', 'truedata', 'local_csv')."""
        pass

    @property
    @abstractmethod
    def max_chunk_days(self) -> int:
        """Maximum calendar days allowed per single historical API request for 5m interval."""
        pass

    @abstractmethod
    def is_authenticated(self) -> bool:
        """Returns True if the provider has all required credentials configured."""
        pass

    @abstractmethod
    def validate_credentials(self) -> None:
        """
        Validates presence of required authentication credentials.
        Raises AuthenticationRequiredError if credentials are missing or invalid.
        """
        pass

    @abstractmethod
    def fetch_chunk(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "5m",
    ) -> pd.DataFrame:
        """
        Fetches a single chunk of historical data from the provider.
        Returns a raw DataFrame as returned by the underlying API.
        """
        pass

    @abstractmethod
    def normalize_to_canonical(self, raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Transforms raw provider DataFrame into Trade-Assist canonical schema:
        - timestamp (timezone-aware IST UTC+05:30, candle close/end semantics)
        - symbol (standardized equity ticker)
        - open, high, low, close (positive float)
        - volume (non-negative float)
        - trading_date (YYYY-MM-DD string)
        - source (provider name)
        """
        pass
