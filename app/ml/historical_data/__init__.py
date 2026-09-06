"""
Historical Data Acquisition & Quality Management Package (Phase 5.4)
"""

from app.ml.historical_data.models import (
    HistoricalRequest,
    ChunkInfo,
    AcquisitionReport,
    HistoricalDataError,
    AuthenticationRequiredError,
    HistoricalRangeExceededError,
    RateLimitExceededError,
    DataValidationError,
)
from app.ml.historical_data.base import BaseHistoricalProvider
from app.ml.historical_data.storage import RawStorageManager
from app.ml.historical_data.universe import HistoricalUniverseManager
from app.ml.historical_data.downloader import HistoricalDownloader
from app.ml.historical_data.providers.kite_adapter import KiteHistoricalProvider
from app.ml.historical_data.providers.yfinance_adapter import YFinanceHistoricalProvider
from app.ml.historical_data.providers.local_csv_adapter import LocalCsvHistoricalProvider

__all__ = [
    "HistoricalRequest",
    "ChunkInfo",
    "AcquisitionReport",
    "HistoricalDataError",
    "AuthenticationRequiredError",
    "HistoricalRangeExceededError",
    "RateLimitExceededError",
    "DataValidationError",
    "BaseHistoricalProvider",
    "RawStorageManager",
    "HistoricalUniverseManager",
    "HistoricalDownloader",
    "KiteHistoricalProvider",
    "YFinanceHistoricalProvider",
    "LocalCsvHistoricalProvider",
]
