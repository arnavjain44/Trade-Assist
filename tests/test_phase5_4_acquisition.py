"""
Unit Tests for Phase 5.4: Historical Data Acquisition & Provider Infrastructure

Verifies:
1. Provider adapter interface compliance
2. Deterministic date chunking
3. Chunk boundary handling
4. Duplicate removal across boundaries
5. Timestamp normalization (start -> close/end semantics)
6. IST timezone conversion
7. Canonical schema enforcement
8. Invalid OHLC rejection
9. Incomplete session detection & classification
10. Holiday and special session handling (Muhurat & Saturday DR)
11. Raw data immutability & checksum verification
12. Missing credential safe behavior (no fabrication)
13. Failed provider request retry and graceful reporting
14. Deterministic sorting and ordering
15. Temporal causality contract
"""

import os
import json
import tempfile
from datetime import datetime, date, time, timedelta, timezone
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from app.ml.historical_data.base import BaseHistoricalProvider
from app.ml.historical_data.models import (
    HistoricalRequest,
    AcquisitionReport,
    AuthenticationRequiredError,
    HistoricalRangeExceededError,
    HistoricalDataError,
    RateLimitExceededError,
)
from app.ml.historical_data.storage import RawStorageManager
from app.ml.historical_data.universe import HistoricalUniverseManager
from app.ml.historical_data.downloader import HistoricalDownloader
from app.ml.historical_data.providers.kite_adapter import KiteHistoricalProvider
from app.ml.historical_data.providers.yfinance_adapter import YFinanceHistoricalProvider
from app.ml.historical_data.providers.local_csv_adapter import LocalCsvHistoricalProvider
from app.ml.historical_data.providers.free_huggingface_adapter import FreeHuggingFaceHistoricalProvider
from app.ml.historical_data_validator import HistoricalDataValidator


# ---------------------------------------------------------------------------
# Test Fixtures & Mock Providers
# ---------------------------------------------------------------------------

class MockProvider(BaseHistoricalProvider):
    """Mock provider for unit testing pagination, chunking, and retries."""

    def __init__(self, fail_attempts: int = 0, authenticated: bool = True):
        self._fail_attempts = fail_attempts
        self._attempts = 0
        self._authenticated = authenticated

    @property
    def provider_name(self) -> str:
        return "mock_provider"

    @property
    def max_chunk_days(self) -> int:
        return 100

    def is_authenticated(self) -> bool:
        return self._authenticated

    def validate_credentials(self) -> None:
        if not self._authenticated:
            raise AuthenticationRequiredError("Mock credentials missing.")

    def fetch_chunk(self, symbol: str, start_date: datetime, end_date: datetime, interval: str = "5m") -> pd.DataFrame:
        self.validate_credentials()
        self._attempts += 1
        if self._attempts <= self._fail_attempts:
            raise RateLimitExceededError(f"Transient error on attempt {self._attempts}")

        # Return a synthetic chunk with 10 bars
        dates = pd.date_range(start_date, periods=10, freq="5min", tz="Asia/Kolkata")
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": symbol,
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000.0,
            "trading_date": dates.strftime("%Y-%m-%d"),
            "source": self.provider_name,
        })
        return df

    def normalize_to_canonical(self, raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if raw_df.empty:
            return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume", "trading_date", "source"])
        df = raw_df.copy()
        df["symbol"] = symbol
        df["source"] = self.provider_name
        return df[["timestamp", "symbol", "open", "high", "low", "close", "volume", "trading_date", "source"]]


# ---------------------------------------------------------------------------
# 1. Provider Adapter Interface Compliance
# ---------------------------------------------------------------------------

def test_provider_adapter_interface():
    """Verify all provider adapters implement the BaseHistoricalProvider interface."""
    kite = KiteHistoricalProvider()
    assert kite.provider_name == "kite"
    assert kite.max_chunk_days == 100

    yf_prov = YFinanceHistoricalProvider()
    assert yf_prov.provider_name == "yfinance"
    assert yf_prov.max_chunk_days == 60

    local_csv = LocalCsvHistoricalProvider()
    assert local_csv.provider_name == "local_csv"
    assert local_csv.max_chunk_days == 3650


# ---------------------------------------------------------------------------
# 2. Date Chunking
# ---------------------------------------------------------------------------

def test_date_chunking():
    """Verify generate_chunks produces contiguous chunks with zero gaps and correct boundaries."""
    start = datetime(2023, 1, 1)
    end = datetime(2023, 8, 1)  # 212 days
    chunk_days = 100

    chunks = HistoricalDownloader.generate_chunks(start, end, chunk_days)
    assert len(chunks) == 3
    # Chunk 1: Jan 1 -> Apr 11 (100 days)
    assert chunks[0][0] == start
    assert chunks[0][1] == start + timedelta(days=100)
    # Chunk 2: Apr 11 -> Jul 20 (100 days)
    assert chunks[1][0] == chunks[0][1]
    assert chunks[1][1] == start + timedelta(days=200)
    # Chunk 3: Jul 20 -> Aug 1 (12 days)
    assert chunks[2][0] == chunks[1][1]
    assert chunks[2][1] == end


# ---------------------------------------------------------------------------
# 3. Chunk Boundary Handling
# ---------------------------------------------------------------------------

def test_chunk_boundary_handling():
    """Verify boundary timestamps do not cause duplicate intervals or data loss."""
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)
    chunks = HistoricalDownloader.generate_chunks(start, end, chunk_size_days=1)
    assert len(chunks) == 2
    assert chunks[0][1] == chunks[1][0]


# ---------------------------------------------------------------------------
# 4. Duplicate Removal
# ---------------------------------------------------------------------------

def test_duplicate_removal():
    """Verify downloader removes duplicate (symbol, timestamp) rows across chunks."""
    provider = MockProvider()
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = RawStorageManager(raw_dir=os.path.join(tmp_dir, "raw"), clean_dir=os.path.join(tmp_dir, "clean"))
        downloader = HistoricalDownloader(provider=provider, storage_manager=storage)

        # Download symbol with a small range
        start = datetime(2024, 1, 1, 9, 15)
        end = datetime(2024, 1, 1, 10, 0)
        report = downloader.download_symbol("TCS.NS", start, end)

        assert report.status == "SUCCESS"
        assert report.total_raw_rows > 0
        # Load saved raw file and check for duplicates
        raw_df, _ = storage.load_raw_download(report.raw_storage_path)
        assert not raw_df.duplicated(subset=["symbol", "timestamp"]).any()


# ---------------------------------------------------------------------------
# 5. Timestamp Normalization (Start to Close/End Semantics)
# ---------------------------------------------------------------------------

def test_timestamp_normalization():
    """Verify Kite adapter shifts candle start time forward by 5m to canonical close/end time."""
    kite = KiteHistoricalProvider()
    raw_df = pd.DataFrame({
        "date": ["2024-01-01 09:15:00+05:30", "2024-01-01 09:20:00+05:30"],
        "open": [2500.0, 2505.0],
        "high": [2510.0, 2515.0],
        "low": [2495.0, 2500.0],
        "close": [2505.0, 2510.0],
        "volume": [1000, 1500],
    })

    canonical_df = kite.normalize_to_canonical(raw_df, "RELIANCE.NS")
    # 09:15 start time must become 09:20 close time
    assert canonical_df.loc[0, "timestamp"] == pd.Timestamp("2024-01-01 09:20:00+05:30")
    assert canonical_df.loc[1, "timestamp"] == pd.Timestamp("2024-01-01 09:25:00+05:30")


# ---------------------------------------------------------------------------
# 6. IST Conversion
# ---------------------------------------------------------------------------

def test_ist_conversion():
    """Verify naive and UTC timestamps are accurately converted to Asia/Kolkata."""
    kite = KiteHistoricalProvider()
    # UTC timestamp representing 03:45 UTC (which is 09:15 IST)
    raw_df = pd.DataFrame({
        "date": ["2024-01-01 03:45:00+00:00"],
        "open": [100.0], "high": [105.0], "low": [95.0], "close": [102.0], "volume": [500.0]
    })
    canonical_df = kite.normalize_to_canonical(raw_df, "INFY.NS")
    # 03:45 UTC is 09:15 IST. Adding 5m gives 09:20 IST.
    ts = canonical_df.loc[0, "timestamp"]
    assert str(ts.tz) in ["Asia/Kolkata", "+05:30", "UTC+05:30"]
    assert ts.hour == 9
    assert ts.minute == 20


# ---------------------------------------------------------------------------
# 7. Canonical Schema Enforcement
# ---------------------------------------------------------------------------

def test_canonical_schema():
    """Verify canonical column structure and data types."""
    kite = KiteHistoricalProvider()
    raw_df = pd.DataFrame({
        "date": ["2024-01-01 09:15:00+05:30"],
        "open": [100.0], "high": [105.0], "low": [95.0], "close": [102.0], "volume": [500.0]
    })
    canonical = kite.normalize_to_canonical(raw_df, "HDFCBANK.NS")
    expected_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "trading_date", "source"]
    assert list(canonical.columns) == expected_cols
    assert canonical.loc[0, "symbol"] == "HDFCBANK.NS"
    assert canonical.loc[0, "source"] == "kite"
    assert canonical.loc[0, "trading_date"] == "2024-01-01"


# ---------------------------------------------------------------------------
# 8. Invalid OHLC Rejection
# ---------------------------------------------------------------------------

def test_invalid_ohlc_rejection():
    """Verify HistoricalDataValidator rejects bars with impossible OHLC relationships."""
    validator = HistoricalDataValidator()
    dates = pd.date_range("2024-01-01 09:15:00+05:30", periods=3, freq="5min")
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "SBIN.NS",
        "open": [500.0, 500.0, 500.0],
        "high": [510.0, 490.0, 510.0],  # Bar 1 high < open (impossible)
        "low": [490.0, 480.0, 520.0],   # Bar 2 low > open (impossible)
        "close": [505.0, 485.0, 505.0],
        "volume": [1000.0, 1000.0, 1000.0],
    })

    clean_df, audit = validator.validate(df)
    assert len(clean_df) == 1
    assert audit["violations"]["invalid_ohlc_math"] == 2


# ---------------------------------------------------------------------------
# 9. Incomplete Session Detection
# ---------------------------------------------------------------------------

def test_incomplete_session_detection():
    """Verify validator identifies sessions with fewer than 75 bars as partial sessions."""
    validator = HistoricalDataValidator()
    # 20 bars instead of 75
    dates = pd.date_range("2024-01-01 09:15:00+05:30", periods=20, freq="5min")
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "ITC.NS",
        "open": 400.0, "high": 405.0, "low": 395.0, "close": 402.0, "volume": 500.0
    })
    clean_df, audit = validator.validate(df)
    assert audit["session_summary"]["partial_sessions_count"] == 1
    assert audit["session_summary"]["standard_sessions_count"] == 0


# ---------------------------------------------------------------------------
# 10. Holiday and Special Session Handling
# ---------------------------------------------------------------------------

def test_special_session_handling():
    """Verify Diwali Muhurat trading evening sessions are accepted without false off-hours rejection."""
    validator = HistoricalDataValidator(allow_special_sessions=True)
    # Diwali Muhurat 2024: Nov 1, 2024 from 18:00 to 19:00 IST (12 bars)
    dates = pd.date_range("2024-11-01 18:00:00+05:30", periods=12, freq="5min")
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "RELIANCE.NS",
        "open": 2500.0, "high": 2510.0, "low": 2490.0, "close": 2505.0, "volume": 1000.0
    })

    clean_df, audit = validator.validate(df)
    assert len(clean_df) == 12
    assert audit["violations"].get("off_hours_or_weekend_contamination", 0) == 0
    assert audit["session_summary"]["special_sessions_count"] == 1


# ---------------------------------------------------------------------------
# 11. Raw Data Immutability & Checksum Verification
# ---------------------------------------------------------------------------

def test_raw_data_immutability():
    """Verify RawStorageManager computes SHA-256 and detects tampering."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = RawStorageManager(raw_dir=os.path.join(tmp_dir, "raw"), clean_dir=os.path.join(tmp_dir, "clean"))
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="5min"),
            "symbol": "TCS.NS",
            "open": [10.0]*5, "high": [12.0]*5, "low": [9.0]*5, "close": [11.0]*5, "volume": [100.0]*5
        })

        p_path, m_path, sha_hash = storage.save_raw_download(
            df, "TCS.NS", "5m", datetime(2024, 1, 1), datetime(2024, 1, 2), "test_provider"
        )
        assert os.path.exists(p_path)
        assert os.path.exists(m_path)
        assert len(sha_hash) == 64

        # Verify load succeeds
        loaded_df, meta = storage.load_raw_download(p_path, verify_checksum=True)
        assert len(loaded_df) == 5
        assert meta["sha256_checksum"] == sha_hash

        # Tamper with the parquet file
        tampered_df = df.copy()
        tampered_df.loc[0, "close"] = 999.0
        tampered_df.to_parquet(p_path, index=False)

        # Checksum verification must catch tampering
        with pytest.raises(ValueError, match="Integrity check failed"):
            storage.load_raw_download(p_path, verify_checksum=True)


# ---------------------------------------------------------------------------
# 12. Missing Credential Behavior (No Fabrication)
# ---------------------------------------------------------------------------

def test_missing_credential_behavior():
    """Verify Kite adapter fails loudly when credentials are unconfigured."""
    kite = KiteHistoricalProvider(api_key=None, access_token=None)
    assert not kite.is_authenticated()

    with pytest.raises(AuthenticationRequiredError, match="Zerodha Kite Connect credentials not configured"):
        kite.validate_credentials()

    with pytest.raises(AuthenticationRequiredError):
        kite.fetch_chunk("RELIANCE.NS", datetime(2024, 1, 1), datetime(2024, 1, 2))


# ---------------------------------------------------------------------------
# 13. Failed Provider Request Behavior & Retries
# ---------------------------------------------------------------------------

def test_failed_provider_request_behavior():
    """Verify downloader retries transient errors and reports failure status."""
    # Provider configured to fail first 2 attempts, succeed on 3rd
    provider = MockProvider(fail_attempts=2)
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = RawStorageManager(raw_dir=os.path.join(tmp_dir, "raw"), clean_dir=os.path.join(tmp_dir, "clean"))
        downloader = HistoricalDownloader(
            provider=provider, storage_manager=storage, max_retries=3, retry_delay_seconds=0.01
        )
        report = downloader.download_symbol("INFY.NS", datetime(2024, 1, 1), datetime(2024, 1, 2))
        assert report.status == "SUCCESS"
        assert report.chunks_successful == 1

    # Provider configured to fail permanently (4 attempts with max_retries=2)
    failing_provider = MockProvider(fail_attempts=10)
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = RawStorageManager(raw_dir=os.path.join(tmp_dir, "raw"), clean_dir=os.path.join(tmp_dir, "clean"))
        downloader = HistoricalDownloader(
            provider=failing_provider, storage_manager=storage, max_retries=2, retry_delay_seconds=0.01
        )
        report = downloader.download_symbol("INFY.NS", datetime(2024, 1, 1), datetime(2024, 1, 2))
        assert report.status == "FAILED"
        assert report.chunks_failed == 1


# ---------------------------------------------------------------------------
# 14. Deterministic Ordering
# ---------------------------------------------------------------------------

def test_deterministic_ordering():
    """Verify output data is always strictly monotonically sorted by timestamp."""
    kite = KiteHistoricalProvider()
    # Provide out-of-order rows
    raw_df = pd.DataFrame({
        "date": ["2024-01-01 09:30:00+05:30", "2024-01-01 09:15:00+05:30", "2024-01-01 09:20:00+05:30"],
        "open": [10.0, 10.0, 10.0],
        "high": [12.0, 12.0, 12.0],
        "low": [8.0, 8.0, 8.0],
        "close": [11.0, 11.0, 11.0],
        "volume": [100.0, 100.0, 100.0],
    })
    canonical_df = kite.normalize_to_canonical(raw_df, "TCS.NS")
    timestamps = canonical_df["timestamp"].tolist()
    assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# 15. Temporal Contract Compliance
# ---------------------------------------------------------------------------

def test_temporal_contract_compliance():
    """Verify historical universe and causal timestamp bounds respect decision time T."""
    universe_mgr = HistoricalUniverseManager(mode="point_in_time")
    # Rebalance on 2024-09-27 added BEL and TRENT, removed DIVISLAB
    # Before rebalance (e.g. 2024-06-01): BEL should NOT be in universe, DIVISLAB should be in universe
    u_before, meta_before = universe_mgr.get_universe_at(date(2024, 6, 1))
    assert "DIVISLAB.NS" in u_before
    assert "BEL.NS" not in u_before
    assert not meta_before["survivorship_biased"]

    # Contemporary mode must flag survivorship bias
    contemporary_mgr = HistoricalUniverseManager(mode="contemporary_with_bias_warning")
    u_contemp, meta_contemp = contemporary_mgr.get_universe_at(date(2024, 6, 1))
    assert meta_contemp["survivorship_biased"]
    assert "BEL.NS" in u_contemp


# ---------------------------------------------------------------------------
# 16. Free Hugging Face Provider Tests (Phase 5.4a)
# ---------------------------------------------------------------------------

def test_free_huggingface_provider_interface_and_properties():
    """Verify FreeHuggingFaceHistoricalProvider conforms to zero-cost, unauthenticated contract."""
    provider = FreeHuggingFaceHistoricalProvider()
    assert provider.provider_name == "free_huggingface"
    assert provider.is_authenticated() is True
    assert provider.max_chunk_days == 365
    # validate_credentials should never raise because it is public/open
    provider.validate_credentials()

    # Shard index mapping
    assert provider.get_shard_index("AXISBANK") == 0
    assert provider.get_shard_index("INFY") == 2
    assert provider.get_shard_index("RELIANCE") == 5
    assert provider.get_shard_index("TCS") == 6
    assert provider.get_shard_index("WIPRO") == 7


def test_free_huggingface_provider_normalization():
    """Verify normalization and resampling logic of free Hugging Face provider."""
    provider = FreeHuggingFaceHistoricalProvider()

    # Raw 5m DataFrame with UTC timestamps
    raw_df = pd.DataFrame({
        "timestamp": [
            "2024-01-08 03:50:00",
            "2024-01-08 03:55:00",
        ],
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.5],
        "close": [101.5, 102.5],
        "volume": [500.0, 700.0],
    })

    canonical = provider.normalize_to_canonical(raw_df, "AXISBANK.NS")

    assert list(canonical.columns) == [
        "timestamp", "symbol", "open", "high", "low", "close", "volume", "trading_date", "source"
    ]
    assert canonical["source"].iloc[0] == "free_huggingface"
    assert canonical["symbol"].iloc[0] == "AXISBANK.NS"
    # UTC 03:50 is 09:20 IST (+05:30)
    assert canonical["timestamp"].iloc[0].tzname() in ["IST", "+0530", "Asia/Kolkata"]
    assert canonical["trading_date"].iloc[0] == "2024-01-08"
    assert canonical["open"].dtype == float
    assert canonical["volume"].dtype == float

