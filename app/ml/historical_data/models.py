"""
Historical Data Acquisition Models & Exceptions (Phase 5.4)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


class HistoricalDataError(Exception):
    """Base exception for all historical data acquisition errors."""
    pass


class AuthenticationRequiredError(HistoricalDataError):
    """Raised when an authorized data provider requires credentials that are not configured."""
    pass


class HistoricalRangeExceededError(HistoricalDataError):
    """Raised when a data provider cannot supply the requested historical lookback range."""
    pass


class RateLimitExceededError(HistoricalDataError):
    """Raised when an API rate limit is exceeded."""
    pass


class DataValidationError(HistoricalDataError):
    """Raised when downloaded data fails critical schema or mathematical validation."""
    pass


@dataclass
class HistoricalRequest:
    """Request specification for historical market data acquisition."""
    symbol: str
    start_date: datetime
    end_date: datetime
    interval: str = "5m"
    chunk_size_days: Optional[int] = None
    adjusted: bool = True


@dataclass
class ChunkInfo:
    """Details for an individual pagination/chunk request."""
    chunk_index: int
    start_date: datetime
    end_date: datetime
    status: str = "PENDING"  # PENDING, SUCCESS, FAILED, SKIPPED
    rows_downloaded: int = 0
    error_message: Optional[str] = None


@dataclass
class AcquisitionReport:
    """Comprehensive audit report for a historical data acquisition job."""
    provider: str
    symbol: str
    requested_range: Tuple[str, str]
    actual_range: Optional[Tuple[str, str]] = None
    chunks_requested: int = 0
    chunks_successful: int = 0
    chunks_failed: int = 0
    total_raw_rows: int = 0
    total_clean_rows: int = 0
    checksum_sha256: str = ""
    status: str = "PENDING"  # SUCCESS, PARTIAL, FAILED, DRY_RUN, BLOCKED_AUTH
    raw_storage_path: Optional[str] = None
    clean_storage_path: Optional[str] = None
    validation_audit: Optional[Dict[str, Any]] = None
    notes: str = ""
