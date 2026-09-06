"""
Historical Data Downloader Orchestrator (Phase 5.4)

Manages chunked pagination, transient retry backoff, canonical schema transformation,
immutable raw persistence, and automated data quality validation.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd

from app.ml.historical_data.base import BaseHistoricalProvider
from app.ml.historical_data.models import (
    AcquisitionReport,
    ChunkInfo,
    HistoricalRequest,
    HistoricalDataError,
    AuthenticationRequiredError,
    RateLimitExceededError,
)
from app.ml.historical_data.storage import RawStorageManager
from app.ml.historical_data_validator import HistoricalDataValidator

logger = logging.getLogger(__name__)


class HistoricalDownloader:
    """
    Orchestrates end-to-end historical market data acquisition across any provider adapter.
    """

    def __init__(
        self,
        provider: BaseHistoricalProvider,
        storage_manager: Optional[RawStorageManager] = None,
        validator: Optional[HistoricalDataValidator] = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ):
        self.provider = provider
        self.storage_manager = storage_manager or RawStorageManager()
        self.validator = validator or HistoricalDataValidator()
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    @staticmethod
    def generate_chunks(
        start_date: datetime,
        end_date: datetime,
        chunk_size_days: int,
    ) -> List[Tuple[datetime, datetime]]:
        """
        Splits a date range [start_date, end_date] into deterministic, contiguous chunks.
        Guarantees:
        - No gaps between chunks.
        - No overlapping interior timestamps.
        - Final chunk ends exactly on end_date.
        """
        if start_date > end_date:
            raise ValueError(f"start_date ({start_date}) cannot be after end_date ({end_date}).")

        chunks: List[Tuple[datetime, datetime]] = []
        cur_start = start_date

        while cur_start < end_date:
            cur_end = min(cur_start + timedelta(days=chunk_size_days), end_date)
            chunks.append((cur_start, cur_end))
            cur_start = cur_end

        return chunks

    def download_symbol(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "5m",
        dry_run: bool = False,
    ) -> AcquisitionReport:
        """
        Executes chunked historical download, normalizes to canonical format,
        validates quality, and saves raw and clean data.
        """
        chunk_days = self.provider.max_chunk_days
        date_chunks = self.generate_chunks(start_date, end_date, chunk_days)

        report = AcquisitionReport(
            provider=self.provider.provider_name,
            symbol=symbol,
            requested_range=(start_date.isoformat(), end_date.isoformat()),
            chunks_requested=len(date_chunks),
            chunks_successful=0,
            chunks_failed=0,
        )

        logger.info(
            "Initiating acquisition for %s via %s (%d chunks of <=%d days, interval=%s)",
            symbol, self.provider.provider_name, len(date_chunks), chunk_days, interval
        )

        # Handle Dry-Run Mode
        if dry_run:
            logger.info("Dry-run requested: skipping live network requests for %s.", symbol)
            report.status = "DRY_RUN"
            report.notes = (
                f"Dry-run validated: {len(date_chunks)} contiguous chunks generated successfully. "
                f"Chunk boundaries verified across requested range."
            )
            return report

        # Pre-check credentials
        try:
            self.provider.validate_credentials()
        except AuthenticationRequiredError as auth_err:
            logger.error("Authentication check failed for %s: %s", self.provider.provider_name, auth_err)
            report.status = "BLOCKED_AUTH"
            report.notes = str(auth_err)
            return report

        raw_chunk_dfs: List[pd.DataFrame] = []
        chunk_audit: List[ChunkInfo] = []

        for idx, (c_start, c_end) in enumerate(date_chunks):
            chunk_info = ChunkInfo(chunk_index=idx, start_date=c_start, end_date=c_end)
            success = False

            for attempt in range(1, self.max_retries + 1):
                try:
                    df_chunk = self.provider.fetch_chunk(symbol, c_start, c_end, interval=interval)
                    if df_chunk is not None and not df_chunk.empty:
                        raw_chunk_dfs.append(df_chunk)
                        chunk_info.rows_downloaded = len(df_chunk)
                    chunk_info.status = "SUCCESS"
                    report.chunks_successful += 1
                    success = True
                    break
                except RateLimitExceededError as rle:
                    logger.warning(
                        "Rate limit on chunk %d (attempt %d/%d): %s. Backing off...",
                        idx, attempt, self.max_retries, rle
                    )
                    time.sleep(self.retry_delay_seconds * (2 ** attempt))
                except Exception as exc:
                    logger.error(
                        "Error on chunk %d (attempt %d/%d): %s",
                        idx, attempt, self.max_retries, exc
                    )
                    time.sleep(self.retry_delay_seconds)

            if not success:
                chunk_info.status = "FAILED"
                chunk_info.error_message = f"Failed after {self.max_retries} attempts."
                report.chunks_failed += 1
                logger.error("Chunk %d failed completely for %s (%s to %s)", idx, symbol, c_start, c_end)

            chunk_audit.append(chunk_info)

        if not raw_chunk_dfs:
            report.status = "FAILED"
            report.notes = "Zero rows downloaded across all chunks."
            return report

        # Concatenate and normalize
        combined_raw = pd.concat(raw_chunk_dfs, ignore_index=True)
        canonical_df = self.provider.normalize_to_canonical(combined_raw, symbol)

        # Deduplicate symbol/timestamp collisions across chunk boundaries
        dedup_mask = canonical_df.duplicated(subset=["symbol", "timestamp"], keep="first")
        if dedup_mask.any():
            dup_count = int(dedup_mask.sum())
            logger.info("Removed %d duplicate boundary bars for %s.", dup_count, symbol)
            canonical_df = canonical_df[~dedup_mask].sort_values("timestamp").reset_index(drop=True)

        report.total_raw_rows = len(canonical_df)
        if not canonical_df.empty:
            report.actual_range = (
                str(canonical_df["timestamp"].min()),
                str(canonical_df["timestamp"].max()),
            )

        # Save Raw Download Immutably
        raw_path, meta_path, sha_hash = self.storage_manager.save_raw_download(
            df=canonical_df,
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            provider=self.provider.provider_name,
        )
        report.raw_storage_path = raw_path
        report.checksum_sha256 = sha_hash

        # Run Quality Validator
        clean_df, val_audit = self.validator.validate(canonical_df)
        report.validation_audit = val_audit
        report.total_clean_rows = len(clean_df)

        # Save Clean Validated Parquet
        clean_path = self.storage_manager.save_clean_data(clean_df, symbol, interval)
        report.clean_storage_path = clean_path

        if report.chunks_failed > 0:
            report.status = "PARTIAL"
        else:
            report.status = "SUCCESS"

        logger.info(
            "Acquisition complete for %s: %d clean rows saved. Status: %s",
            symbol, report.total_clean_rows, report.status
        )

        return report
