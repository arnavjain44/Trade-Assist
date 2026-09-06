"""
Immutable Raw Storage Manager for Historical Market Data (Phase 5.4)
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class RawStorageManager:
    """
    Manages raw, immutable historical dataset storage.
    Enforces that raw provider downloads are never mutated or silently overwritten.
    Maintains cryptographic checksums and provenance metadata.
    """

    def __init__(
        self,
        raw_dir: str = "data/raw/historical",
        clean_dir: str = "data/processed/historical",
    ):
        self.raw_dir = os.path.abspath(raw_dir)
        self.clean_dir = os.path.abspath(clean_dir)
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.clean_dir, exist_ok=True)

    @staticmethod
    def compute_dataframe_hash(df: pd.DataFrame) -> str:
        """Computes a deterministic SHA-256 hash from a DataFrame's parquet byte representation."""
        parquet_bytes = df.to_parquet(index=False)
        return hashlib.sha256(parquet_bytes).hexdigest()

    def save_raw_download(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        provider: str,
        adjustment_status: str = "split_and_bonus_adjusted",
        allow_overwrite: bool = False,
    ) -> Tuple[str, str, str]:
        """
        Saves raw downloaded data as immutable Parquet with companion JSON metadata.
        Returns:
            Tuple[parquet_path, metadata_path, sha256_hash]
        """
        if df.empty:
            raise ValueError(f"Cannot save empty raw DataFrame for {symbol}.")

        clean_sym = symbol.replace(".NS", "").replace(".BO", "").replace("&", "_")
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        base_name = f"{clean_sym}_{interval}_{start_str}_{end_str}_{provider}_raw"

        parquet_path = os.path.join(self.raw_dir, f"{base_name}.parquet")
        meta_path = os.path.join(self.raw_dir, f"{base_name}.meta.json")

        if os.path.exists(parquet_path) and not allow_overwrite:
            logger.warning(
                "Raw file %s already exists. Immutability enforced: not overwriting.",
                parquet_path,
            )
            # Read existing hash from metadata
            with open(meta_path, "r", encoding="utf-8") as f:
                existing_meta = json.load(f)
            return parquet_path, meta_path, existing_meta.get("sha256_checksum", "")

        # Compute deterministic checksum
        sha256_hash = self.compute_dataframe_hash(df)

        # Write Parquet
        df.to_parquet(parquet_path, index=False)

        # Write Metadata JSON
        actual_min_ts = str(df["timestamp"].min()) if "timestamp" in df.columns else "N/A"
        actual_max_ts = str(df["timestamp"].max()) if "timestamp" in df.columns else "N/A"

        metadata: Dict[str, Any] = {
            "symbol": symbol,
            "provider": provider,
            "interval": interval,
            "requested_start": start_date.isoformat(),
            "requested_end": end_date.isoformat(),
            "actual_earliest_timestamp": actual_min_ts,
            "actual_latest_timestamp": actual_max_ts,
            "row_count": len(df),
            "columns": list(df.columns),
            "adjustment_status": adjustment_status,
            "download_timestamp": datetime.now().isoformat(),
            "sha256_checksum": sha256_hash,
            "immutable": True,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "Raw historical data saved immutably: %s (%d rows, hash=%s)",
            parquet_path,
            len(df),
            sha256_hash[:12],
        )

        return parquet_path, meta_path, sha256_hash

    def load_raw_download(self, parquet_path: str, verify_checksum: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads raw Parquet file and verifies cryptographic integrity against its metadata sidecar.
        """
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Raw data file not found: {parquet_path}")

        meta_path = parquet_path.replace(".parquet", ".meta.json")
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        df = pd.read_parquet(parquet_path)

        if verify_checksum and "sha256_checksum" in metadata:
            current_hash = self.compute_dataframe_hash(df)
            expected_hash = metadata["sha256_checksum"]
            if current_hash != expected_hash:
                raise ValueError(
                    f"Integrity check failed for {parquet_path}! "
                    f"Expected hash {expected_hash}, computed {current_hash}."
                )

        return df, metadata

    def save_clean_data(self, df: pd.DataFrame, symbol: str, interval: str) -> str:
        """Saves clean validated canonical DataFrame into processed historical directory."""
        clean_sym = symbol.replace(".NS", "").replace(".BO", "").replace("&", "_")
        clean_path = os.path.join(self.clean_dir, f"{clean_sym}_{interval}_clean.parquet")
        df.to_parquet(clean_path, index=False)
        logger.info("Saved clean validated dataset to %s (%d rows)", clean_path, len(df))
        return clean_path
