import yfinance as yf
import pandas as pd
import asyncio
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Per-ticker hard timeout — if yfinance doesn't respond within this many seconds,
# the fetch is cancelled, the failure is logged, and that ticker is skipped.
TICKER_FETCH_TIMEOUT_SECONDS = 8


class StockDataFetcher:
    """Async stock data fetcher using real yfinance data.

    Each ticker fetch runs in a threadpool with a hard per-ticker timeout.
    On timeout or yfinance error: logs the failure loudly and skips that ticker.
    The calling pipeline receives only successfully fetched tickers.
    """

    @staticmethod
    def _fetch_single_ticker_sync(symbol: str) -> pd.DataFrame:
        """Fetch 1-month daily OHLCV data for a single symbol via yfinance (blocking)."""
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1mo", interval="1d")
        if df is None or df.empty or len(df) < 5:
            raise ValueError(
                f"yfinance returned insufficient data for {symbol} "
                f"(rows={len(df) if df is not None else 0})"
            )


        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        if "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"])
            df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
        elif "datetime" in df.columns:
            df["timestamp"] = pd.to_datetime(df["datetime"])
            df["date_str"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M")
        else:
            df["timestamp"] = pd.to_datetime(df.index)
            df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        return df

    async def _fetch_with_timeout(self, symbol: str, loop: asyncio.AbstractEventLoop) -> tuple[str, pd.DataFrame | None]:
        """Fetches a single ticker with a hard timeout.

        Returns (symbol, DataFrame) on success, (symbol, None) on timeout/failure.
        Never raises — failures are logged and the ticker is skipped.
        """
        try:
            df = await asyncio.wait_for(
                loop.run_in_executor(None, self._fetch_single_ticker_sync, symbol),
                timeout=TICKER_FETCH_TIMEOUT_SECONDS,
            )
            logger.info("Fetched %s successfully (%d rows).", symbol, len(df))
            return symbol, df
        except asyncio.TimeoutError:
            logger.warning(
                "TIMEOUT: %s did not respond within %ds — skipping this ticker.",
                symbol, TICKER_FETCH_TIMEOUT_SECONDS,
            )
            return symbol, None
        except Exception as exc:
            logger.warning("FETCH ERROR: %s — %s — skipping this ticker.", symbol, exc)
            return symbol, None

    async def fetch_stock_data_async(self, symbols: list[str]) -> Dict[str, pd.DataFrame]:
        """Fetches all symbols concurrently.  Tickers that timeout or fail are skipped.

        Raises RuntimeError if *no* tickers succeed (nothing to work with).
        """
        loop = asyncio.get_event_loop()
        tasks = [self._fetch_with_timeout(symbol, loop) for symbol in symbols]
        results_list = await asyncio.gather(*tasks)

        results: Dict[str, pd.DataFrame] = {}
        failed: list[str] = []

        for symbol, df in results_list:
            if df is not None:
                results[symbol] = df
            else:
                failed.append(symbol)

        if failed:
            logger.warning("Skipped %d ticker(s) due to fetch failures: %s", len(failed), failed)

        if not results:
            raise RuntimeError(
                f"All {len(symbols)} ticker fetches failed or timed out: {symbols}. "
                "Check your internet connection or NSE availability."
            )

        return results


data_fetcher = StockDataFetcher()
