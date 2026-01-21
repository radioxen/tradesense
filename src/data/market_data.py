"""Market data ingestion providers."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from src.utils.logging import get_logger


logger = get_logger(__name__)


class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1h",
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'AAPL').
            start: Start datetime.
            end: End datetime.
            interval: Bar interval ('1m', '5m', '15m', '1h', '1d').

        Returns:
            DataFrame with columns: open, high, low, close, volume, adjusted_close.
        """
        ...

    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            Current price.
        """
        ...

    @abstractmethod
    async def stream_quotes(
        self,
        symbols: list[str],
        callback: Any,
    ) -> None:
        """Stream real-time quotes for symbols.

        Args:
            symbols: List of symbols to stream.
            callback: Callback function for quote updates.
        """
        ...


class YFinanceProvider(MarketDataProvider):
    """Yahoo Finance data provider for backtesting.

    Free provider suitable for historical data and development.
    """

    INTERVAL_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "1d": "1d",
        "1wk": "1wk",
        "1mo": "1mo",
    }

    def __init__(self, cache_dir: Path | None = None):
        """Initialize YFinance provider.

        Args:
            cache_dir: Optional directory for caching data.
        """
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1h",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from Yahoo Finance.

        Args:
            symbol: Trading symbol.
            start: Start datetime.
            end: End datetime.
            interval: Bar interval.

        Returns:
            DataFrame with OHLCV data.
        """
        logger.info(f"Fetching {symbol} data from {start} to {end} ({interval})")

        # Check cache first
        if self.cache_dir:
            cache_file = self._get_cache_path(symbol, start, end, interval)
            if cache_file.exists():
                logger.info(f"Loading {symbol} from cache")
                return pd.read_parquet(cache_file)

        # Map interval
        yf_interval = self.INTERVAL_MAP.get(interval, "1h")

        # Fetch from Yahoo Finance
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start,
            end=end,
            interval=yf_interval,
            auto_adjust=False,
        )

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Standardize column names
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adjusted_close",
        })

        # Keep only needed columns
        df = df[["open", "high", "low", "close", "volume", "adjusted_close"]]

        # Add symbol column
        df["symbol"] = symbol

        # Reset index to make timestamp a column
        df = df.reset_index()
        df = df.rename(columns={"Date": "timestamp", "Datetime": "timestamp"})

        # Validate data
        df = self._validate_data(df)

        # Cache the data
        if self.cache_dir:
            cache_file = self._get_cache_path(symbol, start, end, interval)
            df.to_parquet(cache_file, index=False)
            logger.info(f"Cached {symbol} data to {cache_file}")

        logger.info(f"Fetched {len(df)} bars for {symbol}")
        return df

    async def get_current_price(self, symbol: str) -> float:
        """Get current price from Yahoo Finance.

        Args:
            symbol: Trading symbol.

        Returns:
            Current price.
        """
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        if data.empty:
            raise ValueError(f"Could not get current price for {symbol}")
        return float(data["Close"].iloc[-1])

    async def stream_quotes(
        self,
        symbols: list[str],
        callback: Any,
    ) -> None:
        """Stream quotes (not supported by YFinance).

        Args:
            symbols: List of symbols.
            callback: Quote callback.

        Raises:
            NotImplementedError: YFinance doesn't support streaming.
        """
        raise NotImplementedError("YFinance does not support real-time streaming")

    def _get_cache_path(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> Path:
        """Get cache file path for data.

        Args:
            symbol: Trading symbol.
            start: Start datetime.
            end: End datetime.
            interval: Interval.

        Returns:
            Path to cache file.
        """
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        filename = f"{symbol}_{start_str}_{end_str}_{interval}.parquet"
        return self.cache_dir / filename

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLCV data.

        Args:
            df: Raw dataframe.

        Returns:
            Validated dataframe.
        """
        initial_len = len(df)

        # Remove duplicates
        df = df.drop_duplicates(subset=["timestamp"])

        # Sort by timestamp
        df = df.sort_values("timestamp")

        # Check for missing values
        null_count = df[["open", "high", "low", "close", "volume"]].isnull().sum().sum()
        if null_count > 0:
            logger.warning(f"Found {null_count} null values, forward filling")
            df = df.ffill()

        # Validate OHLC relationships
        invalid_ohlc = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        )
        if invalid_ohlc.any():
            logger.warning(f"Found {invalid_ohlc.sum()} invalid OHLC rows, removing")
            df = df[~invalid_ohlc]

        # Validate volume
        df = df[df["volume"] >= 0]

        final_len = len(df)
        if final_len < initial_len:
            logger.info(f"Cleaned data: {initial_len} -> {final_len} rows")

        return df.reset_index(drop=True)


class AlpacaProvider(MarketDataProvider):
    """Alpaca data provider for live trading.

    Supports both historical and real-time data.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://paper-api.alpaca.markets",
    ):
        """Initialize Alpaca provider.

        Args:
            api_key: Alpaca API key.
            api_secret: Alpaca API secret.
            base_url: API base URL (paper or live).
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self._client = None

    async def _get_client(self):
        """Get or create Alpaca client."""
        if self._client is None:
            try:
                from alpaca.data import StockHistoricalDataClient
                self._client = StockHistoricalDataClient(
                    api_key=self.api_key,
                    secret_key=self.api_secret,
                )
            except ImportError:
                raise ImportError("alpaca-py is required for Alpaca provider")
        return self._client

    async def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1h",
    ) -> pd.DataFrame:
        """Fetch historical data from Alpaca.

        Args:
            symbol: Trading symbol.
            start: Start datetime.
            end: End datetime.
            interval: Bar interval.

        Returns:
            DataFrame with OHLCV data.
        """
        from alpaca.data import TimeFrame
        from alpaca.data.requests import StockBarsRequest

        client = await self._get_client()

        # Map interval to Alpaca TimeFrame
        timeframe_map = {
            "1m": TimeFrame.Minute,
            "5m": TimeFrame.Minute,  # Need to handle aggregation
            "15m": TimeFrame.Minute,
            "1h": TimeFrame.Hour,
            "1d": TimeFrame.Day,
        }

        timeframe = timeframe_map.get(interval, TimeFrame.Hour)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )

        bars = client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            return pd.DataFrame()

        # Standardize columns
        df = df.reset_index()
        df = df.rename(columns={
            "symbol": "symbol",
            "timestamp": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        })

        # Add adjusted close (Alpaca returns adjusted by default)
        df["adjusted_close"] = df["close"]

        return df

    async def get_current_price(self, symbol: str) -> float:
        """Get current price from Alpaca.

        Args:
            symbol: Trading symbol.

        Returns:
            Current price.
        """
        from alpaca.data.requests import StockLatestQuoteRequest

        client = await self._get_client()
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = client.get_stock_latest_quote(request)
        return float(quote[symbol].ask_price)

    async def stream_quotes(
        self,
        symbols: list[str],
        callback: Any,
    ) -> None:
        """Stream real-time quotes from Alpaca.

        Args:
            symbols: List of symbols to stream.
            callback: Callback for quote updates.
        """
        from alpaca.data.live import StockDataStream

        stream = StockDataStream(self.api_key, self.api_secret)

        @stream.on_bar(symbols)
        async def on_bar(data):
            await callback(data)

        await stream.run()


def get_provider(
    provider_name: str,
    config: dict[str, Any] | None = None,
) -> MarketDataProvider:
    """Factory function to get market data provider.

    Args:
        provider_name: Provider name ('yfinance', 'alpaca').
        config: Provider configuration.

    Returns:
        Market data provider instance.
    """
    config = config or {}

    if provider_name == "yfinance":
        cache_dir = Path(config.get("cache_dir", "./data/market"))
        return YFinanceProvider(cache_dir=cache_dir)

    elif provider_name == "alpaca":
        return AlpacaProvider(
            api_key=config["api_key"],
            api_secret=config["api_secret"],
            base_url=config.get("base_url", "https://paper-api.alpaca.markets"),
        )

    else:
        raise ValueError(f"Unknown provider: {provider_name}")
