"""Feature engineering pipeline for technical indicators."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # Lookback periods
    short_period: int = 5
    medium_period: int = 20
    long_period: int = 60

    # RSI
    rsi_period: int = 14

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # ATR
    atr_period: int = 14

    # Volume
    volume_ma_period: int = 20


class FeatureBuilder:
    """Builds technical features from OHLCV data.

    Computes a comprehensive set of technical indicators used by the
    Technical Analyst Agent and ML models.
    """

    def __init__(self, config: FeatureConfig | None = None):
        """Initialize feature builder.

        Args:
            config: Feature configuration.
        """
        self.config = config or FeatureConfig()

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all features from OHLCV data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume.

        Returns:
            DataFrame with original data plus feature columns.
        """
        logger.info(f"Building features for {len(df)} rows")
        start_time = datetime.now()

        # Make a copy
        df = df.copy()

        # Ensure required columns exist
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Returns
        df = self._add_returns(df)

        # Momentum indicators
        df = self._add_momentum(df)

        # Trend indicators
        df = self._add_trend(df)

        # Volatility indicators
        df = self._add_volatility(df)

        # Volume indicators
        df = self._add_volume(df)

        # Regime labels
        df = self._add_regime(df)

        elapsed = (datetime.now() - start_time).total_seconds()
        feature_cols = [c for c in df.columns if c not in required + ["timestamp", "symbol", "adjusted_close"]]
        logger.info(f"Built {len(feature_cols)} features in {elapsed:.2f}s")

        return df

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return features."""
        close = df["close"]

        # Simple returns
        df["return_1"] = close.pct_change(1)
        df["return_5"] = close.pct_change(5)
        df["return_10"] = close.pct_change(10)
        df["return_20"] = close.pct_change(20)

        # Log returns
        df["log_return_1"] = np.log(close / close.shift(1))
        df["log_return_5"] = np.log(close / close.shift(5))

        # Cumulative returns
        df["cum_return_5"] = close.pct_change(5).rolling(5).sum()
        df["cum_return_20"] = close.pct_change(20).rolling(20).sum()

        return df

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        close = df["close"]
        cfg = self.config

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(cfg.rsi_period).mean()
        avg_loss = loss.rolling(cfg.rsi_period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

        # RSI z-score
        df["rsi_zscore"] = (df["rsi"] - df["rsi"].rolling(50).mean()) / (df["rsi"].rolling(50).std() + 1e-10)

        # MACD
        ema_fast = close.ewm(span=cfg.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=cfg.macd_slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=cfg.macd_signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # MACD crossover
        df["macd_cross"] = np.sign(df["macd_hist"]) - np.sign(df["macd_hist"].shift(1))

        # Rate of Change (ROC)
        df["roc_5"] = (close - close.shift(5)) / (close.shift(5) + 1e-10) * 100
        df["roc_10"] = (close - close.shift(10)) / (close.shift(10) + 1e-10) * 100
        df["roc_20"] = (close - close.shift(20)) / (close.shift(20) + 1e-10) * 100

        # Stochastic oscillator
        low_14 = df["low"].rolling(14).min()
        high_14 = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        return df

    def _add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        cfg = self.config

        # Moving averages
        df["sma_5"] = close.rolling(cfg.short_period).mean()
        df["sma_20"] = close.rolling(cfg.medium_period).mean()
        df["sma_50"] = close.rolling(50).mean()
        df["sma_200"] = close.rolling(200).mean()

        # EMA
        df["ema_12"] = close.ewm(span=12, adjust=False).mean()
        df["ema_26"] = close.ewm(span=26, adjust=False).mean()

        # Price relative to MAs
        df["price_sma20_ratio"] = close / (df["sma_20"] + 1e-10)
        df["price_sma50_ratio"] = close / (df["sma_50"] + 1e-10)

        # MA slopes
        df["sma20_slope"] = df["sma_20"].diff(5) / 5
        df["sma50_slope"] = df["sma_50"].diff(10) / 10

        # ADX (Average Directional Index)
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - close.shift(1)),
                np.abs(low - close.shift(1))
            )
        )
        plus_dm = np.where(
            (high - high.shift(1)) > (low.shift(1) - low),
            np.maximum(high - high.shift(1), 0),
            0
        )
        minus_dm = np.where(
            (low.shift(1) - low) > (high - high.shift(1)),
            np.maximum(low.shift(1) - low, 0),
            0
        )

        atr_14 = pd.Series(tr).rolling(14).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / (atr_14 + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / (atr_14 + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df["adx"] = pd.Series(dx).rolling(14).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di

        # Trend strength
        df["trend_strength"] = np.where(
            df["adx"] > 25,
            np.where(df["plus_di"] > df["minus_di"], 1, -1),
            0
        )

        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        cfg = self.config

        # Historical volatility
        df["volatility_5"] = df["log_return_1"].rolling(cfg.short_period).std() * np.sqrt(252)
        df["volatility_20"] = df["log_return_1"].rolling(cfg.medium_period).std() * np.sqrt(252)
        df["volatility_60"] = df["log_return_1"].rolling(cfg.long_period).std() * np.sqrt(252)

        # Volatility ratio
        df["volatility_ratio"] = df["volatility_5"] / (df["volatility_20"] + 1e-10)

        # ATR (Average True Range)
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - close.shift(1)),
                np.abs(low - close.shift(1))
            )
        )
        df["atr"] = pd.Series(tr).rolling(cfg.atr_period).mean()
        df["atr_percent"] = df["atr"] / (close + 1e-10) * 100

        # Bollinger Bands
        df["bb_middle"] = close.rolling(cfg.bb_period).mean()
        bb_std = close.rolling(cfg.bb_period).std()
        df["bb_upper"] = df["bb_middle"] + cfg.bb_std * bb_std
        df["bb_lower"] = df["bb_middle"] - cfg.bb_std * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_middle"] + 1e-10)
        df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

        # Keltner Channel
        ema_20 = close.ewm(span=20, adjust=False).mean()
        df["kc_upper"] = ema_20 + 2 * df["atr"]
        df["kc_lower"] = ema_20 - 2 * df["atr"]

        # Squeeze (BB inside KC)
        df["squeeze"] = (
            (df["bb_lower"] > df["kc_lower"]) &
            (df["bb_upper"] < df["kc_upper"])
        ).astype(int)

        return df

    def _add_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators."""
        close = df["close"]
        volume = df["volume"]
        high = df["high"]
        low = df["low"]
        cfg = self.config

        # Volume moving average
        df["volume_sma"] = volume.rolling(cfg.volume_ma_period).mean()
        df["volume_ratio"] = volume / (df["volume_sma"] + 1e-10)

        # Volume z-score
        df["volume_zscore"] = (volume - df["volume_sma"]) / (volume.rolling(cfg.volume_ma_period).std() + 1e-10)

        # On-Balance Volume (OBV)
        obv = np.where(close > close.shift(1), volume, -volume)
        obv = np.where(close == close.shift(1), 0, obv)
        df["obv"] = pd.Series(obv).cumsum()
        df["obv_sma"] = df["obv"].rolling(20).mean()

        # Money Flow Index
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        positive_mf = positive_flow.rolling(14).sum()
        negative_mf = negative_flow.rolling(14).sum()
        mfi_ratio = positive_mf / (negative_mf + 1e-10)
        df["mfi"] = 100 - (100 / (1 + mfi_ratio))

        # VWAP (approximation using cumulative)
        df["vwap"] = (volume * typical_price).cumsum() / (volume.cumsum() + 1e-10)
        df["price_vwap_ratio"] = close / (df["vwap"] + 1e-10)

        return df

    def _add_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime classification labels."""
        # Volatility regime
        vol_percentile = df["volatility_20"].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        df["vol_regime"] = pd.cut(
            vol_percentile,
            bins=[0, 0.25, 0.75, 1.0],
            labels=["low_vol", "normal_vol", "high_vol"]
        )

        # Trend regime
        df["trend_regime"] = np.where(
            df["adx"] > 25,
            np.where(df["plus_di"] > df["minus_di"], "uptrend", "downtrend"),
            "ranging"
        )

        # Momentum regime
        df["momentum_regime"] = pd.cut(
            df["rsi"],
            bins=[0, 30, 70, 100],
            labels=["oversold", "neutral", "overbought"]
        )

        return df


class FeatureStore:
    """Store and retrieve computed features.

    Handles caching and versioning of feature data.
    """

    def __init__(self, storage_path: Path):
        """Initialize feature store.

        Args:
            storage_path: Path to store feature files.
        """
        self.storage_path = storage_path
        storage_path.mkdir(parents=True, exist_ok=True)
        self.builder = FeatureBuilder()

    async def get_features(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        ohlcv_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Get features for a symbol and time range.

        Args:
            symbol: Trading symbol.
            start: Start datetime.
            end: End datetime.
            ohlcv_df: Optional pre-loaded OHLCV data.

        Returns:
            DataFrame with features.
        """
        cache_path = self._get_cache_path(symbol, start, end)

        # Check cache
        if cache_path.exists() and ohlcv_df is None:
            logger.info(f"Loading features from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        # Build features from OHLCV
        if ohlcv_df is None:
            raise ValueError("OHLCV data required when not cached")

        features_df = self.builder.build_features(ohlcv_df)

        # Cache features
        features_df.to_parquet(cache_path, index=False)
        logger.info(f"Cached features to: {cache_path}")

        return features_df

    def get_feature_summary(
        self,
        df: pd.DataFrame,
        last_n_bars: int = 10,
    ) -> dict[str, Any]:
        """Get a compact feature summary for LLM prompts.

        Args:
            df: Features dataframe.
            last_n_bars: Number of recent bars to summarize.

        Returns:
            Dictionary with feature summary.
        """
        recent = df.tail(last_n_bars)
        latest = df.iloc[-1]

        summary = {
            "current_price": float(latest["close"]),
            "returns": {
                "1_bar": float(latest.get("return_1", 0)),
                "5_bar": float(latest.get("return_5", 0)),
                "20_bar": float(latest.get("return_20", 0)),
            },
            "momentum": {
                "rsi": float(latest.get("rsi", 50)),
                "macd_hist": float(latest.get("macd_hist", 0)),
                "stoch_k": float(latest.get("stoch_k", 50)),
            },
            "trend": {
                "adx": float(latest.get("adx", 0)),
                "trend_strength": int(latest.get("trend_strength", 0)),
                "price_vs_sma20": float(latest.get("price_sma20_ratio", 1) - 1) * 100,
            },
            "volatility": {
                "atr_percent": float(latest.get("atr_percent", 0)),
                "bb_position": float(latest.get("bb_position", 0.5)),
                "vol_20d_annualized": float(latest.get("volatility_20", 0)),
            },
            "volume": {
                "volume_ratio": float(latest.get("volume_ratio", 1)),
                "volume_zscore": float(latest.get("volume_zscore", 0)),
            },
            "regime": {
                "volatility": str(latest.get("vol_regime", "normal")),
                "trend": str(latest.get("trend_regime", "ranging")),
                "momentum": str(latest.get("momentum_regime", "neutral")),
            },
        }

        return summary

    def _get_cache_path(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> Path:
        """Get cache file path."""
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        return self.storage_path / f"features_{symbol}_{start_str}_{end_str}.parquet"
