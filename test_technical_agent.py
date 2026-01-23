#!/usr/bin/env python3
"""Test the enhanced Technical Analyst agent on stocks."""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import yfinance as yf

from src.agents.technical import TechnicalAnalystAgent
from src.orchestrator.contracts import PipelineState


def fetch_extended_history(symbol: str, days: int = 300) -> pd.DataFrame:
    """Fetch extended history by batching requests.
    
    yfinance can return up to ~2 years of daily data, but we batch
    to ensure reliability.
    
    Args:
        symbol: Stock ticker
        days: Number of days to fetch
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Fetching {days} days of data for {symbol}...")
    
    ticker = yf.Ticker(symbol)
    
    # Try fetching all at once first (yfinance supports this)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 10)  # Extra buffer for weekends
    
    df = ticker.history(start=start_date, end=end_date)
    
    if len(df) >= days * 0.7:  # 70% of requested (accounting for weekends/holidays)
        print(f"  Got {len(df)} trading days in single request")
        return df
    
    # If single request didn't work, batch by 60-day intervals
    print(f"  Batching requests for more data...")
    all_data = []
    
    for i in range(0, days, 60):
        batch_end = end_date - timedelta(days=i)
        batch_start = batch_end - timedelta(days=65)  # 65 to ensure overlap
        
        batch_df = ticker.history(start=batch_start, end=batch_end)
        if not batch_df.empty:
            all_data.append(batch_df)
            print(f"  Batch {i//60 + 1}: {len(batch_df)} days")
    
    if all_data:
        combined = pd.concat(all_data)
        combined = combined[~combined.index.duplicated(keep='first')]
        combined = combined.sort_index()
        print(f"  Total: {len(combined)} trading days")
        return combined
    
    return df


def calculate_features(df: pd.DataFrame) -> dict:
    """Calculate technical features from OHLCV data."""
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    features = {}
    
    # Returns
    features['return_1'] = close.pct_change(1).iloc[-1]
    features['return_5'] = close.pct_change(5).iloc[-1]
    features['return_20'] = close.pct_change(20).iloc[-1]
    
    # RSI (14-period)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    features['rsi'] = rsi.iloc[-1]
    features['rsi_zscore'] = (rsi.iloc[-1] - rsi.mean()) / rsi.std()
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    features['macd'] = macd.iloc[-1]
    features['macd_signal'] = macd_signal.iloc[-1]
    features['macd_hist'] = macd_hist.iloc[-1]
    
    # ROC
    features['roc_5'] = ((close.iloc[-1] / close.iloc[-5]) - 1) * 100
    features['roc_10'] = ((close.iloc[-1] / close.iloc[-10]) - 1) * 100
    features['roc_20'] = ((close.iloc[-1] / close.iloc[-20]) - 1) * 100
    
    # Stochastic
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    stoch_k = ((close - low_14) / (high_14 - low_14)) * 100
    stoch_d = stoch_k.rolling(3).mean()
    features['stoch_k'] = stoch_k.iloc[-1]
    features['stoch_d'] = stoch_d.iloc[-1]
    
    # ADX
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(14).mean()
    
    features['adx'] = adx.iloc[-1]
    features['plus_di'] = plus_di.iloc[-1]
    features['minus_di'] = minus_di.iloc[-1]
    
    # Volatility
    returns = close.pct_change()
    features['volatility_5'] = returns.rolling(5).std().iloc[-1] * np.sqrt(252)
    features['volatility_20'] = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
    features['volatility_ratio'] = features['volatility_5'] / features['volatility_20']
    
    # ATR %
    features['atr_percent'] = (atr.iloc[-1] / close.iloc[-1]) * 100
    
    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    features['bb_position'] = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
    features['bb_width'] = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / sma20.iloc[-1]
    
    # Volume
    vol_sma = volume.rolling(20).mean()
    features['volume_ratio'] = volume.iloc[-1] / vol_sma.iloc[-1]
    features['volume_zscore'] = (volume.iloc[-1] - vol_sma.iloc[-1]) / volume.rolling(20).std().iloc[-1]
    
    # Price vs SMAs
    sma50 = close.rolling(50).mean()
    features['price_sma20_ratio'] = close.iloc[-1] / sma20.iloc[-1]
    features['price_sma50_ratio'] = close.iloc[-1] / sma50.iloc[-1]
    
    # SMA slopes
    features['sma20_slope'] = (sma20.iloc[-1] - sma20.iloc[-5]) / sma20.iloc[-5]
    features['sma50_slope'] = (sma50.iloc[-1] - sma50.iloc[-5]) / sma50.iloc[-5]
    
    return features


async def test_technical_agent(symbol: str = "TSLA", days: int = 300, epochs: int = 500):
    """Test the Technical Analyst agent on a stock.
    
    Args:
        symbol: Stock ticker
        days: Days of history to fetch (default 300)
        epochs: Training epochs for Evolution Strategy (default 500)
    """
    print(f"\n{'='*60}")
    print(f"  Testing Technical Analyst on {symbol}")
    print(f"  Data: {days} days | Evolution Strategy Epochs: {epochs}")
    print(f"{'='*60}\n")
    
    # Fetch extended history
    df = fetch_extended_history(symbol, days=days)
    
    if df.empty:
        print(f"Error: No data for {symbol}")
        return
    
    print(f"Latest close: ${df['Close'].iloc[-1]:.2f}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    # Calculate features
    print("\nCalculating technical features...")
    features = calculate_features(df)
    
    # Print key features
    print("\n--- Key Indicators ---")
    print(f"  RSI (14):        {features['rsi']:.1f}")
    print(f"  MACD:            {features['macd']:.3f}")
    print(f"  MACD Histogram:  {features['macd_hist']:.3f}")
    print(f"  Stochastic K:    {features['stoch_k']:.1f}")
    print(f"  ADX:             {features['adx']:.1f}")
    print(f"  BB Position:     {features['bb_position']:.2f}")
    print(f"  Volatility (5d): {features['volatility_5']*100:.1f}%")
    print(f"  Volume Ratio:    {features['volume_ratio']:.2f}x")
    
    # Create pipeline state
    state = PipelineState(
        symbol=symbol,
        current_price=df['Close'].iloc[-1],
        features=features,
        timestamp=datetime.now(),
        interval="1d",
        cash=10000.0,
        equity=10000.0,
    )
    
    # Create and run agent
    # Evolution Strategy is disabled by default (struggles in volatile markets)
    # Enable with --evolution flag for experimentation
    use_evo = "--evolution" in sys.argv or "-e" in sys.argv
    
    print(f"\n--- Running Technical Analyst Agent ---")
    print(f"Mode: {'Evolution Strategy + Rule-based' if use_evo else 'Rule-based Only'}")
    if use_evo:
        print(f"Evolution Strategy: window=30, epochs={epochs}")
    
    agent = TechnicalAnalystAgent(
        use_evolution_strategy=use_evo,
        evolution_window=30,
        evolution_epochs=epochs,
    )
    
    # Get price array for Evolution Strategy
    prices = df['Close'].values
    
    # Run analysis
    signal = await agent.analyze(state, prices=prices)
    
    # Print results
    print("\n" + "="*60)
    print("  TECHNICAL ANALYST RESULTS")
    print("="*60)
    print(f"\n  Direction:   {signal.direction.value.upper()}")
    print(f"  Confidence:  {signal.confidence:.1%}")
    print(f"  Regime:      {signal.regime.value}")
    
    print(f"\n  Forecast (next period):")
    print(f"    Q10 (downside): {signal.forecast_q10:+.2f}%")
    print(f"    Q50 (median):   {signal.forecast_q50:+.2f}%")
    print(f"    Q90 (upside):   {signal.forecast_q90:+.2f}%")
    
    print(f"\n  Probabilities:")
    print(f"    P(Up > {signal.threshold_pct}%):   {signal.p_up:.1%}")
    print(f"    P(Down > {signal.threshold_pct}%): {signal.p_down:.1%}")
    
    print(f"\n  Edge after costs: {signal.edge_after_costs:+.2f}%")
    
    if signal.hit_rate:
        print(f"  Backtest hit rate: {signal.hit_rate:.1%}")
    
    print(f"\n  Top Features:")
    for feat in signal.top_features[:5]:
        print(f"    - {feat.feature_name}: {feat.current_value:.2f} ({feat.interpretation})")
    
    # Trading recommendation
    print("\n" + "-"*60)
    if signal.confidence > 0.7:
        strength = "STRONG"
    elif signal.confidence > 0.55:
        strength = "MODERATE"
    else:
        strength = "WEAK"
    
    if signal.direction.value == "long":
        action = "BUY"
        color = "\033[92m"  # Green
    elif signal.direction.value == "short":
        action = "SELL"
        color = "\033[91m"  # Red
    else:
        action = "HOLD"
        color = "\033[93m"  # Yellow
    
    reset = "\033[0m"
    print(f"\n  {color}>>> {strength} {action} SIGNAL <<<{reset}")
    print(f"  Confidence: {signal.confidence:.1%}")
    print("-"*60 + "\n")
    
    return signal


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Technical Analyst Agent")
    parser.add_argument("symbol", nargs="?", default="TSLA", help="Stock symbol")
    parser.add_argument("--days", type=int, default=300, help="Days of history (default: 300)")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs (default: 500)")
    
    args = parser.parse_args()
    
    asyncio.run(test_technical_agent(args.symbol, days=args.days, epochs=args.epochs))
