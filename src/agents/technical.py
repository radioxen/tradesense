"""Technical Analyst Agent with ML ensemble models.

Uses multiple model families for robust predictions:
- XGBoost/LightGBM (tree-based)
- Evolution Strategy Neural Network (from tradioxen)
- LSTM (deep learning for sequences)
- Linear baseline (calibration reference)

Outputs probabilistic forecasts with confidence intervals.

Enhanced with techniques from github.com/radioxen/tradioxen
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple, List
import uuid

import numpy as np
import pandas as pd

from src.agents.base import ModelAgent
from src.orchestrator.contracts import (
    Direction,
    FeatureSummary,
    PipelineState,
    Regime,
    TechnicalSignal,
)
from src.utils.logging import get_logger

# Import Evolution Strategy components
try:
    from src.agents.evolution_strategy import EvolutionStrategyAgent, TradeResult
    HAS_EVOLUTION_STRATEGY = True
except ImportError:
    HAS_EVOLUTION_STRATEGY = False

logger = get_logger(__name__)


class TechnicalAnalystAgent(ModelAgent[TechnicalSignal]):
    """Technical Analyst Agent using ML ensemble.

    Analyzes market data and technical features to produce
    probabilistic return forecasts and trading signals.
    
    Enhanced with Evolution Strategy from tradioxen for:
    - Neural network-based buy/sell signals
    - Price pattern recognition
    - Momentum-based state representation
    """

    # Feature columns used by the models
    FEATURE_COLS = [
        "return_1", "return_5", "return_20",
        "rsi", "rsi_zscore",
        "macd", "macd_hist", "macd_signal",
        "roc_5", "roc_10", "roc_20",
        "stoch_k", "stoch_d",
        "adx", "plus_di", "minus_di",
        "volatility_5", "volatility_20", "volatility_ratio",
        "atr_percent", "bb_position", "bb_width",
        "volume_ratio", "volume_zscore",
        "price_sma20_ratio", "price_sma50_ratio",
        "sma20_slope", "sma50_slope",
    ]

    def __init__(
        self,
        model_dir: Path | None = None,
        threshold_pct: float = 0.5,
        config: dict[str, Any] | None = None,
        use_evolution_strategy: bool = False,  # Disabled by default - use when market is stable
        evolution_window: int = 30,
        evolution_epochs: int = 500,
        evolution_min_backtest: float = -10.0,  # Min backtest return to trust Evolution Strategy
    ):
        """Initialize Technical Analyst.

        Args:
            model_dir: Directory containing trained models.
            threshold_pct: Threshold for p_up/p_down calculations.
            config: Additional configuration.
            use_evolution_strategy: Whether to use Evolution Strategy signals.
                Disabled by default as it requires stable market conditions.
                Enable when you have 200+ days of data and moderate volatility.
            evolution_window: Window size for Evolution Strategy state.
            evolution_epochs: Training epochs for Evolution Strategy (500+ recommended).
            evolution_min_backtest: Minimum backtest return to trust the Evolution Strategy.
        """
        super().__init__(
            name="technical_analyst",
            model_path=str(model_dir) if model_dir else None,
            config=config,
        )
        self.model_dir = model_dir
        self.threshold_pct = threshold_pct
        self._models: dict[str, Any] = {}
        self._model_version = "v1.1.0"  # Updated version with multi-factor signals
        self._data_version = "features_v2"
        
        # Evolution Strategy settings
        self.use_evolution_strategy = use_evolution_strategy and HAS_EVOLUTION_STRATEGY
        self.evolution_window = evolution_window
        self.evolution_epochs = evolution_epochs
        self.evolution_min_backtest = evolution_min_backtest
        self._evolution_agents: dict[str, EvolutionStrategyAgent] = {}
        
        if self.use_evolution_strategy:
            logger.info(
                f"Evolution Strategy enabled: window={evolution_window}, "
                f"epochs={evolution_epochs}, min_backtest={evolution_min_backtest}%"
            )
        else:
            logger.info("Using enhanced rule-based technical analysis (Evolution Strategy disabled)")

    @property
    def description(self) -> str:
        return "Analyzes technical patterns and generates probabilistic forecasts using ML ensemble"

    async def load_model(self) -> None:
        """Load trained models from disk."""
        if self.model_dir is None:
            logger.info("No model directory specified, using rule-based fallback")
            return

        model_dir = Path(self.model_dir)
        if not model_dir.exists():
            logger.warning(f"Model directory not found: {model_dir}")
            return

        # Load XGBoost model if available
        xgb_path = model_dir / "xgboost_model.json"
        if xgb_path.exists():
            try:
                import xgboost as xgb
                self._models["xgboost"] = xgb.Booster()
                self._models["xgboost"].load_model(str(xgb_path))
                logger.info("Loaded XGBoost model")
            except Exception as e:
                logger.warning(f"Failed to load XGBoost: {e}")

        # Load LightGBM model if available
        lgb_path = model_dir / "lightgbm_model.txt"
        if lgb_path.exists():
            try:
                import lightgbm as lgb
                self._models["lightgbm"] = lgb.Booster(model_file=str(lgb_path))
                logger.info("Loaded LightGBM model")
            except Exception as e:
                logger.warning(f"Failed to load LightGBM: {e}")

    async def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Make prediction using ensemble.

        Args:
            features: Feature dictionary.

        Returns:
            Prediction dictionary with quantiles and probabilities.
        """
        # Extract feature vector
        feature_vector = self._extract_features(features)

        if not self._models:
            # Fallback to rule-based prediction
            return self._rule_based_prediction(features)

        predictions = []

        # XGBoost prediction
        if "xgboost" in self._models:
            try:
                import xgboost as xgb
                dmatrix = xgb.DMatrix([feature_vector], feature_names=self.FEATURE_COLS)
                pred = self._models["xgboost"].predict(dmatrix)[0]
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"XGBoost prediction failed: {e}")

        # LightGBM prediction
        if "lightgbm" in self._models:
            try:
                pred = self._models["lightgbm"].predict([feature_vector])[0]
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"LightGBM prediction failed: {e}")

        if not predictions:
            return self._rule_based_prediction(features)

        # Ensemble average
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions) if len(predictions) > 1 else 0.1

        return {
            "forecast_q50": mean_pred,
            "forecast_q10": mean_pred - 1.28 * std_pred,
            "forecast_q90": mean_pred + 1.28 * std_pred,
            "std": std_pred,
        }

    async def analyze(
        self,
        state: PipelineState,
        prices: Optional[np.ndarray] = None,
    ) -> TechnicalSignal:
        """Perform technical analysis with enhanced signals.

        Args:
            state: Current pipeline state with features.
            prices: Optional price array for Evolution Strategy.

        Returns:
            Technical signal with forecasts and confidence.
        """
        features = state.features
        current_price = state.current_price
        symbol = state.symbol

        # Get base prediction from ML ensemble or rule-based
        prediction = await self.predict(features)

        # Get Evolution Strategy signal if prices provided
        evolution_signal = "HOLD"
        evolution_confidence = 0.5
        evolution_result = None
        
        if prices is not None and len(prices) >= 60 and self.use_evolution_strategy:
            evolution_signal, evolution_confidence, evolution_result = \
                await self.get_evolution_signal(symbol, prices)
            logger.info(
                f"Evolution Strategy for {symbol}: {evolution_signal} "
                f"(conf={evolution_confidence:.2f})"
            )

        # Calculate probabilities
        forecast_q50 = prediction["forecast_q50"]
        std = prediction.get("std", 0.1)

        # P(return > threshold) using normal approximation
        threshold = self.threshold_pct / 100
        p_up = 1 - self._normal_cdf((threshold - forecast_q50) / (std + 1e-6))
        p_down = self._normal_cdf((-threshold - forecast_q50) / (std + 1e-6))

        # Combine signals for final direction
        if self.use_evolution_strategy and evolution_signal != "HOLD":
            direction, confidence = self.get_combined_signal(
                forecast_q50,
                evolution_signal,
                evolution_confidence,
                features,
            )
            logger.info(
                f"Combined signal for {symbol}: {direction.value} "
                f"(conf={confidence:.2f})"
            )
        else:
            # Use rule-based direction
            if forecast_q50 > threshold / 2:
                direction = Direction.LONG
            elif forecast_q50 < -threshold / 2:
                direction = Direction.SHORT
            else:
                direction = Direction.NEUTRAL
            
            # Calculate confidence
            confidence = self._calculate_confidence(prediction, features)

        # Adjust confidence based on Evolution Strategy backtest performance
        if evolution_result is not None:
            backtest_return = evolution_result.investment_return
            
            if backtest_return > 10:
                # Strong positive backtest - boost confidence
                confidence = min(confidence + 0.12, 0.92)
            elif backtest_return > 5:
                confidence = min(confidence + 0.06, 0.88)
            elif backtest_return < -30:
                # Very poor backtest - ignore Evolution Strategy, use rule-based
                logger.warning(
                    f"Evolution Strategy backtest shows {backtest_return:.1f}% loss, "
                    f"falling back to rule-based signal"
                )
                if forecast_q50 > threshold / 2:
                    direction = Direction.LONG
                elif forecast_q50 < -threshold / 2:
                    direction = Direction.SHORT
                else:
                    direction = Direction.NEUTRAL
                confidence = self._calculate_confidence(prediction, features)
                confidence = max(confidence - 0.15, 0.25)
            elif backtest_return < -10:
                # Poor backtest - significantly reduce confidence
                confidence = max(confidence - 0.20, 0.25)
            elif backtest_return < -5:
                confidence = max(confidence - 0.12, 0.30)

        # Detect regime
        regime = self._detect_regime(features)

        # Calculate edge after costs (assuming 10bps round trip)
        cost_bps = 10 / 10000
        edge = forecast_q50 - cost_bps

        # Get top features
        top_features = self._get_top_features(features)
        
        # Add Evolution Strategy to top features if active
        if evolution_result is not None:
            top_features.append(FeatureSummary(
                feature_name="Evolution Strategy",
                importance=0.30,
                current_value=evolution_confidence,
                interpretation=f"{evolution_signal} (backtest: {evolution_result.investment_return:+.1f}%)",
            ))

        # Calculate hit rate from Evolution Strategy backtest
        hit_rate = None
        if evolution_result is not None and len(evolution_result.trades) > 0:
            profitable_trades = [t for t in evolution_result.trades 
                               if t.get("pnl_pct", 0) > 0]
            hit_rate = len(profitable_trades) / max(len(evolution_result.trades), 1)

        return TechnicalSignal(
            forecast_q10=prediction["forecast_q10"] * 100,  # Convert to percentage
            forecast_q50=prediction["forecast_q50"] * 100,
            forecast_q90=prediction["forecast_q90"] * 100,
            p_up=min(max(p_up, 0.0), 1.0),
            p_down=min(max(p_down, 0.0), 1.0),
            threshold_pct=self.threshold_pct,
            edge_after_costs=edge * 100,
            regime=regime,
            direction=direction,
            confidence=confidence,
            model_version=self._model_version,
            data_version=self._data_version,
            sharpe_oos=None,  # Would come from model metadata
            hit_rate=hit_rate,
            calibration_error=None,
            top_features=top_features,
            time_horizon="1h",
            timestamp=state.timestamp,
        )

    def _extract_features(self, features: dict[str, Any]) -> list[float]:
        """Extract feature vector from dictionary."""
        vector = []
        for col in self.FEATURE_COLS:
            val = features.get(col, 0.0)
            # Handle NaN
            if val != val:  # NaN check
                val = 0.0
            vector.append(float(val))
        return vector

    def _rule_based_prediction(self, features: dict[str, Any]) -> dict[str, Any]:
        """Enhanced rule-based prediction using multiple signal components."""
        # Extract and sanitize features
        rsi = self._safe_float(features.get("rsi", 50), 50)
        macd_hist = self._safe_float(features.get("macd_hist", 0), 0)
        macd = self._safe_float(features.get("macd", 0), 0)
        macd_signal = self._safe_float(features.get("macd_signal", 0), 0)
        adx = self._safe_float(features.get("adx", 20), 20)
        plus_di = self._safe_float(features.get("plus_di", 25), 25)
        minus_di = self._safe_float(features.get("minus_di", 25), 25)
        bb_position = self._safe_float(features.get("bb_position", 0.5), 0.5)
        stoch_k = self._safe_float(features.get("stoch_k", 50), 50)
        stoch_d = self._safe_float(features.get("stoch_d", 50), 50)
        roc_5 = self._safe_float(features.get("roc_5", 0), 0)
        roc_10 = self._safe_float(features.get("roc_10", 0), 0)
        volume_ratio = self._safe_float(features.get("volume_ratio", 1), 1)
        price_sma20_ratio = self._safe_float(features.get("price_sma20_ratio", 1), 1)
        sma20_slope = self._safe_float(features.get("sma20_slope", 0), 0)

        # Multi-component signal aggregation (tradioxen-inspired)
        signals = []
        weights = []

        # 1. RSI Signal (mean reversion)
        if rsi < 25:
            signals.append(0.015)  # Strong oversold
            weights.append(1.5)
        elif rsi < 35:
            signals.append(0.008)  # Moderate oversold
            weights.append(1.0)
        elif rsi > 75:
            signals.append(-0.015)  # Strong overbought
            weights.append(1.5)
        elif rsi > 65:
            signals.append(-0.008)  # Moderate overbought
            weights.append(1.0)
        else:
            signals.append(0.0)
            weights.append(0.5)

        # 2. MACD Signal (momentum)
        macd_cross = macd - macd_signal
        if macd_cross > 0 and macd_hist > 0:
            signals.append(min(macd_hist * 50, 0.012))  # Bullish crossover
            weights.append(1.2)
        elif macd_cross < 0 and macd_hist < 0:
            signals.append(max(macd_hist * 50, -0.012))  # Bearish crossover
            weights.append(1.2)
        else:
            signals.append(macd_hist * 30)
            weights.append(0.8)

        # 3. Stochastic Signal
        if stoch_k < 20 and stoch_k > stoch_d:
            signals.append(0.010)  # Bullish from oversold
            weights.append(1.0)
        elif stoch_k > 80 and stoch_k < stoch_d:
            signals.append(-0.010)  # Bearish from overbought
            weights.append(1.0)
        else:
            signals.append(0.0)
            weights.append(0.3)

        # 4. Bollinger Band Signal (mean reversion + breakout)
        if bb_position < 0.1:
            signals.append(0.012)  # Near lower band
            weights.append(1.3)
        elif bb_position < 0.25:
            signals.append(0.006)
            weights.append(0.8)
        elif bb_position > 0.9:
            signals.append(-0.012)  # Near upper band
            weights.append(1.3)
        elif bb_position > 0.75:
            signals.append(-0.006)
            weights.append(0.8)
        else:
            signals.append(0.0)
            weights.append(0.3)

        # 5. Trend Signal (ADX + DI)
        if adx > 25:  # Strong trend
            trend_direction = (plus_di - minus_di) / max(plus_di + minus_di, 1)
            trend_signal = trend_direction * 0.008 * (adx / 50)
            signals.append(np.clip(trend_signal, -0.015, 0.015))
            weights.append(1.5)
        else:
            signals.append(0.0)
            weights.append(0.3)

        # 6. Momentum Signal (ROC)
        momentum = (roc_5 + roc_10 * 0.5) / 1.5
        signals.append(np.clip(momentum / 100, -0.01, 0.01))
        weights.append(1.0)

        # 7. Price vs SMA Signal
        price_deviation = (price_sma20_ratio - 1) * 100
        if sma20_slope > 0 and price_deviation > 0:
            signals.append(min(price_deviation * 0.001, 0.008))  # Bullish trend
            weights.append(0.8)
        elif sma20_slope < 0 and price_deviation < 0:
            signals.append(max(price_deviation * 0.001, -0.008))  # Bearish trend
            weights.append(0.8)
        else:
            signals.append(0.0)
            weights.append(0.3)

        # 8. Volume Confirmation
        volume_multiplier = 1.0
        if volume_ratio > 1.5:
            volume_multiplier = 1.3  # High volume confirms signal
        elif volume_ratio < 0.5:
            volume_multiplier = 0.7  # Low volume weakens signal

        # Weighted average of signals
        total_weight = sum(weights)
        weighted_signal = sum(s * w for s, w in zip(signals, weights)) / total_weight
        
        # Apply volume multiplier
        weighted_signal *= volume_multiplier

        # Clamp to reasonable range
        signal = np.clip(weighted_signal, -0.025, 0.025)

        # Calculate uncertainty based on signal agreement
        signal_signs = [1 if s > 0.001 else (-1 if s < -0.001 else 0) for s in signals]
        agreement = abs(sum(signal_signs)) / len(signal_signs)
        std = 0.008 * (1 - agreement * 0.5)  # Lower std when signals agree

        return {
            "forecast_q50": signal,
            "forecast_q10": signal - 1.28 * std,
            "forecast_q90": signal + 1.28 * std,
            "std": std,
            "signal_agreement": agreement,
            "component_signals": list(zip(
                ["RSI", "MACD", "Stochastic", "Bollinger", "Trend", "Momentum", "SMA"],
                signals[:7]
            )),
        }

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        """Safely convert value to float, handling NaN."""
        if value is None:
            return default
        try:
            val = float(value)
            if val != val:  # NaN check
                return default
            return val
        except (ValueError, TypeError):
            return default

    def _detect_regime(self, features: dict[str, Any]) -> Regime:
        """Detect current market regime from features."""
        adx = features.get("adx", 20)
        plus_di = features.get("plus_di", 50)
        minus_di = features.get("minus_di", 50)
        volatility_ratio = features.get("volatility_ratio", 1.0)

        # Handle NaN
        if adx != adx:
            adx = 20
        if plus_di != plus_di:
            plus_di = 50
        if minus_di != minus_di:
            minus_di = 50
        if volatility_ratio != volatility_ratio:
            volatility_ratio = 1.0

        # High volatility regime
        if volatility_ratio > 1.5:
            return Regime.VOLATILE

        # Trending regime
        if adx > 25:
            if plus_di > minus_di:
                return Regime.BULLISH
            else:
                return Regime.BEARISH

        # Ranging regime
        return Regime.NEUTRAL

    def _calculate_confidence(
        self,
        prediction: dict[str, Any],
        features: dict[str, Any],
    ) -> float:
        """Calculate confidence score using multi-factor analysis.
        
        Enhanced confidence calculation that considers:
        - Signal agreement across indicators
        - Trend strength (ADX)
        - Volume confirmation
        - Volatility regime
        - Historical pattern match (Evolution Strategy if available)
        """
        base_confidence = 0.45
        confidence_boosts = []
        confidence_penalties = []

        # Extract features safely
        rsi = self._safe_float(features.get("rsi", 50), 50)
        macd_hist = self._safe_float(features.get("macd_hist", 0), 0)
        macd = self._safe_float(features.get("macd", 0), 0)
        macd_signal = self._safe_float(features.get("macd_signal", 0), 0)
        adx = self._safe_float(features.get("adx", 20), 20)
        plus_di = self._safe_float(features.get("plus_di", 25), 25)
        minus_di = self._safe_float(features.get("minus_di", 25), 25)
        stoch_k = self._safe_float(features.get("stoch_k", 50), 50)
        bb_position = self._safe_float(features.get("bb_position", 0.5), 0.5)
        vol_ratio = self._safe_float(features.get("volatility_ratio", 1.0), 1.0)
        volume_ratio = self._safe_float(features.get("volume_ratio", 1.0), 1.0)
        
        forecast = prediction["forecast_q50"]

        # 1. Signal Agreement Score (from enhanced rule-based)
        signal_agreement = prediction.get("signal_agreement", 0.5)
        if signal_agreement > 0.7:
            confidence_boosts.append(0.15)
        elif signal_agreement > 0.5:
            confidence_boosts.append(0.08)
        elif signal_agreement < 0.3:
            confidence_penalties.append(0.10)

        # 2. RSI Alignment
        if forecast > 0.005:  # Bullish forecast
            if rsi < 35:
                confidence_boosts.append(0.12)  # RSI confirms oversold bounce
            elif rsi > 70:
                confidence_penalties.append(0.08)  # RSI contradicts
        elif forecast < -0.005:  # Bearish forecast
            if rsi > 65:
                confidence_boosts.append(0.12)  # RSI confirms overbought pullback
            elif rsi < 30:
                confidence_penalties.append(0.08)  # RSI contradicts

        # 3. MACD Alignment
        macd_bullish = macd > macd_signal and macd_hist > 0
        macd_bearish = macd < macd_signal and macd_hist < 0
        
        if forecast > 0.003 and macd_bullish:
            confidence_boosts.append(0.10)
        elif forecast < -0.003 and macd_bearish:
            confidence_boosts.append(0.10)
        elif (forecast > 0.005 and macd_bearish) or (forecast < -0.005 and macd_bullish):
            confidence_penalties.append(0.08)

        # 4. Trend Strength (ADX)
        if adx > 30:
            confidence_boosts.append(0.08)  # Strong trend = higher conviction
            # Verify trend direction aligns
            trend_up = plus_di > minus_di
            if (forecast > 0 and trend_up) or (forecast < 0 and not trend_up):
                confidence_boosts.append(0.07)
        elif adx < 15:
            confidence_penalties.append(0.05)  # Weak trend = lower conviction

        # 5. Volume Confirmation
        if volume_ratio > 1.5:
            confidence_boosts.append(0.06)  # High volume confirms move
        elif volume_ratio < 0.5:
            confidence_penalties.append(0.05)  # Low volume = suspicious

        # 6. Bollinger Band Extremes
        if bb_position < 0.15 and forecast > 0:
            confidence_boosts.append(0.08)  # Near lower band, bullish
        elif bb_position > 0.85 and forecast < 0:
            confidence_boosts.append(0.08)  # Near upper band, bearish

        # 7. Stochastic Confirmation
        if stoch_k < 25 and forecast > 0:
            confidence_boosts.append(0.06)
        elif stoch_k > 75 and forecast < 0:
            confidence_boosts.append(0.06)

        # 8. Volatility Regime Adjustment
        if vol_ratio > 2.0:
            confidence_penalties.append(0.12)  # High volatility = uncertain
        elif vol_ratio > 1.5:
            confidence_penalties.append(0.06)
        elif vol_ratio < 0.7:
            confidence_boosts.append(0.04)  # Low volatility = more predictable

        # 9. Signal Magnitude
        signal_strength = abs(forecast)
        if signal_strength > 0.015:
            confidence_boosts.append(0.08)  # Strong signal
        elif signal_strength < 0.003:
            confidence_penalties.append(0.10)  # Weak signal = uncertain

        # Calculate final confidence
        total_boost = sum(confidence_boosts)
        total_penalty = sum(confidence_penalties)
        
        confidence = base_confidence + total_boost - total_penalty

        # Ensure reasonable bounds
        return min(max(confidence, 0.25), 0.92)

    async def get_evolution_signal(
        self,
        symbol: str,
        prices: np.ndarray,
    ) -> Tuple[str, float, Optional[TradeResult]]:
        """Get trading signal from Evolution Strategy neural network.
        
        Args:
            symbol: Stock symbol
            prices: Array of closing prices (at least 60 days, ideally 200+)
            
        Returns:
            Tuple of (signal, confidence, backtest_result)
        """
        if not self.use_evolution_strategy:
            return "HOLD", 0.5, None
        
        if len(prices) < 60:
            logger.warning(f"Insufficient data for Evolution Strategy: {len(prices)} < 60")
            return "HOLD", 0.5, None
        
        try:
            # Check if we have a cached agent for this symbol
            if symbol not in self._evolution_agents:
                # Determine train/test split based on data available
                # Use 85% for training, 15% for validation (min 30 days)
                test_days = max(30, int(len(prices) * 0.15))
                train_prices = prices[:-test_days]
                
                logger.info(
                    f"Training Evolution Strategy for {symbol}: "
                    f"{len(train_prices)} train days, {test_days} test days, "
                    f"{self.evolution_epochs} epochs"
                )
                
                agent = EvolutionStrategyAgent(
                    window_size=self.evolution_window,
                    initial_money=10000.0,
                )
                
                # Train with progress updates
                agent.fit(
                    train_prices, 
                    epochs=self.evolution_epochs, 
                    print_every=max(self.evolution_epochs // 10, 50)
                )
                
                self._evolution_agents[symbol] = agent
            
            agent = self._evolution_agents[symbol]
            
            # Get current signal
            signal, confidence = agent.get_current_signal(prices)
            
            # Get backtest results on recent data (last 30 days or 15% whichever is larger)
            test_days = max(30, int(len(prices) * 0.15))
            result = agent.predict(prices[-test_days:])
            
            logger.info(
                f"Evolution Strategy {symbol}: {signal} "
                f"(conf={confidence:.2f}, backtest_return={result.investment_return:.2f}%, "
                f"trades={len(result.trades)})"
            )
            
            return signal, confidence, result
            
        except Exception as e:
            logger.error(f"Evolution Strategy error for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return "HOLD", 0.5, None

    def get_combined_signal(
        self,
        rule_based_forecast: float,
        evolution_signal: str,
        evolution_confidence: float,
        features: dict[str, Any],
    ) -> Tuple[Direction, float]:
        """Combine rule-based and Evolution Strategy signals.
        
        Args:
            rule_based_forecast: Forecast from rule-based system
            evolution_signal: Signal from Evolution Strategy (BUY/SELL/HOLD)
            evolution_confidence: Confidence from Evolution Strategy
            features: Feature dictionary
            
        Returns:
            Tuple of (direction, combined_confidence)
        """
        # Convert rule-based to direction
        if rule_based_forecast > 0.005:
            rule_direction = Direction.LONG
            rule_confidence = min(abs(rule_based_forecast) * 50, 0.9)
        elif rule_based_forecast < -0.005:
            rule_direction = Direction.SHORT
            rule_confidence = min(abs(rule_based_forecast) * 50, 0.9)
        else:
            rule_direction = Direction.NEUTRAL
            rule_confidence = 0.5
        
        # Convert evolution signal to direction
        if evolution_signal == "BUY":
            evo_direction = Direction.LONG
        elif evolution_signal == "SELL":
            evo_direction = Direction.SHORT
        else:
            evo_direction = Direction.NEUTRAL
        
        # Weight the signals (Evolution Strategy gets 40%, rule-based 60%)
        # When both agree, confidence increases significantly
        if rule_direction == evo_direction and rule_direction != Direction.NEUTRAL:
            # Strong agreement
            combined_conf = min(rule_confidence * 0.6 + evolution_confidence * 0.4 + 0.15, 0.95)
            return rule_direction, combined_conf
        elif rule_direction != Direction.NEUTRAL and evo_direction == Direction.NEUTRAL:
            # Rule-based has signal, evolution is neutral
            return rule_direction, rule_confidence * 0.8
        elif rule_direction == Direction.NEUTRAL and evo_direction != Direction.NEUTRAL:
            # Evolution has signal, rule-based is neutral
            return evo_direction, evolution_confidence * 0.75
        elif rule_direction != evo_direction:
            # Conflict - reduce confidence, go with stronger signal
            if rule_confidence > evolution_confidence:
                return rule_direction, max(rule_confidence - 0.2, 0.3)
            else:
                return evo_direction, max(evolution_confidence - 0.2, 0.3)
        else:
            # Both neutral
            return Direction.NEUTRAL, 0.5

    def _get_top_features(self, features: dict[str, Any]) -> list[FeatureSummary]:
        """Get top contributing features for explainability."""
        top = []

        # RSI
        rsi = features.get("rsi", 50)
        if rsi == rsi:  # Not NaN
            interpretation = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
            top.append(FeatureSummary(
                feature_name="RSI",
                importance=0.25,
                current_value=float(rsi),
                interpretation=interpretation,
            ))

        # MACD histogram
        macd_hist = features.get("macd_hist", 0)
        if macd_hist == macd_hist:
            interpretation = "bullish momentum" if macd_hist > 0 else "bearish momentum"
            top.append(FeatureSummary(
                feature_name="MACD Histogram",
                importance=0.20,
                current_value=float(macd_hist),
                interpretation=interpretation,
            ))

        # ADX
        adx = features.get("adx", 20)
        if adx == adx:
            interpretation = "strong trend" if adx > 25 else "weak/no trend"
            top.append(FeatureSummary(
                feature_name="ADX",
                importance=0.15,
                current_value=float(adx),
                interpretation=interpretation,
            ))

        # Volatility ratio
        vol_ratio = features.get("volatility_ratio", 1.0)
        if vol_ratio == vol_ratio:
            interpretation = "elevated volatility" if vol_ratio > 1.2 else "normal volatility"
            top.append(FeatureSummary(
                feature_name="Volatility Ratio",
                importance=0.15,
                current_value=float(vol_ratio),
                interpretation=interpretation,
            ))

        return top[:5]

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    async def train_models(
        self,
        data: pd.DataFrame,
        target_col: str = "target_return",
        test_size: float = 0.2,
        save_models: bool = True,
    ) -> dict[str, Any]:
        """Train ensemble models on historical data.

        Args:
            data: DataFrame with features and target.
            target_col: Name of target column.
            test_size: Fraction for test split.
            save_models: Whether to save trained models.

        Returns:
            Dictionary with training metrics.
        """
        logger.info(f"Training models on {len(data)} samples")
        
        # Create target if not present (next period return)
        data = data.copy()
        if target_col not in data.columns:
            data[target_col] = data["close"].pct_change().shift(-1)
        
        # Extract features - fill NaN with 0 before selecting
        feature_cols = [c for c in self.FEATURE_COLS if c in data.columns]
        
        # Drop rows where target is NaN (only the last row)
        valid_mask = data[target_col].notna()
        data_valid = data[valid_mask]
        
        if len(data_valid) < 50:
            logger.warning(f"Not enough valid samples: {len(data_valid)}")
            return {"models_trained": [], "test_metrics": {}}
        
        X = data_valid[feature_cols].fillna(0).values
        y = data_valid[target_col].values
        
        # Train/test split (time-based)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        metrics = {"models_trained": [], "test_metrics": {}}
        
        # Train XGBoost
        try:
            import xgboost as xgb
            
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
            dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)
            
            params = {
                "objective": "reg:squarederror",
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "eval_metric": "rmse",
                "seed": 42,
            }
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                evals=[(dtest, "test")],
                early_stopping_rounds=10,
                verbose_eval=False,
            )
            
            self._models["xgboost"] = model
            metrics["models_trained"].append("xgboost")
            
            # Evaluate
            preds = model.predict(dtest)
            rmse = np.sqrt(np.mean((preds - y_test) ** 2))
            metrics["test_metrics"]["xgboost_rmse"] = rmse
            
            # Feature importance
            importance = model.get_score(importance_type="gain")
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            metrics["top_features_xgb"] = top_features
            
            logger.info(f"XGBoost trained: RMSE={rmse:.6f}")
            
            # Save model
            if save_models and self.model_dir:
                self.model_dir.mkdir(parents=True, exist_ok=True)
                model.save_model(str(self.model_dir / "xgboost_model.json"))
                logger.info(f"Saved XGBoost to {self.model_dir}")
                
        except ImportError:
            logger.warning("XGBoost not available, skipping")
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
        
        # Train LightGBM
        try:
            import lightgbm as lgb
            
            train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
            test_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_cols, reference=train_data)
            
            params = {
                "objective": "regression",
                "metric": "rmse",
                "max_depth": 6,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "seed": 42,
                "verbose": -1,
            }
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[test_data],
                callbacks=[lgb.early_stopping(10, verbose=False)],
            )
            
            self._models["lightgbm"] = model
            metrics["models_trained"].append("lightgbm")
            
            # Evaluate
            preds = model.predict(X_test)
            rmse = np.sqrt(np.mean((preds - y_test) ** 2))
            metrics["test_metrics"]["lightgbm_rmse"] = rmse
            
            logger.info(f"LightGBM trained: RMSE={rmse:.6f}")
            
            # Save model
            if save_models and self.model_dir:
                model.save_model(str(self.model_dir / "lightgbm_model.txt"))
                logger.info(f"Saved LightGBM to {self.model_dir}")
                
        except ImportError:
            logger.warning("LightGBM not available, skipping")
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
        
        # Update model version
        self._model_version = f"v1.0_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        logger.info(f"Training complete: {len(metrics['models_trained'])} models trained")
        return metrics
