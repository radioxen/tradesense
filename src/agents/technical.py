"""Technical Analyst Agent with ML ensemble models.

Uses multiple model families for robust predictions:
- XGBoost/LightGBM (tree-based)
- LSTM (deep learning for sequences)
- Linear baseline (calibration reference)

Outputs probabilistic forecasts with confidence intervals.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
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


logger = get_logger(__name__)


class TechnicalAnalystAgent(ModelAgent[TechnicalSignal]):
    """Technical Analyst Agent using ML ensemble.

    Analyzes market data and technical features to produce
    probabilistic return forecasts and trading signals.
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
    ):
        """Initialize Technical Analyst.

        Args:
            model_dir: Directory containing trained models.
            threshold_pct: Threshold for p_up/p_down calculations.
            config: Additional configuration.
        """
        super().__init__(
            name="technical_analyst",
            model_path=str(model_dir) if model_dir else None,
            config=config,
        )
        self.model_dir = model_dir
        self.threshold_pct = threshold_pct
        self._models: dict[str, Any] = {}
        self._model_version = "v0.1.0"
        self._data_version = "features_v1"

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

    async def analyze(self, state: PipelineState) -> TechnicalSignal:
        """Perform technical analysis.

        Args:
            state: Current pipeline state with features.

        Returns:
            Technical signal with forecasts and confidence.
        """
        features = state.features
        current_price = state.current_price

        # Get prediction
        prediction = await self.predict(features)

        # Calculate probabilities
        forecast_q50 = prediction["forecast_q50"]
        std = prediction.get("std", 0.1)

        # P(return > threshold) using normal approximation
        threshold = self.threshold_pct / 100
        p_up = 1 - self._normal_cdf((threshold - forecast_q50) / (std + 1e-6))
        p_down = self._normal_cdf((-threshold - forecast_q50) / (std + 1e-6))

        # Determine direction
        if forecast_q50 > threshold / 2:
            direction = Direction.LONG
        elif forecast_q50 < -threshold / 2:
            direction = Direction.SHORT
        else:
            direction = Direction.NEUTRAL

        # Detect regime
        regime = self._detect_regime(features)

        # Calculate edge after costs (assuming 10bps round trip)
        cost_bps = 10 / 10000
        edge = forecast_q50 - cost_bps

        # Confidence based on model agreement and feature quality
        confidence = self._calculate_confidence(prediction, features)

        # Get top features
        top_features = self._get_top_features(features)

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
            hit_rate=None,
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
        """Fallback rule-based prediction when no models available."""
        rsi = features.get("rsi", 50)
        macd_hist = features.get("macd_hist", 0)
        adx = features.get("adx", 20)
        bb_position = features.get("bb_position", 0.5)

        # Handle NaN
        if rsi != rsi:
            rsi = 50
        if macd_hist != macd_hist:
            macd_hist = 0
        if adx != adx:
            adx = 20
        if bb_position != bb_position:
            bb_position = 0.5

        # Simple momentum signal
        signal = 0.0

        # RSI component
        if rsi < 30:
            signal += 0.003  # Oversold, expect bounce
        elif rsi > 70:
            signal -= 0.003  # Overbought, expect pullback

        # MACD component
        signal += np.clip(macd_hist * 100, -0.005, 0.005)

        # Bollinger position
        if bb_position < 0.2:
            signal += 0.002
        elif bb_position > 0.8:
            signal -= 0.002

        # Trend strength adjustment
        if adx > 25:
            signal *= 1.5

        # Clamp to reasonable range
        signal = np.clip(signal, -0.02, 0.02)

        return {
            "forecast_q50": signal,
            "forecast_q10": signal - 0.01,
            "forecast_q90": signal + 0.01,
            "std": 0.005,
        }

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
        """Calculate confidence score."""
        base_confidence = 0.5

        # Higher confidence when signals agree
        rsi = features.get("rsi", 50)
        macd_hist = features.get("macd_hist", 0)
        forecast = prediction["forecast_q50"]

        # Handle NaN
        if rsi != rsi:
            rsi = 50
        if macd_hist != macd_hist:
            macd_hist = 0

        # Check signal alignment
        rsi_bullish = rsi < 40
        rsi_bearish = rsi > 60
        macd_bullish = macd_hist > 0
        macd_bearish = macd_hist < 0

        alignment = 0
        if forecast > 0:
            if rsi_bullish:
                alignment += 1
            if macd_bullish:
                alignment += 1
        elif forecast < 0:
            if rsi_bearish:
                alignment += 1
            if macd_bearish:
                alignment += 1

        confidence = base_confidence + alignment * 0.15

        # Lower confidence in high volatility
        vol_ratio = features.get("volatility_ratio", 1.0)
        if vol_ratio != vol_ratio:
            vol_ratio = 1.0
        if vol_ratio > 1.5:
            confidence -= 0.1

        return min(max(confidence, 0.2), 0.95)

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
