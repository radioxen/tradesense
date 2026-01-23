"""LSTM Price Forecaster.

Based on tradioxen repository's LSTM implementation that achieved 95.7% accuracy.
Provides probabilistic price forecasts for short-term trading.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import TensorFlow, fall back to numpy-based forecasting if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("TensorFlow not available, using fallback forecaster")


@dataclass
class Forecast:
    """Price forecast result."""
    predicted_prices: np.ndarray
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]]
    direction: str  # "UP", "DOWN", or "NEUTRAL"
    confidence: float
    expected_return: float


class LSTMForecaster:
    """LSTM-based price forecaster for trading signals.
    
    Uses a multi-layer LSTM architecture trained on historical prices.
    Produces probabilistic forecasts with confidence intervals.
    """

    def __init__(
        self,
        sequence_length: int = 30,
        num_layers: int = 2,
        layer_size: int = 128,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ):
        """Initialize LSTM forecaster.
        
        Args:
            sequence_length: Number of time steps for input sequences
            num_layers: Number of LSTM layers
            layer_size: Units per LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for Adam optimizer
        """
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model: Optional[Sequential] = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False

    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build the LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            
        Returns:
            Compiled Keras Sequential model
        """
        if not HAS_TENSORFLOW:
            raise RuntimeError("TensorFlow is required for LSTM forecasting")
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            self.layer_size,
            return_sequences=(self.num_layers > 1),
            input_shape=input_shape,
        ))
        model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, self.num_layers):
            return_seq = i < self.num_layers - 1
            model.add(LSTM(self.layer_size, return_sequences=return_seq))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer - predict next price
        model.add(Dense(1))
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae'],
        )
        
        return model

    def _create_sequences(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training.
        
        Args:
            data: Normalized price data
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def fit(
        self,
        prices: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.1,
        verbose: int = 1,
    ) -> dict:
        """Train the LSTM model on historical prices.
        
        Args:
            prices: Array of closing prices
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
            verbose: Verbosity level (0, 1, or 2)
            
        Returns:
            Training history dictionary
        """
        if not HAS_TENSORFLOW:
            logger.warning("TensorFlow not available, skipping LSTM training")
            self.is_trained = True
            return {"loss": [], "val_loss": []}
        
        # Normalize prices
        prices_reshaped = prices.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(prices_reshaped)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build model
        self.model = self._build_model((self.sequence_length, 1))
        
        logger.info(f"Training LSTM on {len(X)} sequences...")
        
        # Train
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
        )
        
        self.is_trained = True
        logger.info("LSTM training complete")
        
        return history.history

    def predict(
        self,
        recent_prices: np.ndarray,
        forecast_days: int = 5,
        n_simulations: int = 100,
    ) -> Forecast:
        """Generate price forecasts with confidence intervals.
        
        Uses Monte Carlo dropout for uncertainty estimation.
        
        Args:
            recent_prices: Recent price history
            forecast_days: Number of days to forecast
            n_simulations: Number of MC simulations for confidence
            
        Returns:
            Forecast object with predictions and confidence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        
        if not HAS_TENSORFLOW:
            # Fallback: simple trend-based forecast
            return self._fallback_forecast(recent_prices, forecast_days)
        
        # Normalize
        prices_reshaped = recent_prices.reshape(-1, 1)
        scaled_data = self.scaler.transform(prices_reshaped)
        
        # Generate forecasts
        all_predictions = []
        
        for _ in range(n_simulations):
            current_sequence = scaled_data[-self.sequence_length:].copy()
            predictions = []
            
            for _ in range(forecast_days):
                X = current_sequence.reshape(1, self.sequence_length, 1)
                pred = self.model(X, training=True)  # Enable dropout for MC
                predictions.append(pred[0, 0].numpy())
                
                # Update sequence
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = pred[0, 0].numpy()
            
            all_predictions.append(predictions)
        
        all_predictions = np.array(all_predictions)
        
        # Calculate statistics
        mean_predictions = all_predictions.mean(axis=0)
        std_predictions = all_predictions.std(axis=0)
        
        # Inverse transform
        predicted_prices = self.scaler.inverse_transform(
            mean_predictions.reshape(-1, 1)
        ).flatten()
        
        lower_bound = self.scaler.inverse_transform(
            (mean_predictions - 2 * std_predictions).reshape(-1, 1)
        ).flatten()
        
        upper_bound = self.scaler.inverse_transform(
            (mean_predictions + 2 * std_predictions).reshape(-1, 1)
        ).flatten()
        
        # Determine direction and confidence
        current_price = recent_prices[-1]
        final_price = predicted_prices[-1]
        expected_return = (final_price - current_price) / current_price * 100
        
        if expected_return > 1.0:
            direction = "UP"
            confidence = min(abs(expected_return) / 5.0, 1.0)  # Scale to 0-1
        elif expected_return < -1.0:
            direction = "DOWN"
            confidence = min(abs(expected_return) / 5.0, 1.0)
        else:
            direction = "NEUTRAL"
            confidence = 0.5
        
        return Forecast(
            predicted_prices=predicted_prices,
            confidence_intervals=(lower_bound, upper_bound),
            direction=direction,
            confidence=confidence,
            expected_return=expected_return,
        )

    def _fallback_forecast(
        self, recent_prices: np.ndarray, forecast_days: int
    ) -> Forecast:
        """Simple fallback forecast when TensorFlow is not available.
        
        Uses exponential moving average and momentum.
        """
        # Calculate momentum
        returns = np.diff(recent_prices) / recent_prices[:-1]
        avg_return = np.mean(returns[-10:])  # Last 10 days average
        volatility = np.std(returns[-20:])  # Last 20 days volatility
        
        # Simple linear extrapolation with momentum
        current_price = recent_prices[-1]
        predicted_prices = []
        
        for i in range(forecast_days):
            next_price = current_price * (1 + avg_return)
            predicted_prices.append(next_price)
            current_price = next_price
        
        predicted_prices = np.array(predicted_prices)
        
        # Confidence intervals based on volatility
        lower_bound = predicted_prices * (1 - 2 * volatility)
        upper_bound = predicted_prices * (1 + 2 * volatility)
        
        # Determine direction
        expected_return = (predicted_prices[-1] - recent_prices[-1]) / recent_prices[-1] * 100
        
        if expected_return > 0.5:
            direction = "UP"
        elif expected_return < -0.5:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"
        
        confidence = min(abs(avg_return / volatility), 1.0) if volatility > 0 else 0.5
        
        return Forecast(
            predicted_prices=predicted_prices,
            confidence_intervals=(lower_bound, upper_bound),
            direction=direction,
            confidence=confidence,
            expected_return=expected_return,
        )


class EnsembleForecaster:
    """Ensemble of forecasting methods for robust predictions.
    
    Combines LSTM, momentum, and mean reversion signals.
    """

    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.lstm = LSTMForecaster(sequence_length=sequence_length)

    def fit(self, prices: np.ndarray, **kwargs) -> dict:
        """Train the ensemble."""
        return self.lstm.fit(prices, **kwargs)

    def get_signal(
        self, recent_prices: np.ndarray, rsi: float = 50.0
    ) -> Tuple[str, float, str]:
        """Get ensemble trading signal.
        
        Args:
            recent_prices: Recent price history
            rsi: Current RSI value (0-100)
            
        Returns:
            Tuple of (signal, confidence, reasoning)
        """
        signals = []
        weights = []
        
        # 1. LSTM Forecast
        try:
            forecast = self.lstm.predict(recent_prices, forecast_days=3)
            lstm_signal = forecast.direction
            lstm_conf = forecast.confidence
            signals.append((lstm_signal, lstm_conf))
            weights.append(0.4)  # 40% weight
        except Exception as e:
            logger.warning(f"LSTM forecast failed: {e}")
        
        # 2. Momentum Signal
        returns = np.diff(recent_prices) / recent_prices[:-1]
        short_momentum = np.mean(returns[-5:])
        long_momentum = np.mean(returns[-20:])
        
        if short_momentum > 0.01 and short_momentum > long_momentum:
            mom_signal = "UP"
            mom_conf = min(abs(short_momentum) * 10, 1.0)
        elif short_momentum < -0.01 and short_momentum < long_momentum:
            mom_signal = "DOWN"
            mom_conf = min(abs(short_momentum) * 10, 1.0)
        else:
            mom_signal = "NEUTRAL"
            mom_conf = 0.5
        
        signals.append((mom_signal, mom_conf))
        weights.append(0.3)  # 30% weight
        
        # 3. Mean Reversion (RSI-based)
        if rsi < 30:
            mr_signal = "UP"  # Oversold, expect bounce
            mr_conf = (30 - rsi) / 30
        elif rsi > 70:
            mr_signal = "DOWN"  # Overbought, expect pullback
            mr_conf = (rsi - 70) / 30
        else:
            mr_signal = "NEUTRAL"
            mr_conf = 0.5
        
        signals.append((mr_signal, mr_conf))
        weights.append(0.3)  # 30% weight
        
        # Aggregate signals
        signal_scores = {"UP": 0, "DOWN": 0, "NEUTRAL": 0}
        total_weight = sum(weights)
        
        for (sig, conf), w in zip(signals, weights):
            signal_scores[sig] += conf * w
        
        # Normalize
        for k in signal_scores:
            signal_scores[k] /= total_weight
        
        # Pick best signal
        best_signal = max(signal_scores, key=signal_scores.get)
        confidence = signal_scores[best_signal]
        
        # Build reasoning
        reasons = []
        if lstm_conf > 0.6:
            reasons.append(f"LSTM predicts {lstm_signal}")
        if abs(short_momentum) > 0.01:
            reasons.append(f"Momentum is {'positive' if short_momentum > 0 else 'negative'}")
        if rsi < 30 or rsi > 70:
            reasons.append(f"RSI at {rsi:.0f} ({'oversold' if rsi < 30 else 'overbought'})")
        
        reasoning = "; ".join(reasons) if reasons else "Mixed signals"
        
        return best_signal, confidence, reasoning
