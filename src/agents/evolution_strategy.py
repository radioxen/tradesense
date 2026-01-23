"""Evolution Strategy Trading Agent.

Based on the tradioxen repository's successful evolution strategy approach:
- Neural network policy for buy/sell decisions
- Evolution strategy training (no gradients required)
- State representation using price differences

This module can be used standalone or integrated with the CrewAI technical agent.
"""

import numpy as np
import pandas as pd
import time
from typing import Optional, Tuple, List
from dataclasses import dataclass
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TradeResult:
    """Result of a trading simulation."""
    buy_signals: List[int]
    sell_signals: List[int]
    total_gains: float
    investment_return: float
    trades: List[dict]


class DeepEvolutionStrategy:
    """Evolution Strategy optimizer for neural network weights.
    
    Uses population-based training without gradients.
    Particularly effective for reinforcement learning in trading.
    """

    def __init__(
        self,
        weights: List[np.ndarray],
        reward_function,
        population_size: int = 15,
        sigma: float = 0.1,
        learning_rate: float = 0.03,
    ):
        """Initialize evolution strategy.
        
        Args:
            weights: Initial neural network weights
            reward_function: Function that takes weights and returns reward
            population_size: Number of weight perturbations per generation
            sigma: Standard deviation for weight noise
            learning_rate: Learning rate for weight updates
        """
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(
        self, weights: List[np.ndarray], population: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Apply population noise to weights."""
        weights_population = []
        for index, i in enumerate(population):
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        return weights_population

    def get_weights(self) -> List[np.ndarray]:
        """Get current weights."""
        return self.weights

    def train(self, epochs: int = 100, print_every: int = 10) -> List[float]:
        """Train the neural network using evolution strategy.
        
        Args:
            epochs: Number of training epochs
            print_every: Print progress every N epochs
            
        Returns:
            List of rewards per epoch
        """
        lasttime = time.time()
        history = []
        
        for i in range(epochs):
            population = []
            rewards = np.zeros(self.population_size)
            
            # Generate population of weight perturbations
            for k in range(self.population_size):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)
            
            # Evaluate each population member
            for k in range(self.population_size):
                weights_population = self._get_weight_from_population(
                    self.weights, population[k]
                )
                rewards[k] = self.reward_function(weights_population)
            
            # Normalize rewards
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
            
            # Update weights using weighted sum of perturbations
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = (
                    w
                    + self.learning_rate
                    / (self.population_size * self.sigma)
                    * np.dot(A.T, rewards).T
                )
            
            current_reward = self.reward_function(self.weights)
            history.append(current_reward)
            
            if (i + 1) % print_every == 0:
                logger.info(f"Epoch {i + 1}: reward = {current_reward:.2f}%")
        
        logger.info(f"Training completed in {time.time() - lasttime:.2f} seconds")
        return history


class TradingNeuralNetwork:
    """Simple neural network for trading decisions.
    
    Architecture: Input -> Hidden Layer (ReLU) -> Output (3 classes: hold, buy, sell)
    """

    def __init__(self, input_size: int, layer_size: int = 64, output_size: int = 3):
        """Initialize neural network.
        
        Args:
            input_size: Size of state vector (window_size)
            layer_size: Hidden layer size
            output_size: Number of actions (3: hold, buy, sell)
        """
        self.weights = [
            np.random.randn(input_size, layer_size) * 0.1,
            np.random.randn(layer_size, output_size) * 0.1,
            np.random.randn(1, layer_size) * 0.1,  # Bias
        ]

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the network.
        
        Args:
            inputs: State vector
            
        Returns:
            Action logits (3 values for hold, buy, sell)
        """
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        feed = np.maximum(feed, 0)  # ReLU activation
        decision = np.dot(feed, self.weights[1])
        return decision

    def get_weights(self) -> List[np.ndarray]:
        return self.weights

    def set_weights(self, weights: List[np.ndarray]):
        self.weights = weights


class EvolutionStrategyAgent:
    """Trading agent using evolution strategy.
    
    This agent learns to trade by evolving neural network weights
    based on cumulative returns as the reward signal.
    """

    # Hyperparameters (from tradioxen defaults)
    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(
        self,
        window_size: int = 30,
        skip: int = 1,
        initial_money: float = 10000.0,
        layer_size: int = 64,
    ):
        """Initialize the evolution strategy agent.
        
        Args:
            window_size: Number of price points for state representation
            skip: Skip factor for trading frequency
            initial_money: Starting capital
            layer_size: Hidden layer size for neural network
        """
        self.window_size = window_size
        self.skip = skip
        self.initial_money = initial_money
        self.layer_size = layer_size
        
        # Will be initialized when training data is provided
        self.model: Optional[TradingNeuralNetwork] = None
        self.es: Optional[DeepEvolutionStrategy] = None
        self.trend: Optional[np.ndarray] = None
        self.is_trained = False

    def _get_state(self, t: int) -> np.ndarray:
        """Get state representation at time t.
        
        State is the vector of price differences over the window.
        This captures momentum and trend information.
        
        Args:
            t: Current time index
            
        Returns:
            State vector of shape (1, window_size)
        """
        d = t - self.window_size + 1
        if d >= 0:
            block = self.trend[d : t + 1]
        else:
            # Pad with first value if not enough history
            block = np.concatenate([
                np.full(-d, self.trend[0]),
                self.trend[0 : t + 1]
            ])
        
        # Compute price differences
        res = []
        for i in range(len(block) - 1):
            res.append(block[i + 1] - block[i])
        
        return np.array([res])

    def _get_reward(self, weights: List[np.ndarray]) -> float:
        """Calculate reward for a set of weights.
        
        Reward is the percentage return on investment.
        
        Args:
            weights: Neural network weights to evaluate
            
        Returns:
            Percentage return
        """
        self.model.weights = weights
        initial_money = self.initial_money
        current_money = initial_money
        inventory = []
        
        state = self._get_state(0)
        
        for t in range(0, len(self.trend) - 1, self.skip):
            action = self._act(state)
            next_state = self._get_state(min(t + 1, len(self.trend) - 1))
            
            if action == 1 and current_money >= self.trend[t]:
                # Buy
                inventory.append(self.trend[t])
                current_money -= self.trend[t]
            elif action == 2 and len(inventory) > 0:
                # Sell
                bought_price = inventory.pop(0)
                current_money += self.trend[t]
            # action == 0 is hold
            
            state = next_state
        
        # Liquidate remaining inventory at final price
        while inventory:
            current_money += self.trend[-1]
            inventory.pop()
        
        return ((current_money - initial_money) / initial_money) * 100

    def _act(self, state: np.ndarray) -> int:
        """Get action from neural network.
        
        Args:
            state: Current state vector
            
        Returns:
            Action: 0=hold, 1=buy, 2=sell
        """
        decision = self.model.predict(state)
        return np.argmax(decision[0])

    def fit(
        self,
        prices: np.ndarray,
        epochs: int = 500,
        print_every: int = 50,
    ) -> List[float]:
        """Train the agent on historical price data.
        
        Args:
            prices: Array of closing prices
            epochs: Number of training epochs
            print_every: Print progress every N epochs
            
        Returns:
            Training history (rewards per epoch)
        """
        self.trend = np.array(prices, dtype=np.float64)
        
        # Initialize neural network
        self.model = TradingNeuralNetwork(
            input_size=self.window_size,
            layer_size=self.layer_size,
            output_size=3,
        )
        
        # Initialize evolution strategy
        self.es = DeepEvolutionStrategy(
            self.model.get_weights(),
            self._get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )
        
        logger.info(f"Training Evolution Strategy agent on {len(prices)} data points...")
        history = self.es.train(epochs, print_every)
        
        # Update model with best weights
        self.model.set_weights(self.es.get_weights())
        self.is_trained = True
        
        return history

    def predict(self, prices: np.ndarray) -> TradeResult:
        """Generate trading signals for price data.
        
        Args:
            prices: Array of closing prices
            
        Returns:
            TradeResult with buy/sell signals and performance
        """
        if not self.is_trained:
            raise ValueError("Agent must be trained before predicting")
        
        self.trend = np.array(prices, dtype=np.float64)
        
        initial_money = self.initial_money
        current_money = initial_money
        states_buy = []
        states_sell = []
        trades = []
        inventory = []
        
        state = self._get_state(0)
        
        for t in range(0, len(self.trend) - 1, self.skip):
            action = self._act(state)
            next_state = self._get_state(min(t + 1, len(self.trend) - 1))
            price = self.trend[t]
            
            if action == 1 and current_money >= price:
                # Buy signal
                inventory.append(price)
                current_money -= price
                states_buy.append(t)
                trades.append({
                    "day": t,
                    "action": "BUY",
                    "price": float(price),
                    "balance": float(current_money),
                })
                logger.debug(f"Day {t}: BUY at ${price:.2f}, balance: ${current_money:.2f}")
                
            elif action == 2 and len(inventory) > 0:
                # Sell signal
                bought_price = inventory.pop(0)
                current_money += price
                states_sell.append(t)
                pnl = ((price - bought_price) / bought_price) * 100
                trades.append({
                    "day": t,
                    "action": "SELL",
                    "price": float(price),
                    "bought_at": float(bought_price),
                    "pnl_pct": float(pnl),
                    "balance": float(current_money),
                })
                logger.debug(f"Day {t}: SELL at ${price:.2f}, P&L: {pnl:.2f}%, balance: ${current_money:.2f}")
            
            state = next_state
        
        # Calculate final performance
        total_gains = current_money - initial_money
        investment_return = (total_gains / initial_money) * 100
        
        return TradeResult(
            buy_signals=states_buy,
            sell_signals=states_sell,
            total_gains=total_gains,
            investment_return=investment_return,
            trades=trades,
        )

    def get_current_signal(self, recent_prices: np.ndarray) -> Tuple[str, float]:
        """Get trading signal for the current moment.
        
        Args:
            recent_prices: Recent price history (at least window_size prices)
            
        Returns:
            Tuple of (signal, confidence) where signal is "BUY", "SELL", or "HOLD"
        """
        if not self.is_trained:
            raise ValueError("Agent must be trained before getting signals")
        
        if len(recent_prices) < self.window_size:
            return "HOLD", 0.0
        
        self.trend = np.array(recent_prices[-self.window_size - 1:], dtype=np.float64)
        state = self._get_state(len(self.trend) - 1)
        
        decision = self.model.predict(state)
        probabilities = np.exp(decision[0]) / np.sum(np.exp(decision[0]))  # Softmax
        
        action = np.argmax(probabilities)
        confidence = float(probabilities[action])
        
        signal_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        return signal_map[action], confidence


def train_evolution_agent(
    symbol: str,
    prices: np.ndarray,
    initial_money: float = 10000.0,
    window_size: int = 30,
    epochs: int = 500,
) -> Tuple[EvolutionStrategyAgent, TradeResult]:
    """Train an evolution strategy agent and get results.
    
    Args:
        symbol: Stock symbol (for logging)
        prices: Historical closing prices
        initial_money: Starting capital
        window_size: State window size
        epochs: Training epochs
        
    Returns:
        Tuple of (trained agent, trading results)
    """
    logger.info(f"Training Evolution Strategy agent for {symbol}")
    
    # Split data: 70% train, 30% test
    split_idx = int(len(prices) * 0.7)
    train_prices = prices[:split_idx]
    test_prices = prices[split_idx:]
    
    # Create and train agent
    agent = EvolutionStrategyAgent(
        window_size=window_size,
        initial_money=initial_money,
    )
    
    history = agent.fit(train_prices, epochs=epochs)
    
    # Test on held-out data
    result = agent.predict(test_prices)
    
    logger.info(f"{symbol} Results: Return={result.investment_return:.2f}%, Trades={len(result.trades)}")
    
    return agent, result
