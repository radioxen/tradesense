"""RL Position Sizing Environment.

Gym environment for training RL agents to optimize position sizing
based on signal quality and market conditions.
"""

from typing import Any, SupportsFloat
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from src.utils.logging import get_logger


logger = get_logger(__name__)


class SizingEnv(gym.Env):
    """RL environment for position sizing.

    Goal: Maximize risk-adjusted returns by optimally sizing positions
    based on signal quality, regime, and portfolio constraints.

    State:
        - signal_strength: Hybrid signal strength (-1 to 1)
        - signal_confidence: Signal confidence (0 to 1)
        - volatility: Current volatility regime
        - current_position: Current position fraction
        - time_of_day: Fraction of trading day
        - drawdown: Current drawdown fraction

    Action:
        - Continuous: target position (-1 to 1) representing short to long

    Reward:
        - PnL after costs - lambda * drawdown_penalty - eta * turnover_penalty
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_steps: int = 252,  # One trading year
        initial_capital: float = 100000,
        max_position: float = 0.2,  # 20% max position
        cost_bps: float = 10,  # 10 bps round trip
        lambda_drawdown: float = 0.5,  # Drawdown penalty weight
        eta_turnover: float = 0.1,  # Turnover penalty weight
        render_mode: str | None = None,
    ):
        """Initialize sizing environment.

        Args:
            max_steps: Maximum steps per episode.
            initial_capital: Starting capital.
            max_position: Maximum position as fraction of capital.
            cost_bps: Transaction costs in basis points.
            lambda_drawdown: Drawdown penalty coefficient.
            eta_turnover: Turnover penalty coefficient.
            render_mode: Rendering mode.
        """
        super().__init__()

        self.max_steps = max_steps
        self.initial_capital = initial_capital
        self.max_position = max_position
        self.cost_bps = cost_bps
        self.lambda_drawdown = lambda_drawdown
        self.eta_turnover = eta_turnover
        self.render_mode = render_mode

        # State: [signal, confidence, volatility, position, time, drawdown]
        self.observation_space = spaces.Box(
            low=np.array([-1, 0, 0, -1, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 3, 1, 1, 1], dtype=np.float32),
        )

        # Action: target position fraction
        self.action_space = spaces.Box(
            low=np.array([-1], dtype=np.float32),
            high=np.array([1], dtype=np.float32),
        )

        self._reset_state()

    def _reset_state(self):
        """Reset internal state."""
        self.current_step = 0
        self.capital = self.initial_capital
        self.current_position = 0.0
        self.high_water_mark = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self._generate_signals()

    def _generate_signals(self):
        """Generate synthetic signal series for episode."""
        n = self.max_steps

        # Generate returns with momentum
        returns = np.random.normal(0.0005, 0.015, n)

        # Add some trending behavior
        trend = np.cumsum(np.random.normal(0, 0.0001, n))
        returns += trend

        # Generate signals (noisy predictor of returns)
        noise = np.random.normal(0, 0.3, n)
        self.true_returns = returns
        self.signals = np.clip(returns * 30 + noise, -1, 1)  # Scaled and noisy

        # Generate confidence (higher when signal is extreme)
        self.confidences = 0.5 + 0.3 * np.abs(self.signals) + np.random.normal(0, 0.1, n)
        self.confidences = np.clip(self.confidences, 0.2, 0.95)

        # Generate volatility regime
        vol_changes = np.cumsum(np.random.normal(0, 0.05, n))
        self.volatilities = np.clip(1.0 + vol_changes, 0.5, 2.5)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment."""
        super().reset(seed=seed)
        self._reset_state()

        obs = self._get_observation()
        info = {"capital": self.capital}

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        """Execute one step.

        Args:
            action: Target position [-1, 1].

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Parse action
        target_position = float(np.clip(action[0], -1, 1)) * self.max_position

        # Calculate turnover
        turnover = abs(target_position - self.current_position)

        # Transaction cost
        cost = turnover * self.capital * self.cost_bps / 10000

        # Update position
        old_position = self.current_position
        self.current_position = target_position

        # Simulate return
        market_return = self.true_returns[self.current_step]
        position_return = self.current_position * market_return

        # Update capital
        pnl = position_return * self.capital - cost
        self.capital += pnl
        self.equity_curve.append(self.capital)

        # Update high water mark and drawdown
        if self.capital > self.high_water_mark:
            self.high_water_mark = self.capital
        drawdown = (self.high_water_mark - self.capital) / self.high_water_mark

        # Calculate reward
        normalized_pnl = pnl / self.initial_capital
        drawdown_penalty = self.lambda_drawdown * drawdown
        turnover_penalty = self.eta_turnover * turnover

        reward = normalized_pnl - drawdown_penalty - turnover_penalty

        # Advance step
        self.current_step += 1

        # Check termination
        terminated = self.capital <= 0.5 * self.initial_capital  # 50% loss
        truncated = self.current_step >= self.max_steps

        obs = self._get_observation()
        info = {
            "capital": self.capital,
            "pnl": pnl,
            "drawdown": drawdown,
            "position": self.current_position,
            "turnover": turnover,
            "sharpe": self._calculate_sharpe(),
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_step >= self.max_steps:
            # End of episode
            signal = 0.0
            confidence = 0.5
            volatility = 1.0
        else:
            signal = self.signals[self.current_step]
            confidence = self.confidences[self.current_step]
            volatility = self.volatilities[self.current_step]

        time_fraction = self.current_step / self.max_steps
        drawdown = (self.high_water_mark - self.capital) / max(self.high_water_mark, 1)

        return np.array([
            signal,
            confidence,
            volatility,
            self.current_position / self.max_position,  # Normalized
            time_fraction,
            drawdown,
        ], dtype=np.float32)

    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio of episode so far."""
        if len(self.equity_curve) < 5:
            return 0.0

        returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
        if returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252)

    def render(self):
        """Render environment state."""
        if self.render_mode == "human":
            print(f"Step {self.current_step}: Capital=${self.capital:.2f}, "
                  f"Position={self.current_position:.1%}, "
                  f"Sharpe={self._calculate_sharpe():.2f}")
