"""RL Execution Environment.

Gym environment for training RL agents to minimize execution slippage.
"""

from typing import Any, SupportsFloat
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # Fallback for older gym
    import gym
    from gym import spaces

from src.utils.logging import get_logger


logger = get_logger(__name__)


class ExecutionEnv(gym.Env):
    """RL environment for order execution.

    Goal: Minimize slippage + fees while executing a target order.

    State:
        - spread: Current bid-ask spread (normalized)
        - volatility: Recent price volatility
        - volume_ratio: Current volume vs average
        - time_remaining: Fraction of time remaining
        - inventory: Remaining quantity to execute (normalized)
        - price_drift: Recent price movement direction

    Action:
        - Discrete: {wait, market_small, market_large, limit_aggressive, limit_passive}
        - Or continuous: (order_size_fraction, limit_offset)

    Reward:
        - Negative of execution cost (slippage + fees + adverse selection)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_steps: int = 100,
        target_quantity: float = 1000,
        initial_price: float = 100.0,
        volatility: float = 0.02,
        spread_bps: float = 5.0,
        market_impact_bps: float = 10.0,
        discrete_actions: bool = True,
        render_mode: str | None = None,
    ):
        """Initialize execution environment.

        Args:
            max_steps: Maximum steps per episode.
            target_quantity: Target order quantity.
            initial_price: Starting price.
            volatility: Price volatility (per step).
            spread_bps: Bid-ask spread in basis points.
            market_impact_bps: Market impact for large orders.
            discrete_actions: Use discrete action space.
            render_mode: Rendering mode.
        """
        super().__init__()

        self.max_steps = max_steps
        self.target_quantity = target_quantity
        self.initial_price = initial_price
        self.volatility = volatility
        self.spread_bps = spread_bps
        self.market_impact_bps = market_impact_bps
        self.discrete_actions = discrete_actions
        self.render_mode = render_mode

        # State space: [spread, volatility, volume_ratio, time_remaining, inventory, price_drift]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, -1], dtype=np.float32),
            high=np.array([1, 1, 3, 1, 1, 1], dtype=np.float32),
        )

        if discrete_actions:
            # 0: wait, 1: market_10%, 2: market_25%, 3: limit_tight, 4: limit_wide
            self.action_space = spaces.Discrete(5)
        else:
            # [order_fraction (0-1), limit_offset_bps (0-50)]
            self.action_space = spaces.Box(
                low=np.array([0, 0], dtype=np.float32),
                high=np.array([1, 50], dtype=np.float32),
            )

        self._reset_state()

    def _reset_state(self):
        """Reset internal state."""
        self.current_step = 0
        self.current_price = self.initial_price
        self.remaining_quantity = self.target_quantity
        self.executed_quantity = 0
        self.total_cost = 0.0
        self.price_history = [self.initial_price]
        self.pending_limit_order = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment.

        Args:
            seed: Random seed.
            options: Additional options.

        Returns:
            Initial observation and info.
        """
        super().reset(seed=seed)
        self._reset_state()

        obs = self._get_observation()
        info = {"remaining_quantity": self.remaining_quantity}

        return obs, info

    def step(
        self,
        action: int | np.ndarray,
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        """Execute one step.

        Args:
            action: Action to take.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1

        # Parse action
        if self.discrete_actions:
            order_fraction, limit_offset = self._decode_discrete_action(action)
        else:
            order_fraction = float(action[0])
            limit_offset = float(action[1])

        # Simulate price movement
        price_change = np.random.normal(0, self.volatility * self.current_price)
        self.current_price = max(0.01, self.current_price + price_change)
        self.price_history.append(self.current_price)

        # Process pending limit orders
        if self.pending_limit_order:
            filled = self._check_limit_fill()
            if filled:
                self.pending_limit_order = None

        # Execute new order
        reward = 0.0
        if order_fraction > 0 and self.remaining_quantity > 0:
            order_quantity = order_fraction * self.remaining_quantity

            if limit_offset == 0:
                # Market order
                reward = self._execute_market_order(order_quantity)
            else:
                # Limit order
                self._place_limit_order(order_quantity, limit_offset)

        # Check termination
        terminated = self.remaining_quantity <= 0
        truncated = self.current_step >= self.max_steps

        # Penalty for not completing order
        if truncated and self.remaining_quantity > 0:
            # Force complete at market with penalty
            penalty = self._execute_market_order(self.remaining_quantity)
            reward += penalty - 0.01 * self.target_quantity  # Extra penalty

        obs = self._get_observation()
        info = {
            "remaining_quantity": self.remaining_quantity,
            "total_cost": self.total_cost,
            "avg_price": self.total_cost / max(self.executed_quantity, 1),
            "slippage_bps": self._calculate_slippage_bps(),
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Normalized spread
        spread = self.spread_bps / 100  # 0 to ~0.1

        # Realized volatility
        if len(self.price_history) > 5:
            returns = np.diff(np.log(self.price_history[-10:]))
            vol = np.std(returns) / self.volatility  # Normalized
        else:
            vol = 0.5

        # Volume ratio (simulated)
        volume_ratio = 1.0 + np.random.normal(0, 0.3)

        # Time remaining
        time_remaining = 1 - self.current_step / self.max_steps

        # Inventory
        inventory = self.remaining_quantity / self.target_quantity

        # Price drift
        if len(self.price_history) > 3:
            drift = (self.current_price - self.price_history[-3]) / self.price_history[-3]
            drift = np.clip(drift * 10, -1, 1)
        else:
            drift = 0.0

        return np.array([
            spread, vol, volume_ratio, time_remaining, inventory, drift
        ], dtype=np.float32)

    def _decode_discrete_action(self, action: int) -> tuple[float, float]:
        """Decode discrete action to (order_fraction, limit_offset)."""
        if action == 0:  # Wait
            return 0.0, 0.0
        elif action == 1:  # Market 10%
            return 0.1, 0.0
        elif action == 2:  # Market 25%
            return 0.25, 0.0
        elif action == 3:  # Limit tight
            return 0.15, 5.0
        else:  # action == 4: Limit wide
            return 0.2, 15.0

    def _execute_market_order(self, quantity: float) -> float:
        """Execute market order and return reward."""
        # Calculate execution price with spread and impact
        half_spread = self.spread_bps / 10000 / 2
        impact = (quantity / self.target_quantity) * self.market_impact_bps / 10000

        exec_price = self.current_price * (1 + half_spread + impact)
        cost = quantity * exec_price

        self.remaining_quantity -= quantity
        self.executed_quantity += quantity
        self.total_cost += cost

        # Reward is negative cost relative to mid price
        fair_cost = quantity * self.current_price
        slippage = cost - fair_cost

        return -slippage / self.initial_price  # Normalized

    def _place_limit_order(self, quantity: float, offset_bps: float):
        """Place a limit order."""
        limit_price = self.current_price * (1 + offset_bps / 10000)
        self.pending_limit_order = {
            "quantity": quantity,
            "price": limit_price,
            "placed_step": self.current_step,
        }

    def _check_limit_fill(self) -> bool:
        """Check if pending limit order is filled."""
        if not self.pending_limit_order:
            return False

        # Fill if price touches limit
        if self.current_price <= self.pending_limit_order["price"]:
            quantity = self.pending_limit_order["quantity"]
            exec_price = self.pending_limit_order["price"]
            cost = quantity * exec_price

            self.remaining_quantity -= quantity
            self.executed_quantity += quantity
            self.total_cost += cost
            return True

        return False

    def _calculate_slippage_bps(self) -> float:
        """Calculate slippage in basis points."""
        if self.executed_quantity == 0:
            return 0.0
        avg_price = self.total_cost / self.executed_quantity
        slippage = (avg_price - self.initial_price) / self.initial_price * 10000
        return slippage

    def render(self):
        """Render environment state."""
        if self.render_mode == "human":
            print(f"Step {self.current_step}: Price=${self.current_price:.2f}, "
                  f"Remaining={self.remaining_quantity:.0f}, "
                  f"Slippage={self._calculate_slippage_bps():.1f}bps")
