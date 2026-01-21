"""Base agent class and interfaces for all trading agents."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from src.orchestrator.contracts import PipelineState
from src.utils.logging import get_logger


logger = get_logger(__name__)

# Type variable for agent output
OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseAgent(ABC, Generic[OutputT]):
    """Base class for all trading agents.

    All agents must inherit from this class and implement the `analyze` method.
    Agents receive the current pipeline state and produce a typed output.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """Initialize the agent.

        Args:
            name: Unique name for this agent instance.
            config: Optional configuration dictionary.
        """
        self.name = name
        self.config = config or {}
        self._logger = get_logger(f"agent.{name}")

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the agent's role."""
        ...

    @abstractmethod
    async def analyze(self, state: PipelineState) -> OutputT:
        """Perform analysis and produce output.

        Args:
            state: Current pipeline state with all available context.

        Returns:
            Agent-specific output (must be a Pydantic model).
        """
        ...

    async def __call__(self, state: PipelineState) -> OutputT:
        """Execute the agent with logging and error handling.

        Args:
            state: Current pipeline state.

        Returns:
            Agent output.
        """
        start_time = datetime.now()
        self._logger.info(
            f"Agent '{self.name}' starting analysis for {state.symbol} at {state.timestamp}"
        )

        try:
            result = await self.analyze(state)
            elapsed = (datetime.now() - start_time).total_seconds()
            self._logger.info(
                f"Agent '{self.name}' completed in {elapsed:.2f}s"
            )
            return result
        except Exception as e:
            self._logger.error(f"Agent '{self.name}' failed: {e}", exc_info=True)
            raise


class LLMAgent(BaseAgent[OutputT]):
    """Base class for LLM-powered agents.

    Provides common functionality for agents that use LLM for reasoning.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        config: dict[str, Any] | None = None,
    ):
        """Initialize LLM agent.

        Args:
            name: Agent name.
            system_prompt: System prompt for the LLM.
            config: Configuration dictionary.
        """
        super().__init__(name, config)
        self.system_prompt = system_prompt
        # Default to GPT-5 mini, can be overridden by config
        self._model = config.get("model", "gpt-5-mini-2025-08-07") if config else "gpt-5-mini-2025-08-07"
        self._temperature = config.get("temperature", 0.1) if config else 0.1

    @abstractmethod
    def _build_user_prompt(self, state: PipelineState) -> str:
        """Build the user prompt from current state.

        Args:
            state: Current pipeline state.

        Returns:
            Formatted user prompt string.
        """
        ...

    @abstractmethod
    def _parse_response(self, response: str) -> OutputT:
        """Parse LLM response into output model.

        Args:
            response: Raw LLM response text.

        Returns:
            Parsed output model.
        """
        ...


class ModelAgent(BaseAgent[OutputT]):
    """Base class for ML model-powered agents.

    Provides common functionality for agents that use trained ML models.
    """

    def __init__(
        self,
        name: str,
        model_path: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        """Initialize model agent.

        Args:
            name: Agent name.
            model_path: Path to trained model.
            config: Configuration dictionary.
        """
        super().__init__(name, config)
        self.model_path = model_path
        self._model = None

    @abstractmethod
    async def load_model(self) -> None:
        """Load the trained model from disk."""
        ...

    @abstractmethod
    async def predict(self, features: dict[str, Any]) -> Any:
        """Make prediction using the model.

        Args:
            features: Feature dictionary.

        Returns:
            Model prediction.
        """
        ...


class RuleBasedAgent(BaseAgent[OutputT]):
    """Base class for deterministic rule-based agents.

    Used for agents that apply fixed rules (like Risk Guardian).
    """

    @abstractmethod
    def _apply_rules(self, state: PipelineState) -> OutputT:
        """Apply rules to produce output.

        Args:
            state: Current pipeline state.

        Returns:
            Rule-based output.
        """
        ...

    async def analyze(self, state: PipelineState) -> OutputT:
        """Run rule-based analysis.

        Args:
            state: Current pipeline state.

        Returns:
            Rule-based output.
        """
        return self._apply_rules(state)
