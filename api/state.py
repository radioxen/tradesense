"""Application state management."""

import os
from pathlib import Path
from typing import Any

import structlog
import yaml

from src.execution.broker import AlpacaBroker, SimulatedBroker, Broker
from src.agents.researcher import ResearcherAgent

logger = structlog.get_logger()


class AppState:
    """Centralized application state."""
    
    def __init__(self):
        self.broker: Broker | None = None
        self.researcher: ResearcherAgent | None = None
        self.agent_configs: dict[str, dict] = {}
        self.activity_log: list[dict] = []
        self.scan_results: list[dict] = []
        
    async def initialize(self):
        """Initialize all services."""
        # Load agent configs
        self._load_agent_configs()
        
        # Initialize broker
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_API_SECRET")
        
        if api_key and api_secret:
            self.broker = AlpacaBroker(
                api_key=api_key,
                api_secret=api_secret,
                paper=True,  # Always paper trading for safety
            )
            logger.info("Initialized Alpaca broker (paper trading)")
        else:
            self.broker = SimulatedBroker(initial_cash=100_000)
            logger.warning("Using simulated broker - no Alpaca credentials")
        
        # Initialize researcher
        self.researcher = ResearcherAgent(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            perplexity_api_key=os.getenv("PERPLEXITY_API_KEY"),
        )
        logger.info("Initialized researcher agent")
        
    def _load_agent_configs(self):
        """Load agent configurations from YAML."""
        config_path = Path(__file__).parent.parent / "configs" / "agents.yaml"
        
        if config_path.exists():
            with open(config_path) as f:
                self.agent_configs = yaml.safe_load(f) or {}
            logger.info("Loaded agent configs", path=str(config_path))
        else:
            # Default configs
            self.agent_configs = self._default_agent_configs()
            # Save defaults
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                yaml.dump(self.agent_configs, f, default_flow_style=False)
            logger.info("Created default agent configs")
    
    def _default_agent_configs(self) -> dict:
        """Default agent configurations."""
        return {
            "technical": {
                "name": "Technical Analyst",
                "role": "Analyze price action, volume, and technical indicators",
                "goal": "Identify high-probability entry points based on technicals",
                "model": "gpt-4o-mini",
                "temperature": 0.3,
                "enabled": True,
            },
            "fundamental": {
                "name": "Fundamental Analyst",
                "role": "Analyze news, catalysts, and market sentiment",
                "goal": "Detect upcoming catalysts and sentiment shifts",
                "model": "gpt-4o",
                "temperature": 0.5,
                "enabled": True,
            },
            "executive": {
                "name": "Executive Trader",
                "role": "Make final BUY/SELL/HOLD decisions",
                "goal": "Maximize profit while managing risk, target 20-30% daily",
                "model": "gpt-4o",
                "temperature": 0.2,
                "enabled": True,
            },
        }
    
    def save_agent_configs(self):
        """Save agent configs to file."""
        config_path = Path(__file__).parent.parent / "configs" / "agents.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.agent_configs, f, default_flow_style=False)
        logger.info("Saved agent configs")
    
    def log_activity(self, activity: dict):
        """Add activity to log."""
        from datetime import datetime
        activity["timestamp"] = datetime.now().isoformat()
        self.activity_log.insert(0, activity)
        # Keep last 1000 entries
        self.activity_log = self.activity_log[:1000]
        
    async def shutdown(self):
        """Cleanup on shutdown."""
        logger.info("Shutting down app state")
