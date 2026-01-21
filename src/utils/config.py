"""Configuration management for AI Trading System."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# Load environment variables
load_dotenv()


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "openai"
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.1
    max_tokens: int = 2000


class DatabaseConfig(BaseModel):
    """Database connection configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "trading"
    user: str = "trading"
    password: str = ""

    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class RiskConfig(BaseModel):
    """Risk guardian configuration."""

    max_position_pct: float = 0.20
    max_daily_loss_pct: float = 0.05
    max_trades_per_day: int = 10
    min_confidence: float = 0.5


class BacktestConfig(BaseModel):
    """Backtesting configuration."""

    initial_capital: float = 100000.0
    commission_type: str = "per_share"
    commission_value: float = 0.005
    slippage_type: str = "bps"
    slippage_value: float = 5.0
    margin: float = 1.0


class ExecutionConfig(BaseModel):
    """Execution configuration."""

    mode: str = "paper"  # backtest, paper, live_hil, live_auto
    broker: str = "alpaca"


class Settings(BaseSettings):
    """Main application settings."""

    # App metadata
    app_name: str = "ai-trading-system"
    app_version: str = "0.1.0"
    env: str = "development"
    log_level: str = "INFO"

    # API Keys (from environment)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    perplexity_api_key: str = Field(default="", alias="PERPLEXITY_API_KEY")
    alpaca_api_key: str = Field(default="", alias="ALPACA_API_KEY")
    alpaca_api_secret: str = Field(default="", alias="ALPACA_API_SECRET")

    # Database passwords
    timescale_password: str = Field(default="", alias="TIMESCALE_PASSWORD")
    postgres_password: str = Field(default="", alias="POSTGRES_PASSWORD")

    # Notification channels
    slack_webhook_url: str = Field(default="", alias="SLACK_WEBHOOK_URL")
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    # Nested configs (loaded from YAML)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    # Data settings
    default_symbols: list[str] = Field(default_factory=lambda: ["AAPL", "GOOGL", "MSFT"])
    default_interval: str = "1h"
    market_data_provider: str = "yfinance"

    # Paths
    artifacts_dir: Path = Field(default=Path("./artifacts"))
    data_dir: Path = Field(default=Path("./data"))
    models_dir: Path = Field(default=Path("./models"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary with configuration values.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Expand environment variables in string values
    def expand_env_vars(obj: Any) -> Any:
        if isinstance(obj, str):
            if obj.startswith("${") and obj.endswith("}"):
                env_var = obj[2:-1]
                return os.environ.get(env_var, "")
            return obj
        elif isinstance(obj, dict):
            return {k: expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand_env_vars(item) for item in obj]
        return obj

    return expand_env_vars(config)


@lru_cache
def get_settings(config_path: str | None = None) -> Settings:
    """Get application settings.

    Loads settings from environment variables and optionally from a YAML config file.

    Args:
        config_path: Optional path to YAML configuration file.

    Returns:
        Settings instance.
    """
    settings_dict = {}

    # Load from YAML if provided
    if config_path:
        yaml_config = load_yaml_config(config_path)
        settings_dict = _flatten_config(yaml_config)

    return Settings(**settings_dict)


def _flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested configuration dictionary."""
    result = {}
    for key, value in config.items():
        new_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict) and not _is_nested_model_key(key):
            result.update(_flatten_config(value, new_key))
        else:
            result[new_key] = value
    return result


def _is_nested_model_key(key: str) -> bool:
    """Check if key corresponds to a nested Pydantic model."""
    return key in {"llm", "risk", "backtest", "execution", "timescaledb", "postgres"}
