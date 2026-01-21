"""MLflow integration for experiment tracking.

Provides tracking and logging of:
- Model experiments
- Agent decisions
- Backtest results
- Hyperparameters
"""

from datetime import datetime
from pathlib import Path
from typing import Any
import json

from src.utils.logging import get_logger


logger = get_logger(__name__)


class MLflowTracker:
    """MLflow experiment tracker.

    Tracks:
    - Backtest runs with metrics
    - Model training experiments
    - Agent decision logs
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "trading_system",
    ):
        """Initialize MLflow tracker.

        Args:
            tracking_uri: MLflow tracking server URI.
            experiment_name: Experiment name.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._mlflow = None
        self._experiment_id = None
        self._run_id = None

    def _get_mlflow(self):
        """Get or import MLflow."""
        if self._mlflow is None:
            try:
                import mlflow
                mlflow.set_tracking_uri(self.tracking_uri)
                self._mlflow = mlflow
            except ImportError:
                logger.error("mlflow package not installed")
                raise
        return self._mlflow

    def start_experiment(self, experiment_name: str | None = None) -> str:
        """Start or get experiment.

        Args:
            experiment_name: Optional experiment name override.

        Returns:
            Experiment ID.
        """
        mlflow = self._get_mlflow()
        name = experiment_name or self.experiment_name

        experiment = mlflow.get_experiment_by_name(name)
        if experiment:
            self._experiment_id = experiment.experiment_id
        else:
            self._experiment_id = mlflow.create_experiment(name)

        mlflow.set_experiment(name)
        logger.info(f"MLflow experiment: {name} (id={self._experiment_id})")
        return self._experiment_id

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Start a new run.

        Args:
            run_name: Run name.
            tags: Run tags.

        Returns:
            Run ID.
        """
        mlflow = self._get_mlflow()

        if self._experiment_id is None:
            self.start_experiment()

        run = mlflow.start_run(run_name=run_name, tags=tags)
        self._run_id = run.info.run_id
        logger.info(f"Started MLflow run: {self._run_id}")
        return self._run_id

    def end_run(self, status: str = "FINISHED"):
        """End current run.

        Args:
            status: Run status.
        """
        mlflow = self._get_mlflow()
        mlflow.end_run(status=status)
        logger.info(f"Ended MLflow run: {self._run_id}")
        self._run_id = None

    def log_params(self, params: dict[str, Any]):
        """Log parameters.

        Args:
            params: Parameter dictionary.
        """
        mlflow = self._get_mlflow()
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Failed to log param {key}: {e}")

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ):
        """Log metrics.

        Args:
            metrics: Metrics dictionary.
            step: Step number.
        """
        mlflow = self._get_mlflow()
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metric {key}: {e}")

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None):
        """Log artifact file.

        Args:
            local_path: Local file path.
            artifact_path: Artifact subdirectory.
        """
        mlflow = self._get_mlflow()
        mlflow.log_artifact(str(local_path), artifact_path)

    def log_dict(self, data: dict, filename: str):
        """Log dictionary as JSON artifact.

        Args:
            data: Dictionary to log.
            filename: Artifact filename.
        """
        mlflow = self._get_mlflow()
        mlflow.log_dict(data, filename)

    def log_backtest(
        self,
        config: dict[str, Any],
        metrics: dict[str, float],
        equity_curve_path: str | Path | None = None,
    ):
        """Log a complete backtest run.

        Args:
            config: Backtest configuration.
            metrics: Backtest metrics.
            equity_curve_path: Path to equity curve file.
        """
        self.start_run(
            run_name=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={"type": "backtest"},
        )

        try:
            # Log config as params
            flat_config = self._flatten_dict(config)
            self.log_params(flat_config)

            # Log metrics
            self.log_metrics(metrics)

            # Log equity curve
            if equity_curve_path:
                self.log_artifact(equity_curve_path, "results")

            logger.info(f"Logged backtest: return={metrics.get('total_return', 0):.2%}")

        finally:
            self.end_run()

    def log_decision(
        self,
        decision: dict[str, Any],
        step: int,
        symbol: str,
    ):
        """Log a trading decision.

        Args:
            decision: Decision dictionary.
            step: Step number.
            symbol: Symbol.
        """
        mlflow = self._get_mlflow()

        # Log key metrics
        self.log_metrics({
            f"{symbol}_action": 1 if decision.get("action") == "BUY" else -1 if decision.get("action") == "SELL" else 0,
            f"{symbol}_confidence": decision.get("confidence", 0),
            f"{symbol}_size_pct": decision.get("size_pct", 0),
        }, step=step)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_name: str | None = None,
    ):
        """Log a model.

        Args:
            model: Model to log.
            artifact_path: Artifact path.
            registered_name: Optional registered model name.
        """
        mlflow = self._get_mlflow()

        # Determine model flavor
        try:
            import sklearn
            if hasattr(model, "predict"):
                mlflow.sklearn.log_model(model, artifact_path, registered_model_name=registered_name)
                return
        except ImportError:
            pass

        try:
            import torch
            if isinstance(model, torch.nn.Module):
                mlflow.pytorch.log_model(model, artifact_path, registered_model_name=registered_name)
                return
        except ImportError:
            pass

        # Fallback to pickle
        logger.warning("Using pickle for model logging")
        import pickle
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(model, f)
            self.log_artifact(f.name, artifact_path)

    def _flatten_dict(
        self,
        d: dict,
        parent_key: str = "",
        sep: str = ".",
    ) -> dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class DecisionLogger:
    """Logs all agent decisions for auditing."""

    def __init__(
        self,
        output_dir: Path,
        mlflow_tracker: MLflowTracker | None = None,
    ):
        """Initialize decision logger.

        Args:
            output_dir: Output directory for logs.
            mlflow_tracker: Optional MLflow tracker.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_tracker = mlflow_tracker

        self._log_file = self.output_dir / "decisions.jsonl"

    def log(
        self,
        decision: dict[str, Any],
        state: dict[str, Any] | None = None,
        agent_outputs: dict[str, Any] | None = None,
    ):
        """Log a decision.

        Args:
            decision: Decision dictionary.
            state: Pipeline state.
            agent_outputs: Individual agent outputs.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "state": state,
            "agent_outputs": agent_outputs,
        }

        # Append to JSONL file
        with open(self._log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Log to MLflow if available
        if self.mlflow_tracker and self.mlflow_tracker._run_id:
            step = len(open(self._log_file).readlines()) - 1
            self.mlflow_tracker.log_decision(
                decision,
                step=step,
                symbol=decision.get("symbol", "UNKNOWN"),
            )

    def get_history(self, limit: int = 100) -> list[dict]:
        """Get decision history.

        Args:
            limit: Maximum entries to return.

        Returns:
            List of decision entries.
        """
        if not self._log_file.exists():
            return []

        entries = []
        with open(self._log_file, "r") as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return entries[-limit:]
