"""Artifact storage utilities for reproducible runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import json
import uuid

import pandas as pd
import yaml

from src.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class ArtifactPaths:
    """Paths for a single run's artifacts."""

    root: Path
    data_dir: Path
    features_dir: Path
    agents_dir: Path
    trades_dir: Path
    reports_dir: Path
    results_dir: Path


class ArtifactStore:
    """Creates and writes structured run artifacts."""

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)

    def create_run(
        self,
        run_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactPaths:
        """Create a new run folder with standard subdirectories."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{run_type}_{timestamp}_{uuid.uuid4().hex[:8]}"
        root = self.base_dir / run_id

        paths = ArtifactPaths(
            root=root,
            data_dir=root / "data",
            features_dir=root / "features",
            agents_dir=root / "agents",
            trades_dir=root / "trades",
            reports_dir=root / "reports",
            results_dir=root / "results",
        )

        for path in paths.__dict__.values():
            path.mkdir(parents=True, exist_ok=True)

        if metadata:
            self.write_json(metadata, root / "metadata.json")

        logger.info(f"Created artifact run directory: {run_id} -> {root}")
        return paths

    def write_yaml(self, data: dict[str, Any], path: Path) -> None:
        """Write a YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)

    def write_json(self, data: dict[str, Any], path: Path) -> None:
        """Write a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(data, handle, indent=2, default=_json_default)

    def write_jsonl(self, items: Iterable[dict[str, Any]], path: Path) -> None:
        """Write JSONL lines."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            for item in items:
                handle.write(json.dumps(item, default=_json_default))
                handle.write("\n")

    def write_dataframe(
        self,
        df: pd.DataFrame,
        path: Path,
        prefer_parquet: bool = True,
    ) -> Path:
        """Write a dataframe to disk with parquet fallback."""
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix:
            return self._write_dataframe_with_suffix(df, path)

        parquet_path = path.with_suffix(".parquet")
        if prefer_parquet:
            try:
                df.to_parquet(parquet_path, index=False)
                return parquet_path
            except Exception as exc:
                logger.warning(
                    f"Parquet write failed, falling back to CSV: {parquet_path} ({exc})"
                )

        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path

    def _write_dataframe_with_suffix(self, df: pd.DataFrame, path: Path) -> Path:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            try:
                df.to_parquet(path, index=False)
                return path
            except Exception as exc:
                logger.warning(
                    f"Parquet write failed, falling back to CSV: {path} ({exc})"
                )
                csv_path = path.with_suffix(".csv")
                df.to_csv(csv_path, index=False)
                return csv_path

        if suffix == ".csv":
            df.to_csv(path, index=False)
            return path

        df.to_csv(path, index=False)
        return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)
