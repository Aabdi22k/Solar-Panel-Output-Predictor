"""Path helpers for project directories.

This module centralizes filesystem paths for artifacts and datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ProjectPaths:
    """Stores common filesystem paths used throughout the project.

    Attributes:
        root: Repository root directory.
        artifacts_dir: Root directory for generated artifacts.
        models_dir: Directory for trained model artifacts.
        forecasts_dir: Directory for generated forecasts.
        raw_data_dir: Directory for raw ingested data.
        processed_data_dir: Directory for processed/cleaned datasets.
        history_dir: Directory for persisted prediction history JSON files.
    """

    root: Path
    artifacts_dir: Path
    models_dir: Path
    forecasts_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    history_dir: Path

    @staticmethod
    def from_repo_root(repo_root: Path) -> "ProjectPaths":
        """Build standard project paths relative to the repo root.

        Args:
            repo_root: Path to the repository root.

        Returns:
            A ProjectPaths instance with standard subdirectories set.
        """

        artifacts = repo_root / "artifacts"
        data_dir = repo_root / "data"
        raw_data_dir = data_dir / "raw"

        return ProjectPaths(
            root=repo_root,
            artifacts_dir=artifacts,
            models_dir=artifacts / "models",
            forecasts_dir=artifacts / "forecasts",
            raw_data_dir=raw_data_dir,
            processed_data_dir=data_dir / "processed",
            history_dir=raw_data_dir / "history",
        )

    def ensure_dirs(self) -> None:
        """Create the expected project directories if missing."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.forecasts_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def history_file(self, tag: str) -> Path:
        """Return the JSON history file path for a model/location tag.

        Args:
            tag: Stable model/location tag, e.g. "33p448376_-112p074036".

        Returns:
            Path to the corresponding history JSON file.
        """
        safe_tag = tag.strip().replace("/", "_").replace("\\", "_")
        return self.history_dir / f"history_{safe_tag}.json"