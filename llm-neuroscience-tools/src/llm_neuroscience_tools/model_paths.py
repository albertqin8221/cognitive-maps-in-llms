"""Resolve local model directories shared by presets and download tools."""

from __future__ import annotations

import os
from pathlib import Path
import re


MODELS_DIR_ENV = "LNT_MODELS_DIR"


def normalize_repo_id(repo_id: str) -> str:
    """Validate the owner/repository form used for model folder names."""
    normalized = repo_id.strip()
    parts = normalized.split("/")
    if len(parts) != 2 or any(
        part in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", part)
        for part in parts
    ):
        raise ValueError(
            f"model repo ID must look like 'owner/repo', got {repo_id!r}"
        )
    return normalized


def default_local_name(repo_id: str) -> str:
    """Map a model repository ID to its canonical local folder name."""
    return normalize_repo_id(repo_id).replace("/", "__")


def get_models_dir(models_dir: str | Path | None = None) -> Path:
    """Read the model root, allowing an explicit download destination override."""
    if models_dir is None:
        models_dir = os.environ.get(MODELS_DIR_ENV)
        if models_dir is None or not models_dir.strip():
            raise ValueError(
                f"Set {MODELS_DIR_ENV} to your model directory, for example: "
                f"export {MODELS_DIR_ENV}=/path/to/models"
            )
    if isinstance(models_dir, str) and not models_dir.strip():
        raise ValueError("model directory must be non-empty")
    root = Path(models_dir).expanduser().absolute()
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"Model directory is not a directory: {root}")
    return root


def get_model_path(
    repo_id: str,
    *,
    models_dir: str | Path | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve a canonical folder; downloads may target a new directory."""
    folder_name = default_local_name(repo_id)
    path = get_models_dir(models_dir) / folder_name
    if must_exist and not path.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {path}. "
            f"Check {MODELS_DIR_ENV}, or from the repository root run: "
            f"python tools/download_model.py --model {normalize_repo_id(repo_id)}"
        )
    return path
