"""CLI for downloading Hugging Face model snapshots."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any, Sequence

from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from llm_neuroscience_tools.model_paths import (
    default_local_name,
    get_models_dir,
    normalize_repo_id,
)

_SNAPSHOT_DOWNLOAD_PARAMS = set(inspect.signature(snapshot_download).parameters)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the model download CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Download a Hugging Face model snapshot into a local directory."
        ),
        allow_abbrev=False,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model",
        help="Hugging Face repo ID, e.g. 'meta-llama/Llama-3.1-8B-Instruct'.",
    )
    source.add_argument(
        "--model-preset",
        "-m",
        "--preset",
        help="Package model preset, e.g. 'qwen-0p6b'.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help=(
            "Parent directory under which the model snapshot will be stored. "
            "Defaults to LNT_MODELS_DIR. "
            "By default the final folder name is derived from the repo ID."
        ),
    )
    parser.add_argument(
        "--revision",
        help="Optional branch, tag, pull-request ref, or full commit hash.",
    )
    parser.add_argument(
        "--local-name",
        help=(
            "Optional override for the final model directory name. "
            "Preset loading only discovers the canonical owner__repository name."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Preview the resolved download target without downloading files. "
            "If the installed huggingface_hub supports detailed dry runs, "
            "remote file metadata is shown as well."
        ),
    )
    return parser.parse_args(argv)


def validate_local_name(local_name: str) -> str:
    """Ensure the local directory override is a single path component."""
    normalized = local_name.strip()
    if not normalized:
        raise ValueError("local_name must be non-empty if provided")

    parts = Path(normalized).parts
    if len(parts) != 1 or normalized in {".", ".."}:
        raise ValueError(
            "local_name must be a single directory name, not a nested path"
        )
    return normalized


def normalize_output_dir(output_dir: str | Path | None) -> Path:
    """Return a validated output root path."""
    return get_models_dir(output_dir)


def resolve_target_dir(
    output_dir: str | Path | None,
    repo_id: str,
    local_name: str | None = None,
) -> Path:
    """Resolve the final on-disk path for a downloaded snapshot."""
    output_root = normalize_output_dir(output_dir)
    directory_name = (
        validate_local_name(local_name)
        if local_name is not None
        else default_local_name(repo_id)
    )
    return output_root / directory_name


def _snapshot_supports_parameter(name: str) -> bool:
    """Return whether the installed snapshot_download supports a parameter."""
    return name in _SNAPSHOT_DOWNLOAD_PARAMS


def build_snapshot_download_kwargs(
    repo_id: str,
    target_dir: Path,
    revision: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Construct kwargs for snapshot_download in a version-compatible way."""
    kwargs: dict[str, Any] = {
        "repo_id": normalize_repo_id(repo_id),
        "repo_type": "model",
        "local_dir": str(target_dir),
    }
    if revision is not None:
        kwargs["revision"] = revision
    if _snapshot_supports_parameter("resume_download"):
        kwargs["resume_download"] = True
    if dry_run and _snapshot_supports_parameter("dry_run"):
        kwargs["dry_run"] = True
    return kwargs


def run_snapshot_download(
    repo_id: str,
    target_dir: Path,
    revision: str | None = None,
) -> str:
    """Download a model snapshot and return the path from snapshot_download."""
    kwargs = build_snapshot_download_kwargs(
        repo_id=repo_id,
        target_dir=target_dir,
        revision=revision,
    )
    return snapshot_download(**kwargs)


def print_basic_dry_run_preview(
    repo_id: str,
    target_dir: Path,
    revision: str | None = None,
) -> None:
    """Print a lightweight preview when detailed dry-run support is absent."""
    print("Dry run mode: no files will be downloaded.")
    print(f"Model repo: {repo_id}")
    print(f"Target directory: {target_dir}")
    print(f"Revision: {revision or 'main (default)'}")
    print(
        "Installed huggingface_hub does not expose detailed dry-run metadata "
        "in this environment, so the preview stops here."
    )


def run_dry_run_preview(
    repo_id: str,
    target_dir: Path,
    revision: str | None = None,
) -> None:
    """Preview a model download without writing files."""
    if not _snapshot_supports_parameter("dry_run"):
        print_basic_dry_run_preview(
            repo_id=repo_id,
            target_dir=target_dir,
            revision=revision,
        )
        return

    kwargs = build_snapshot_download_kwargs(
        repo_id=repo_id,
        target_dir=target_dir,
        revision=revision,
        dry_run=True,
    )
    file_infos = snapshot_download(**kwargs)

    print("Dry run mode: no files will be downloaded.")
    print(f"Model repo: {repo_id}")
    print(f"Target directory: {target_dir}")
    print(f"Revision: {revision or 'main (default)'}")
    print(f"Hub reported {len(file_infos)} file entries:")
    for file_info in file_infos:
        print(f"  {file_info}")


def format_download_error(
    exc: Exception,
    repo_id: str,
    revision: str | None = None,
) -> str:
    """Translate common hub exceptions into clearer CLI messages."""
    if isinstance(exc, GatedRepoError):
        return (
            f"Repository {repo_id!r} is gated. Authenticate with "
            "`hf auth login` or set `HF_TOKEN`, then retry."
        )
    if isinstance(exc, RepositoryNotFoundError):
        return (
            f"Repository {repo_id!r} was not found or is not accessible with "
            "the current credentials."
        )
    if isinstance(exc, RevisionNotFoundError):
        return (
            f"Revision {revision!r} was not found for repository {repo_id!r}."
        )
    if isinstance(exc, HfHubHTTPError):
        return f"Hugging Face Hub request failed: {exc}"
    return str(exc)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the download-model CLI."""
    args = parse_args(argv)
    repo_id = args.model or args.model_preset

    try:
        if args.model_preset is not None:
            from llm_neuroscience_tools.presets.model_configs import MODEL_CONFIGS

            if args.model_preset not in MODEL_CONFIGS:
                raise ValueError(
                    f"Unknown model preset {args.model_preset!r}. "
                    f"Available presets: {', '.join(sorted(MODEL_CONFIGS))}"
                )
            repo_id = MODEL_CONFIGS[args.model_preset].repo_id
        repo_id = normalize_repo_id(repo_id)
        output_dir = normalize_output_dir(args.output_dir)
        target_dir = resolve_target_dir(
            output_dir=output_dir,
            repo_id=repo_id,
            local_name=args.local_name,
        )

        if args.dry_run:
            run_dry_run_preview(
                repo_id=repo_id,
                target_dir=target_dir,
                revision=args.revision,
            )
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {repo_id} -> {target_dir}")
        if args.revision is not None:
            print(f"Using revision: {args.revision}")

        snapshot_path = run_snapshot_download(
            repo_id=repo_id,
            target_dir=target_dir,
            revision=args.revision,
        )

        print(f"Download complete: {target_dir}")
        if snapshot_path != str(target_dir):
            print(f"snapshot_download returned: {snapshot_path}")
        return 0

    except (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
        ValueError,
        OSError,
    ) as exc:
        print(
            "Error:",
            format_download_error(exc, repo_id=repo_id, revision=args.revision),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
