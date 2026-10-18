"""Build a tokenizer-compatible female name pool."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from transformers import AutoTokenizer

from llm_neuroscience_tools.presets.model_configs import MODEL_CONFIGS


DEFAULT_SOURCE_PATH = Path(__file__).with_name("SSA_female_names_2000.txt")


# ------ Helper functions to load tokenizers ------


def _load_tokenizer(tokenizer_path: str):
    """Load a tokenizer from a local model path."""
    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )


def _get_tokenizer_path_from_preset(model_name: str) -> str:
    """Retrieve the tokenizer path for a model preset."""
    try:
        model_config = MODEL_CONFIGS[model_name]
    except KeyError as exc:
        raise ValueError(f"Model name {model_name!r} not recognized.") from exc
    return model_config.tokenizer_path


# ------ Helper functions to load candidate names ------


def load_female_names(source_path: Path = DEFAULT_SOURCE_PATH) -> list[str]:
    """Load the ordered female-name list from the SSA source file."""
    names: list[str] = []
    for line in source_path.read_text().splitlines():
        if not line.strip():
            continue
        name, _sex, _count = line.split(",", maxsplit=2)
        names.append(name)
    return names


# ------ Helper functions to filter for tokenizer compatibility ------


def tokenizes_as_single_token(tokenizer, string: str) -> bool:
    """Return whether both ``string`` and ``" " + string`` are single tokens."""
    return (
        len(tokenizer.encode(string, add_special_tokens=False)) == 1
        and len(tokenizer.encode(" " + string, add_special_tokens=False)) == 1
    )


def get_first_n_compatible_female_names(
    tokenizer,
    n: int,
    source_path: Path = DEFAULT_SOURCE_PATH,
) -> list[str]:
    """Return the first ``n`` source names that satisfy the tokenizer check."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return []

    matching_names: list[str] = []
    for name in load_female_names(source_path):
        if tokenizes_as_single_token(tokenizer, name):
            matching_names.append(name)
            if len(matching_names) == n:
                return matching_names

    return matching_names


def get_first_n_compatible_female_names_for_preset(
    model_name: str,
    n: int,
    source_path: Path = DEFAULT_SOURCE_PATH,
) -> list[str]:
    """Load a preset tokenizer and return the first ``n`` compatible names."""
    tokenizer_path = _get_tokenizer_path_from_preset(model_name)
    tokenizer = _load_tokenizer(tokenizer_path)
    return get_first_n_compatible_female_names(tokenizer, n, source_path)


# ------------------------------------------------------------------------------


def _write_names(output_path: Path, names: list[str]) -> None:
    """Write one name per line to ``output_path``."""
    output_path.write_text("\n".join(names) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a female name pool from the SSA source list by filtering for "
            "tokenizer single-token compatibility."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "-n",
        "--num-names",
        type=int,
        required=True,
        help="Number of names to generate.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        required=True,
        help="Path to the output .txt file.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "-t",
        "--tokenizer-path",
        help="Local path to a tokenizer.",
    )
    source_group.add_argument(
        "--model-preset",
        "-m",
        "-p",
        "--preset-name",
        choices=sorted(MODEL_CONFIGS),
        help="Model preset name from MODEL_CONFIGS.",
    )
    args = parser.parse_args(argv)
    if args.num_names <= 0:
        parser.error("--num-names must be positive.")
    if Path(args.output_path).suffix != ".txt":
        parser.error("--output-path must end with .txt.")
    if not Path(args.output_path).parent.is_dir():
        parser.error("--output-path parent directory must already exist.")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = Path(args.output_path)

    if args.tokenizer_path is not None:
        tokenizer = _load_tokenizer(args.tokenizer_path)
        print("Selecting compatible strings...")
        names = get_first_n_compatible_female_names(
            tokenizer=tokenizer,
            n=args.num_names,
        )
    else:
        print("Selecting compatible strings...")
        names = get_first_n_compatible_female_names_for_preset(
            model_name=args.model_preset,
            n=args.num_names,
        )

    if len(names) < args.num_names:
        raise ValueError(
            f"Only found {len(names)} compatible strings, fewer than requested "
            f"{args.num_names}."
        )

    _write_names(output_path, names)
    print(f"Wrote {len(names)} names to {output_path}")


if __name__ == "__main__":
    main()
