"""Compare local preset tokenizers interactively."""

from __future__ import annotations

import argparse
from typing import Sequence

from transformers import AutoTokenizer, PreTrainedTokenizer

from llm_neuroscience_tools.presets.model_configs import MODEL_CONFIGS


PRESET_NAMES = ["gemma-27b", "llama-70b", "qwen-32b"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare token IDs from local preset tokenizers.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--model-preset", "-m",
        action="append",
        dest="model_presets",
        choices=sorted(MODEL_CONFIGS),
        help=(
            "Model tokenizer to inspect; repeat to compare several. "
            f"Defaults to: {', '.join(PRESET_NAMES)}."
        ),
    )
    args = parser.parse_args(argv)
    if args.model_presets is None:
        args.model_presets = list(PRESET_NAMES)
    return args


def _load_tokenizer(tokenizer_path: str):
    """Load a tokenizer from a local model path."""
    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )


def load_tokenizers(preset_names: list[str]) -> dict[str, PreTrainedTokenizer]:
    """Load one tokenizer per preset name."""
    tokenizers = {}

    for preset_name in preset_names:
        try:
            model_config = MODEL_CONFIGS[preset_name]
        except KeyError as exc:
            raise ValueError(f"Model name {preset_name!r} not recognized.") from exc
        tokenizers[preset_name] = _load_tokenizer(model_config.tokenizer_path)

    return tokenizers


def print_tokenizations(string: str, tokenizers: dict[str, PreTrainedTokenizer]) -> None:
    """Print token IDs for ``string`` under each tokenizer."""
    print(f"String {string!r} tokenizes as:")
    for preset_name, tokenizer in tokenizers.items():
        token_ids = tokenizer.encode(string, add_special_tokens=False)
        print(f"{preset_name}: {token_ids}")
    print()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print('Loading tokenizers...', end=' ')
    tokenizers = load_tokenizers(args.model_presets)
    print('done!')
    print()

    while True:
        try:
            string = input("Enter a string: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        print_tokenizations(string, tokenizers)


if __name__ == "__main__":
    main()
