"""List the available model and prompt presets without loading models."""

from __future__ import annotations

import argparse
from typing import Sequence

from llm_neuroscience_tools.presets.model_configs import MODEL_CONFIGS
from llm_neuroscience_tools.presets.prompt_configs import PROMPT_CONFIGS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List available model and prompt presets.",
        allow_abbrev=False,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    parse_args(argv)
    print(f"Model presets ({len(MODEL_CONFIGS)})")
    for name in sorted(MODEL_CONFIGS):
        print(f"  {name}")

    print()
    print(f"Prompt presets ({len(PROMPT_CONFIGS)})")
    for name in sorted(PROMPT_CONFIGS):
        print(f"  {name}")


if __name__ == "__main__":
    main()
