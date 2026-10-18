"""Build a tokenizer-compatible places name pool."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from transformers import AutoTokenizer

from llm_neuroscience_tools.presets.model_configs import MODEL_CONFIGS


PLACES = [
    "library",
    "church",
    "store",
    "supermarket",
    "market",
    "bakery",
    "pharmacy",
    "florist",
    "restaurant",
    "cafe",
    "diner",
    "bar",
    "bank",
    "hospital",
    "clinic",
    "school",
    "university",
    "gym",
    "stadium",
    "theater",
    "cinema",
    "museum",
    "gallery",
    "hotel",
    "motel",
    "airport",
    "station",
    "subway",
    "garage",
    "parking",
    "office",
    "factory",
    "warehouse",
    "courthouse",
    "embassy",
    "prison",
    "apartment",
    "house",
    "kitchen",
    "bedroom",
    "bathroom",
    "basement",
    "attic",
    "hallway",
    "balcony",
    "lobby",
    "cafeteria",
    "laundromat",
    "salon",
    "arcade",
    "aquarium",
    "zoo",
    "campus",
    "classroom",
    "laboratory",
    "studio",
    "kindergarten",
    "daycare",
    "nursery",
    "bookstore",
    "boutique",
    "mall",
    "hostel",
    "resort",
    "spa",
    "chapel",
    "mosque",
    "synagogue",
    "temple",
    "cathedral",
    "brewery",
    "dealership",
    "terminal",
    "runway",
    "dock",
    "depot",
    "junction",
    "avenue",
    "plaza",
    "square",
    "alley",
    "corridor",
    "auditorium",
    "canteen",
    "arena",
    "pool",
    "rink",
    "court",
    "field",
    "tower",
    "skyscraper",
    "cottage",
    "mansion",
    "villa",
    "archive",
    "shelter",
    "cemetery",
    "kiosk",
    "stable",
    "greenhouse",
    "hangar",
    "castle",
    "palace",
    "courtyard",
    "foyer",
]


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


# ------ Helper functions to filter for tokenizer compatibility ------


def tokenizes_as_single_token(tokenizer, string: str) -> bool:
    """Return whether both ``string`` and ``" " + string`` are single tokens."""
    return (
        len(tokenizer.encode(string, add_special_tokens=False)) == 1
        and len(tokenizer.encode(" " + string, add_special_tokens=False)) == 1
    )


def get_compatible_places(tokenizer) -> list[str]:
    """Return the hard-coded places that satisfy the tokenizer check."""
    return [place for place in PLACES if tokenizes_as_single_token(tokenizer, place)]


def get_compatible_places_for_preset(model_name: str) -> list[str]:
    """Load a preset tokenizer and return the compatible places."""
    tokenizer_path = _get_tokenizer_path_from_preset(model_name)
    tokenizer = _load_tokenizer(tokenizer_path)
    return get_compatible_places(tokenizer)


# ------------------------------------------------------------------------------


def _write_names(output_path: Path, names: list[str]) -> None:
    """Write one name per line to ``output_path``."""
    output_path.write_text("\n".join(names) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a place name pool from a fixed list by filtering for "
            "tokenizer single-token compatibility."
        ),
        allow_abbrev=False,
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
        names = get_compatible_places(tokenizer)
    else:
        print("Selecting compatible strings...")
        names = get_compatible_places_for_preset(args.model_preset)

    _write_names(output_path, names)
    print(f"Wrote {len(names)} names to {output_path}")


if __name__ == "__main__":
    main()
