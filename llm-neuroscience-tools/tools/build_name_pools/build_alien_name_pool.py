"""Build a tokenizer-compatible alien name pool."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable, Sequence

from transformers import AutoTokenizer

from llm_neuroscience_tools.presets.model_configs import MODEL_CONFIGS


# ------ Helper functions to load tokenizers ------


def _load_tokenizer(tokenizer_path: str):
    """Load a tokenizer from a local model path."""
    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )

def _get_tokenizer_path_from_preset(model_name: str) -> str:
    """Retrieves the tokenizer path for a model preset."""
    try:
        model_config = MODEL_CONFIGS[model_name]
    except KeyError as exc:
        raise ValueError(f"Model name {model_name!r} not recognized.") from exc
    return model_config.tokenizer_path


# ------ Helper functions to compute word frequencies ------


DEFAULT_LANGUAGES = (
    "en",
    "es",
    "fr",
    "de",
    "pt",
    "it",
    "nl",
    "ru",
    "tr",
    "pl",
)
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def iter_three_letter_strings(alphabet: str = ALPHABET) -> Iterable[str]:
    """Yield all 3-letter strings over ``alphabet``."""
    for chars in itertools.product(alphabet, repeat=3):
        yield "".join(chars)


def compute_scores(languages: tuple[str, ...]) -> list[dict[str, float | str]]:
    """Compute per-language and mean probabilities for each 3-letter string."""
    try:
        from wordfreq import word_frequency
    except ImportError as exc:
        raise ImportError(
            "Building alien name pools requires wordfreq: "
            "python -m pip install wordfreq"
        ) from exc

    results: list[dict[str, float | str]] = []
    for string in iter_three_letter_strings():
        per_language = {
            language: word_frequency(string, language)
            for language in languages
        }
        mean_probability = sum(per_language.values()) / len(languages)
        results.append(
            {
                "string": string,
                "mean_probability": mean_probability,
                **per_language,
            }
        )
    results.sort(
        key=lambda row: (
            float(row["mean_probability"]),
            str(row["string"]),
        )
    )
    return results


# ------ Helper functions to filter for tokenizer compatibility ------


def tokenizes_as_single_token(tokenizer, string: str) -> bool:
    """Return whether both ``string`` and ``" " + string`` are single tokens."""
    return (
        len(tokenizer.encode(string, add_special_tokens=False)) == 1
        and len(tokenizer.encode(" " + string, add_special_tokens=False)) == 1
    )


def get_first_n_low_probability_alien_names(
    tokenizer,
    n: int,
    languages: tuple[str, ...] = DEFAULT_LANGUAGES,
) -> list[str]:
    """Return the first ``n`` 3-letter strings that satisfy the tokenizer check.

    The candidate strings are considered in order of increasing mean wordfreq
    probability across ``languages``.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return []

    matching_strings: list[str] = []
    ranked_strings = compute_scores(languages)
    for row in ranked_strings:
        string = str(row["string"])
        if tokenizes_as_single_token(tokenizer, string):
            matching_strings.append(string)
            if len(matching_strings) == n:
                return matching_strings

    return matching_strings


def get_first_n_low_probability_alien_names_for_preset(
    model_name: str,
    n: int,
    languages: tuple[str, ...] = DEFAULT_LANGUAGES,
) -> list[str]:
    """Load a preset tokenizer and return the first ``n`` compatible strings."""
    tokenizer_path = _get_tokenizer_path_from_preset(model_name)
    tokenizer = _load_tokenizer(tokenizer_path)
    return get_first_n_low_probability_alien_names(tokenizer, n, languages)


# ------------------------------------------------------------------------------


def _write_names(output_path: Path, names: list[str]) -> None:
    """Write one name per line to ``output_path``."""
    output_path.write_text("\n".join(names) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a low-probability alien name pool by ranking lowercase "
            "3-letter strings with wordfreq and filtering for tokenizer "
            "single-token compatibility."
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
        print('Selecting compatible strings...')
        names = get_first_n_low_probability_alien_names(
            tokenizer=tokenizer,
            n=args.num_names,
        )
    else:
        print('Selecting compatible strings...')
        names = get_first_n_low_probability_alien_names_for_preset(
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
