"""Chat interactively with a local model preset."""

from __future__ import annotations

import argparse
import math
from threading import Thread
from typing import Sequence

import torch
from transformers import TextIteratorStreamer

from llm_neuroscience_tools.presets.model_configs import MODEL_CONFIGS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat interactively with a locally available model preset.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "legacy_model_preset",
        nargs="?",
        metavar="MODEL_PRESET",
        help="Legacy positional alternative to --model-preset.",
    )
    parser.add_argument(
        "--model-preset", "-m", choices=sorted(MODEL_CONFIGS),
        help="Model preset to load.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1280,
        help="Maximum number of tokens generated per response (default: 1280).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature; use 0 for greedy decoding (default: 0.7).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus-sampling threshold (default: 0.9).",
    )
    args = parser.parse_args(argv)
    if args.model_preset is not None and args.legacy_model_preset is not None:
        parser.error("Use --model-preset or the positional model preset, not both.")
    args.model_preset = args.model_preset or args.legacy_model_preset
    del args.legacy_model_preset
    if args.model_preset is None:
        parser.error("--model-preset is required.")
    if args.model_preset not in MODEL_CONFIGS:
        parser.error(f"Unknown model preset {args.model_preset!r}.")
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("--temperature must be finite and non-negative.")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in the interval (0, 1].")


def generate_response(backend, messages: list[dict[str, str]], args) -> str:
    tokenizer = backend.tokenizer
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    device = backend.model.get_input_embeddings().weight.device
    model_inputs = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs |= {
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
    else:
        generation_kwargs["do_sample"] = False

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generation_errors: list[BaseException] = []

    def run_generation() -> None:
        try:
            with torch.inference_mode():
                backend.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    streamer=streamer,
                    **generation_kwargs,
                )
        except BaseException as exc:
            generation_errors.append(exc)
            streamer.end()

    generation_thread = Thread(target=run_generation)
    generation_thread.start()

    response_parts = []
    print("\nAssistant: ", end="", flush=True)
    for text in streamer:
        response_parts.append(text)
        print(text, end="", flush=True)
    print()

    generation_thread.join()
    if generation_errors:
        raise generation_errors[0]
    return "".join(response_parts).strip()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model_config = MODEL_CONFIGS[args.model_preset]

    print(f"Loading {args.model_preset}...", flush=True)
    backend = model_config.backend_cls.from_model_path(
        model_config.model_path,
        **model_config.backend_kwargs,
    )
    backend.model.eval()
    print("Ready. Enter /clear to reset the conversation or /quit to exit.")

    messages: list[dict[str, str]] = []
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_text in {"/quit", "/exit"}:
            break
        if user_text == "/clear":
            messages.clear()
            print("Conversation cleared.")
            continue
        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})
        try:
            response = generate_response(backend, messages, args)
        except Exception:
            messages.pop()
            raise
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
