"""Export the two token positions used by the paper's PCA figure.

Usage: python tools/export_pca_inputs.py RECORDING_DIRECTORY NEW_OUTPUT_DIRECTORY
The destination must not exist. Source recordings are never modified.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def export_pca_inputs(source: Path, destination: Path) -> None:
    with (source / "completed.json").open() as handle:
        json.load(handle)
    with (source / "recordings_metadata.json").open() as handle:
        original = json.load(handle)
    tokens = ("R_BC", "C1")
    indices = [original["context_landmarks"][token] for token in tokens]
    activations = np.load(source / "Z_gstla.npy", mmap_mode="r", allow_pickle=False)
    shape = list(activations.shape)
    shape[2] = len(tokens)

    destination.mkdir(parents=True, exist_ok=False)
    subset = np.lib.format.open_memmap(
        destination / "Z_gstla.npy", mode="w+", dtype=activations.dtype, shape=tuple(shape)
    )
    # Copy and check one graph/seed slice at a time to bound memory use.
    for graph in range(shape[0]):
        for seed in range(shape[1]):
            for token, source_index in enumerate(indices):
                subset[graph, seed, token] = activations[graph, seed, source_index]
    subset.flush()
    del subset
    subset = np.load(destination / "Z_gstla.npy", mmap_mode="r", allow_pickle=False)
    for graph in range(shape[0]):
        for seed in range(shape[1]):
            for token, source_index in enumerate(indices):
                if subset[graph, seed, token].tobytes() != activations[graph, seed, source_index].tobytes():
                    raise ValueError("Exported activations differ from the source.")

    metadata = {
        key: original[key]
        for key in (
            "activation_axes", "activation_dtype", "model_preset", "prompt_preset",
            "model_shape", "n_trials", "start_seed", "trial_seeds", "relation_patterns",
        )
    }
    metadata.update(
        activation_shape=shape,
        activation_path="Z_gstla.npy",
        context_landmarks={token: index for index, token in enumerate(tokens)},
        subset={
            "purpose": "PCA figure inputs; contains selected tokens and no KV caches",
            "source_directory": str(source.resolve()),
            "source_activation_shape": list(activations.shape),
            "source_token_indices": dict(zip(tokens, indices)),
            "source_completion": json.loads((source / "completed.json").read_text()),
        },
    )
    with (destination / "recordings_metadata.json").open("x") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    with (destination / "completed.json").open("x") as handle:
        json.dump({
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "operation": "export_pca_inputs",
            "verified_against_source": True,
        }, handle, indent=2)
        handle.write("\n")
    print(f"Exported and verified {destination} ({subset.nbytes / 2**30:.2f} GiB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export_pca_inputs(args.source, args.destination)
