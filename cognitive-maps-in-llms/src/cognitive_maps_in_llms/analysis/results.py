from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cognitive_maps_in_llms.analysis.pca import (
    ActivationRecordings,
    load_activation_recordings as _load_activation_recordings,
)
from cognitive_maps_in_llms.analysis.erasure import (
    ErasureTrajectory,
    load_erasure_trajectory as _load_erasure_trajectory,
)


RESULTS_FILENAME = "results.ndjson"
SEED_AXIS = "seed"


@dataclass(frozen=True)
class Series:
    """One query and one non-seed trial condition, collected across seeds."""

    records: list[dict[str, Any]]
    query: str
    trial_axes: dict[str, Any]
    ndjson_path: Path
    prompt_preset: str | None = None


class RunResults:
    """Locate and load the results and artifacts produced by one run."""

    def __init__(
        self,
        *,
        results_root: Path | str | None = None,
        artifacts_root: Path | str | None = None,
    ) -> None:
        if results_root is None and artifacts_root is None:
            raise ValueError(
                "At least one of results_root or artifacts_root is required."
            )
        self.results_root = _optional_directory(results_root, name="Results root")
        self.artifacts_root = _optional_directory(
            artifacts_root,
            name="Artifacts root",
        )
        self._records_by_path: dict[Path, list[dict[str, Any]]] = {}
        self._trajectories_by_path: dict[Path, ErasureTrajectory] = {}

    def load_activation_recordings(
        self,
        *,
        model_preset: str,
        prompt_preset: str,
        task: str = "activation_and_kv_recording",
    ) -> ActivationRecordings:
        """Memory-map the context activations from a completed recording task."""
        if self.artifacts_root is None:
            raise ValueError("This RunResults object has no artifacts_root.")
        directory = self.artifacts_root / model_preset / prompt_preset / task
        return _load_activation_recordings(directory)

    def load_series(
        self,
        *,
        model_preset: str,
        prompt_preset: str,
        experiment: str,
        query: str,
        **trial_axes: Any,
    ) -> Series:
        """Load one query and non-seed trial condition across all seeds."""
        path = self._mosaic_path(
            model_preset=model_preset,
            prompt_preset=prompt_preset,
            experiment=experiment,
        )
        records = self._load_mosaic_records(path)
        expected_axes = _non_seed_axes(records, ndjson_path=path)
        _validate_trial_axis_selection(expected_axes, trial_axes)

        selected_records: list[dict[str, Any]] = []
        matched_trial_count = 0
        for record_idx, record in enumerate(records):
            trial_key = _trial_key(record, ndjson_path=path, record_idx=record_idx)
            if not _trial_axes_match(trial_key, trial_axes):
                continue

            matched_trial_count += 1
            results = record.get("results")
            if not isinstance(results, Mapping):
                raise ValueError(
                    f"Record {record_idx} in {path} has no mapping-valued "
                    "'results' field."
                )
            if query not in results:
                raise KeyError(
                    f"Query {query!r} is missing from record {record_idx} in "
                    f"{path} for trial_key={dict(trial_key)!r}."
                )
            query_result = results[query]
            if not isinstance(query_result, Mapping):
                raise ValueError(
                    f"Query {query!r} in record {record_idx} of {path} does "
                    "not map to a dictionary-like result."
                )
            selected_records.append(
                _merge_query_result_and_trial_key(query_result, trial_key)
            )

        if matched_trial_count == 0:
            raise ValueError(
                f"No records in {path} matched trial axes {trial_axes!r}."
            )

        return Series(
            records=selected_records,
            query=query,
            trial_axes=dict(trial_axes),
            ndjson_path=path,
            prompt_preset=prompt_preset,
        )

    def get_mosaic_options(
        self,
        *,
        model_preset: str,
        prompt_preset: str,
        experiment: str,
    ) -> dict[str, Any]:
        """Return observed queries and non-seed trial-axis values."""
        path = self._mosaic_path(
            model_preset=model_preset,
            prompt_preset=prompt_preset,
            experiment=experiment,
        )
        records = self._load_mosaic_records(path)
        first_trial_key = _trial_key(records[0], ndjson_path=path, record_idx=0)
        axis_names = tuple(axis for axis in first_trial_key if axis != SEED_AXIS)
        axis_values: dict[str, list[Any]] = {axis: [] for axis in axis_names}
        queries: list[str] = []

        for record_idx, record in enumerate(records):
            trial_key = _trial_key(
                record,
                ndjson_path=path,
                record_idx=record_idx,
            )
            record_axes = tuple(axis for axis in trial_key if axis != SEED_AXIS)
            if set(record_axes) != set(axis_names):
                raise ValueError(
                    f"Record {record_idx} in {path} has non-seed trial axes "
                    f"{sorted(record_axes)}, expected {sorted(axis_names)}."
                )
            for axis in axis_names:
                value = trial_key[axis]
                if not any(
                    _axis_values_equal(value, existing)
                    for existing in axis_values[axis]
                ):
                    axis_values[axis].append(value)

            results = record.get("results")
            if not isinstance(results, Mapping):
                raise ValueError(
                    f"Record {record_idx} in {path} has no mapping-valued "
                    "'results' field."
                )
            for query in results:
                query = str(query)
                if query not in queries:
                    queries.append(query)

        return {
            "queries": tuple(queries),
            "trial_axes": {
                axis: tuple(values)
                for axis, values in axis_values.items()
            },
        }

    def load_erasure_trajectory(
        self,
        *,
        model_preset: str,
        prompt_preset: str,
        task: str = "iterative_erasure",
    ) -> ErasureTrajectory:
        """Load one erasure trajectory from this run's results."""
        directory = self._erasure_path(
            model_preset=model_preset,
            prompt_preset=prompt_preset,
            task=task,
        )
        if directory not in self._trajectories_by_path:
            self._trajectories_by_path[directory] = _load_erasure_trajectory(
                directory
            )
        return self._trajectories_by_path[directory]

    def _mosaic_path(
        self,
        *,
        model_preset: str,
        prompt_preset: str,
        experiment: str,
    ) -> Path:
        if self.results_root is None:
            raise ValueError("This RunResults object has no results_root.")
        return (
            self.results_root
            / model_preset
            / prompt_preset
            / experiment
            / RESULTS_FILENAME
        )

    def _erasure_path(
        self,
        *,
        model_preset: str,
        prompt_preset: str,
        task: str,
    ) -> Path:
        if self.results_root is None:
            raise ValueError("This RunResults object has no results_root.")
        return self.results_root / model_preset / prompt_preset / task

    def _load_mosaic_records(self, path: Path) -> list[dict[str, Any]]:
        if path in self._records_by_path:
            return self._records_by_path[path]
        if not path.exists():
            raise FileNotFoundError(f"No mosaic results found at {path}")

        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Line {line_number} in {path} is not a JSON object."
                    )
                records.append(record)
        if not records:
            raise ValueError(f"No records found in {path}")

        self._records_by_path[path] = records
        return records


def _optional_directory(
    root: Path | str | None,
    *,
    name: str,
) -> Path | None:
    if root is None:
        return None
    path = Path(root)
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {path}")
    return path


def _non_seed_axes(
    records: Sequence[Mapping[str, Any]],
    *,
    ndjson_path: Path,
) -> set[str]:
    first_trial_key = _trial_key(
        records[0],
        ndjson_path=ndjson_path,
        record_idx=0,
    )
    expected_keys = set(first_trial_key)
    for record_idx, record in enumerate(records[1:], start=1):
        trial_key = _trial_key(
            record,
            ndjson_path=ndjson_path,
            record_idx=record_idx,
        )
        if set(trial_key) != expected_keys:
            raise ValueError(
                f"Record {record_idx} in {ndjson_path} has trial axes "
                f"{sorted(trial_key)}, expected {sorted(expected_keys)}."
            )
    return expected_keys - {SEED_AXIS}


def _validate_trial_axis_selection(
    expected_axes: set[str],
    trial_axes: Mapping[str, Any],
) -> None:
    actual_axes = set(trial_axes)
    if actual_axes == expected_axes:
        return
    missing = sorted(expected_axes - actual_axes)
    extra = sorted(actual_axes - expected_axes)
    raise ValueError(
        "Trial-axis arguments must specify every non-seed axis exactly once. "
        f"Missing axes: {missing}. Extra axes: {extra}."
    )


def _trial_key(
    record: Mapping[str, Any],
    *,
    ndjson_path: Path,
    record_idx: int,
) -> Mapping[str, Any]:
    trial_key = record.get("trial_key")
    if not isinstance(trial_key, Mapping):
        raise ValueError(
            f"Record {record_idx} in {ndjson_path} has no mapping-valued "
            "'trial_key' field."
        )
    return trial_key


def _trial_axes_match(
    trial_key: Mapping[str, Any],
    trial_axes: Mapping[str, Any],
) -> bool:
    return all(
        _axis_values_equal(trial_key[axis], value)
        for axis, value in trial_axes.items()
    )


def _axis_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return tuple(left) == tuple(right)
    return left == right


def _merge_query_result_and_trial_key(
    query_result: Mapping[str, Any],
    trial_key: Mapping[str, Any],
) -> dict[str, Any]:
    collisions = sorted(set(query_result) & set(trial_key))
    if collisions:
        raise ValueError(
            "Selected query result and trial_key share key(s): "
            f"{collisions}. Refusing to silently overwrite values."
        )
    return {**query_result, **trial_key}
