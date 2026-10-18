# Defining an experiment

[README](../README.md)

A `BaseExperiment` represents an experiment as a Cartesian grid. Each axis of the grid corresponds to an experimental variable; each point in the grid corresponds to a trial.

Each trial has two phases: 

1. Setup: Construct the trial-specific state.
2. Query: Run a collection of queries against that state.

Each query produces model outputs, which are converted into a serializable result and recorded.

---

## Implementation of an experiment

Import the base class and trial-state container with:

```python
from llm_neuroscience_tools.base_experiment import BaseExperiment, TrialState
```

To implement an experiment, subclass `BaseExperiment` and fill in the five hooks: 

- `define_trial_axes()`
- `build_trial_state()`
- `build_trial_queries()`
- `run_query()`
- `serialize_result()`

To execute an experiment, call `.run()` or `.run_ndjson()` on a BaseExperiment subclass instance. Both methods follow the same control flow:

```text
define_trial_axes()
    -> trial_axes: Mapping[str, Iterable[Any]]
    -> expanded into trial_key dicts

for each trial_key:
    build_trial_state(trial_key)
        -> trial_state: TrialState
            - trial_info
            - tensor_data

    build_trial_queries(trial_state)
        -> queries: Mapping[str, Any]

    for each query_name, query in queries.items():
        run_query(trial_state, query)
            -> logits: np.ndarray

        serialize_result(trial_state, query, logits)
            -> result: Dict[str, Any]

```

In either case, the output consists of trial records of the form:

```python
{
    "trial_key": trial_key,
    "trial_info": trial_state.trial_info,
    "results": {
        query_name: result,
        # Additional query results...
    }
}
```

Here, `trial_key` identifies the trial, `trial_info` is the metadata from `trial_state.trial_info`, and `results` maps each query name to the output of `serialize_result()`.

`.run()` returns a list of all trial records. `.run_ndjson(out_path)` appends each
record to a file and flushes after every trial. The parent directory must already
exist. It does not detect completed trials or refuse an existing file: repeating
a run with the same path appends records again. Choose a new output path for an
independent run, or enforce an overwrite guard in the calling script.

Trial keys, `trial_info`, and serialized results must be compatible with
`json.dumps` when writing NDJSON. Convert NumPy arrays and scalars to ordinary
Python lists and scalars in `serialize_result()` where needed. `tensor_data`
is not automatically serialized.

---

## Subclass hooks

Subclasses must implement the following hooks:

---

```text
define_trial_axes() -> Mapping[str, Iterable[Any]]
```

Defines the Cartesian grid of trials.

This hook should return a mapping from axis name to a finite iterable of values. The Cartesian product of these iterables defines the full set of trial keys.

For example, returning

```python
{"seed": range(3), "condition": ["control", "treatment"]}
```

defines six trials.

---

```text
build_trial_state(trial_key: Dict[str, Any]) -> TrialState
```

Defines the setup phase of an experiment. 

The input `trial_key` identifies the current trial by assigning one value to each trial axis. 

The hook should return a `TrialState` with two fields:

- `trial_info`: small metadata to record in the output
- `tensor_data`: larger ephemeral artifacts used during query execution

Only `trial_info` is written into the final trial record.

---

```text
build_trial_queries(trial_state: TrialState) -> Mapping[str, Any]
```

Defines the named queries for a trial.

This hook should return a mapping from `query_name` to query specification. Query specifications may have any shape, as long as they can be consumed by `run_query()`.

---

```text
run_query(trial_state: TrialState, query: Any) -> np.ndarray
```

Executes one query for one trial.

This hook should use the provided `trial_state` and query specification to run the relevant model computation. It should return the resulting logits as a NumPy array.

---

```text
serialize_result(trial_state: TrialState, query: Any, logits: np.ndarray)
	 -> Dict[str, Any]
```

Extracts, formats, and/or reduces the raw logits into the recorded result. 

This hook should return a small serializable dictionary, which is stored under that query’s name in the final results.
