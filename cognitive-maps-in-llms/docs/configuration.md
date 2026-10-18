# TOML configuration file formats

[README](../README.md)

## Formatting a manifest configuration file

A manifest defines a run over a set of models, prompt families, and tasks. 
For example: 

```toml
run_id = "example_2026-09-17"
model_presets = ["llama-70b", "qwen-32b", "gemma-27b"]
prompt_presets = ["ABC_family_tree_yes_no", "ABC_family_tree_who_is"]
results = "/absolute/path/to/results"
artifacts = "/absolute/path/to/artifacts"

[[tasks]]
name = "activation_and_kv_recording"
kind = "recording"
settings = { n_trials = 50, start_seed = 10000 }

[[tasks]]
name = "iterative_erasure"
kind = "erasure"
settings = { n_trials = 10, start_seed = 20000 }
[tasks.dependencies]
recordings = "activation_and_kv_recording"

[[tasks]]
name = "exp_1_behavior"
kind = "mosaic_experiment"
experiment = "exp_1_behavior"
settings = { n_trials = 50 }
```

The top-level fields are:

- `run_id`: a name used to construct output paths. 
- `model_presets`: a nonempty list of model preset names.
- `prompt_presets`: a nonempty list of prompt preset names.
- `results`: absolute output root, required when the manifest contains a
  mosaic experiment or erasure task.
- `artifacts`: absolute output root, required when the manifest contains a
  recording task.
- `tasks`: a nonempty array of task definitions.

Each task has a unique `name`, a `kind`, and optional `settings` and
`dependencies` tables. The allowed values of `kind` and `experiments` are 
defined in [Experiments and Tasks](experiments.md). Entries in `settings` are 
passed as keyword arguments to the function that runs the task. See 
[Experiments and Tasks](experiments.md) for the settings required
by each task or experiment. 

The task's `name` determines its output directory:
`<output-root>/<run_id>/<model-preset>/<prompt-preset>/<task-name>/`.
Mosaic experiments and erasure tasks use the `results` root; recording tasks
use the `artifacts` root.

Tasks may additionally define `settings_by_model`. Each entry names a model
from the manifest's `model_presets` list and overrides selected shared settings
for that model. All other settings remain unchanged. For example, the paper's
erasure and bottleneck tasks use a halfway split except for Qwen:

```toml
[tasks.settings_by_model]
qwen-32b = { dividing_layer_fraction = 0.6666666666666666 }
```

Some tasks consume outputs produced by other tasks. These dependencies can be 
specified in two ways. First, if the producer task is defined in the same manifest, 
the dependency can be specified by the producer task's name. For example:

```toml
[tasks.dependencies]
recordings = "activation_and_kv_recording"
```

In this case, the generator links each dependent task to the configuration file 
generated for the `activation_and_kv_recording` task with the matching model and prompt family.

Second, a dependency on artifacts from an earlier run can be specified by providing
a collection of producer configuration files. For example:

```toml
[tasks.dependencies]
recordings = { configs = [
    "/absolute/path/to/gemma_who_is_config.toml",
    "/absolute/path/to/gemma_yes_no_config.toml",
    "/absolute/path/to/llama_who_is_config.toml",
    "/absolute/path/to/llama_yes_no_config.toml",
    # ...
] }
```

In this case, the generator selects the unique producer configuration whose
`model_preset` and `prompt_preset` match those of the dependent task. If no 
unique match exists, the generator raises an error. 

## Formatting a task configuration file

A task configuration file contains all the information needed to run a single task.
For example:

```toml
output_folder = "/absolute/path/to/results/example_2026-09-17/gemma-27b/ABC_family_tree_who_is/iterative_erasure"

[metadata]
source_manifest = "/absolute/path/to/manifest.toml"
run_id = "example_2026-09-17"
created_at = 2026-09-17T19:00:00Z

[inputs]
kind = "erasure"
model_preset = "gemma-27b"
prompt_preset = "ABC_family_tree_who_is"
settings = { n_trials = 10, start_seed = 20000 }

[dependencies]
recordings = "/absolute/path/to/activation_and_kv_recording_config.toml"

[scheduling]
```

`output_folder` specifies the absolute path where the task writes its outputs.

Dependencies are specified as absolute paths to the configuration files of 
producer tasks. At runtime, the runner passes each producer task's 
`output_folder` to the function that runs the dependent task. Producer tasks
must be marked as complete by the presence of a valid `completed.json` file in 
the relevant `output_folder`; otherwise, the runner raises an error. 

Each configuration file also contains a `[scheduling]` table. This table
is used only by the orchestration tools `populate_scheduling.py` and 
`plan_slurm_jobs.py` when partitioning tasks into SLURM jobs.

After annotation by `populate_scheduling.py`, an active task has:

```toml
[scheduling]
estimated_runtime_seconds = 4200
gpu_profile = "1xgpu80"
```

To retain a generated config while excluding it from generated
job plans, add the optional Boolean flag:

```toml
[scheduling]
estimated_runtime_seconds = 4200
gpu_profile = "1xgpu80"
skip = true
```

Typically, task configuration files are generated and executed
programmatically. However, both steps can also be performed manually: task
configuration files can be written by hand, and `tools/run_configs.py` can be
called directly on a list of configuration files. Note that when executing 
multiple configurations, `tools/run_configs.py` orders the tasks to group 
tasks that use the same model, then run each producer task before any task
that depends on it. 
