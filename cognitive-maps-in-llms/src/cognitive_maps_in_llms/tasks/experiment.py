from pathlib import Path
from typing import Any

from llm_neuroscience_tools.backends.base import LMBackend
from llm_neuroscience_tools.base_experiment import BaseExperiment
from llm_neuroscience_tools.prompts.relational import RelationalPromptRuntime


RESULTS_FILENAME = "results.ndjson"


def run_experiment(
    backend: LMBackend,
    prompt_runtime: RelationalPromptRuntime,
    output_dir: Path,
    *,
    experiment_cls: type[BaseExperiment],
    **experiment_kwargs: Any,
) -> None:
    """Instantiate and run one mosaic experiment.

    Args:
        backend: Initialized model backend.
        prompt_runtime: Initialized relational prompt runtime.
        output_dir: Directory in which the NDJSON results will be written.
        experiment_cls: Registered :class:`BaseExperiment` subclass.
        **experiment_kwargs: Settings and resolved dependency paths forwarded
            to the experiment constructor.

    Raises:
        TypeError: If ``experiment_cls`` is not a :class:`BaseExperiment`
            subclass.
        FileExistsError: If the results artifact already exists.
    """
    if not isinstance(experiment_cls, type) or not issubclass(
        experiment_cls,
        BaseExperiment,
    ):
        raise TypeError("experiment_cls must be a BaseExperiment subclass")

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / RESULTS_FILENAME
    if results_path.exists() or results_path.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite existing artifact: {results_path}"
        )

    experiment = experiment_cls(
        backend,
        prompt_runtime,
        **experiment_kwargs,
    )
    experiment.run_ndjson(results_path)
