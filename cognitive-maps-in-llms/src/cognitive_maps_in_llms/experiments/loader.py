"""Resolve registered mosaic-experiment classes on demand."""

from __future__ import annotations

from importlib import import_module
from typing import TypeAlias, cast

from llm_neuroscience_tools.base_experiment import BaseExperiment


ExperimentClass: TypeAlias = type[BaseExperiment]

_EXPERIMENTS: dict[str, tuple[str, str]] = {
    "exp_1_behavior": (
        ".exp_1_behavior", 
        "Experiment_1_Behavior"
    ),
    "exp_2_separability": (
        ".exp_2_separability", 
        "Experiment_2_Separability"
    ),
    "exp_3_additivity": (
        ".exp_3_additivity", 
        "Experiment_3_Additivity"
    ),
    "exp_4_retrospective_prospective": (
        ".exp_4_retrospective_prospective",
        "Experiment_4_Retrospective_Prospective",
    ),
    "exp_5a_bottleneck": (
        ".exp_5a_bottleneck",
        "Experiment_5A_Relational_Bottleneck",
    ),
    "exp_5b_sparse_mosaic": (
        ".exp_5b_sparse_mosaic",
        "Experiment_5B_Sparse_Mosaic",
    ),
    "exp_5c_editing": (
        ".exp_5c_editing",
        "Experiment_5C_Editing",
    ),
    "exp_s1_positions_rope": (
        ".exp_s1_positions_rope",
        "Experiment_S1_Positions_RoPE",
    ),
    "exp_s2_positions_indirect_swap": (
        ".exp_s2_positions_indirect_swap",
        "Experiment_S2_Positions_Indirect_Swap",
    ),
}


def get_experiment_names() -> tuple[str, ...]:
    """Return the registered experiment names in definition order."""
    return tuple(_EXPERIMENTS)


def get_experiment_class(name: str) -> ExperimentClass:
    """Import and return the registered experiment class for ``name``.

    Raises:
        ValueError: If ``name`` is unregistered or does not resolve to a
            :class:`BaseExperiment` subclass.
    """
    try:
        module_name, class_name = _EXPERIMENTS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown experiment {name!r}; expected one of "
            f"{sorted(_EXPERIMENTS)}"
        ) from exc

    module = import_module(module_name, package=__package__)
    experiment_cls = getattr(module, class_name, None)
    if not isinstance(experiment_cls, type) or not issubclass(
        experiment_cls,
        BaseExperiment,
    ):
        raise ValueError(f"Registered experiment {name!r} is not runnable")
    return cast(ExperimentClass, experiment_cls)
