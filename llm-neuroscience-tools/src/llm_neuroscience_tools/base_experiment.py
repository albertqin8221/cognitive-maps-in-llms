from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Iterator
from pathlib import Path

import numpy as np
from tqdm import tqdm
import json


@dataclass(frozen=True)
class TrialState:
    """
    A minimal container for the data associated with a single trial.

    Fields:
        trial_info: Small JSON-like metadata to record in the final
            trial output.
        tensor_data: Larger ephemeral artifacts used during query execution
            but not written directly into the final results.
    """
    trial_info: Any
    tensor_data: Any


class BaseExperiment(ABC):
    """
    Base class implementing an experiment as a Cartesian grid of trials.

    Each axis of the grid corresponds to an experimental variable; each
    point in the grid corresponds to a trial.

    Each trial has two phases:
        1) Setup: construct the trial-specific state.
        2) Query: run a collection of queries against that state.

    Each query produces model outputs, which are converted into a small
    recorded result.

    To implement an experiment, subclass `BaseExperiment` and fill in the
    five hooks:
    - `define_trial_axes()`
    - `build_trial_state()`
    - `build_trial_queries()`
    - `run_query()`
    - `serialize_result()`
    """

    # ------ Core template method ------

    def _drive_trial_loop(self) -> Iterator[Dict[str, Any]]:
        """
        Internal generator implementing the core control flow of an experiment.

        This method:
        1) Calls `define_trial_axes()` to define the Cartesian grid of trials.
        2) Expands that grid into trial keys.
        3) For each trial key:
           - calls `build_trial_state(trial_key)`
           - calls `build_trial_queries(trial_state)`
           - for each query:
             - calls `run_query(trial_state, query)`
             - calls `serialize_result(trial_state, query, logits)`

        Yields:
            Dict[str, Any]: A trial record of the form:

                {
                    "trial_key": trial_key,
                    "trial_info": trial_state.trial_info,
                    "results": {
                        query_name: result,
                        ...
                    }
                }
        """
        # Turn trial axes into a list of trial keys
        trial_axes = self.define_trial_axes()
        trial_keys = []
        keys = list(trial_axes)
        for values in product(*(trial_axes[k] for k in keys)):
            trial_keys.append(dict(zip(keys, values)))
        
        # Main experiment loop
        for trial_key in tqdm(trial_keys, disable=None):

            # 1). Set up this trial
            trial_state = self.build_trial_state(trial_key)

            # 1a). Allocate a new trial record dictionary
            trial_record = {
                'trial_key': trial_key,
                'trial_info': trial_state.trial_info,
                'results': {}
            }

            # 2). Run all queries for this trial
            queries = self.build_trial_queries(trial_state)
            for query_name, query in queries.items():

                # 2a). Run the forward pass
                logits = self.run_query(trial_state, query)

                # 2b). Record the results 
                trial_record['results'][query_name] = self.serialize_result(
                    trial_state,
                    query,
                    logits,
                )

            yield trial_record


    def run(self) -> List[Dict[str, Any]]:
        """
        Run the experiment and return the resulting trial records.

        This method materializes the trial records produced by
        `_drive_trial_loop()` into a list.

        Returns:
            List[Dict[str, Any]]: A list of trial records, one per trial.
        """
        return list(self._drive_trial_loop())
    

    def run_ndjson(self, out_path: str | Path) -> None:
        """
        Run the experiment and stream trial records to an NDJSON file.

        This method executes the same control flow as `run()`, but writes each
        trial record to `out_path` as a JSON line as soon as it is produced.

        Args:
            out_path (str | Path): Path to the output NDJSON file.
        """
        with open(out_path, 'a') as f:
            for trial_dict in self._drive_trial_loop():
                f.write(json.dumps(trial_dict) + '\n')
                f.flush()

    
    # ---- Hooks for subclasses ----

    @abstractmethod
    def define_trial_axes(self) -> Mapping[str, Iterable[Any]]:
        """
        Define the Cartesian grid of trials.

        Returns:
            Mapping[str, Iterable[Any]]: A mapping from axis name to a finite
                iterable of values. The Cartesian product of these iterables
                defines the full set of trial keys.
        """
        raise NotImplementedError

    @abstractmethod
    def build_trial_state(
            self, 
            trial_key: Dict[str, Any]
        ) -> TrialState:
        """
        Construct the per-trial state.

        Args:
            trial_key (Dict[str, Any]): The current trial key, assigning one
                value to each trial axis.

        Returns:
            TrialState: The state for this trial, with two fields:
                - `trial_info`: small metadata to record in the output
                - `tensor_data`: larger ephemeral artifacts used during query
                  execution

                Only `trial_info` is written into the final trial record.
        """
        raise NotImplementedError

    @abstractmethod
    def build_trial_queries(
            self,
            trial_state: TrialState,
        ) -> Mapping[str, Any]:
        """
        Build the named queries to run for a trial.

        Args:
            trial_state (TrialState): The per-trial state produced by
                `build_trial_state()`.

        Returns:
            Mapping[str, Any]: A mapping from `query_name` to query
                specification. Query specifications may have any shape, as long
                as they can be consumed by `run_query()`.
        """
        raise NotImplementedError

    @abstractmethod
    def run_query(
            self,
            trial_state: TrialState,
            query: Any,
        ) -> np.ndarray:
        """
        Execute a single query for a given trial.

        Args:
            trial_state (TrialState): The per-trial state produced by
                `build_trial_state()`.
            query (Any): The query specification to execute.

        Returns:
            np.ndarray: The logits produced for this query.
        """
        raise NotImplementedError

    @abstractmethod
    def serialize_result(
            self, 
            trial_state: TrialState,
            query: Any,
            logits: np.ndarray
        ) -> Dict[str, Any]:
        """
        Extract, format, and/or reduce the raw logits into the recorded result.

        Args:
            trial_state (TrialState): The per-trial state produced by
                `build_trial_state()`.
            query (Any): The query specification associated with `logits`.
            logits (np.ndarray): The logits produced by `run_query()`.

        Returns:
            Dict[str, Any]: A small serializable dictionary, stored under the
                query's name in the final results.
        """

        raise NotImplementedError
