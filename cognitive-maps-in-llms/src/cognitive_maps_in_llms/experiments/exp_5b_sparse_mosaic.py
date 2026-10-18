from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch

from llm_neuroscience_tools.backends.base import LMBackend
from llm_neuroscience_tools.base_experiment import BaseExperiment, TrialState
from llm_neuroscience_tools.prompts.relational import RelationalPromptRuntime

from cognitive_maps_in_llms.tasks.key_value_erasure_tools import load_average_cache

from .utils_erasure import load_erasure_conditions
from .utils import get_answer_token_logits_and_probs


RELATIONS = {"R_AB": "FWD", "R_BC": "FWD"}
ERASURE_CONDITIONS = [
    "REPLACE_ALL",
    "REPLACE_NONE",
    "FIDELITY_50",
    "FIDELITY_80",
    "FIDELITY_95",
    "FIDELITY_95_SITES_X2",
]


class Experiment_5B_Sparse_Mosaic(BaseExperiment):

    def __init__(
        self,
        backend: LMBackend,
        prompt_runtime: RelationalPromptRuntime,
        n_trials: int,
        recordings: Path | str,
        erasure_trajectory: Path | str
    ):
        """Store handles to experiment components.

        Extra args:
            recordings (Path | str): Directory containing
                ``recordings_metadata.json``, ``K_gstlia.npy``, and
                ``V_gstlia.npy`` for the current prompt/model preset pair.
            erasure_trajectory (Path | str): Directory containing
                ``erasure_state.json`` and ``erasure_metadata.json`` produced
                by the erasure task. Used to resolve named erasure conditions
                into key and value masks.
        """
        self.backend = backend
        self.prompt_runtime = prompt_runtime
        self.tokenizer = prompt_runtime.tokenizer
        self.n_trials = n_trials

        # Extract some useful landmarks
        self.CTXT_START = self.prompt_runtime.landmarks["CTXT_START"]
        self.CTXT_END = self.prompt_runtime.landmarks["CTXT_END"]
        self.CONTEXT = slice(self.CTXT_START, self.CTXT_END + 1)

        # Load the two dependencies
        self.KV_data_avg = load_average_cache(
            recordings,
            backend,
            prompt_runtime,
        )
        self.masks_by_condition = load_erasure_conditions(
            erasure_dir = erasure_trajectory,
            condition_names = ERASURE_CONDITIONS
        )

    def define_trial_axes(self) -> Mapping[str, Iterable[Any]]:
        """The trial grid is defined by:
            (1) One axis over the random seed used to sample entities,
            (2) One axis that specifies what "level of erasure" to use.
        """
        return {
            'seed': range(self.n_trials),
            'condition': ERASURE_CONDITIONS
        }

    def build_trial_state(self, trial_key: Dict[str, Any]) -> TrialState:
        """
        Build a mosaic key-value memory that retains information only in a
        sparse subset of keys and values.
        """

        seed = trial_key["seed"]
        condition = trial_key["condition"]

        # (1) Record KV cache on a single prompt
        entities = self.prompt_runtime.sample_entities(seed)
        prompt = self.prompt_runtime.render_prompt_as_tokens(
            context_entities=entities,
            context_relations=RELATIONS,
        )
        KV_data_control, _ = self.backend.forward_with_KV_recording(prompt)

        # (2) Pull the relevant masks and turn them into torch tensors
        resolved_condition = self.masks_by_condition[condition]
        remaining_K_tli = resolved_condition.key_mask
        remaining_V_tli = resolved_condition.value_mask
        remaining_K_tli = torch.as_tensor(
            remaining_K_tli,
            device=KV_data_control.K_tlia.device,
        ).unsqueeze(-1)
        remaining_V_tli = torch.as_tensor(
            remaining_V_tli,
            device=KV_data_control.V_tlia.device,
        ).unsqueeze(-1)

        # (3) Construct the mosaic
        KV_data_mosaic = KV_data_control.clone()
        KV_data_mosaic.K_tlia[self.CONTEXT] = torch.where(
            remaining_K_tli,
            KV_data_control.K_tlia[self.CONTEXT],
            self.KV_data_avg.K_tlia[self.CONTEXT],
        )
        KV_data_mosaic.V_tlia[self.CONTEXT] = torch.where(
            remaining_V_tli,
            KV_data_control.V_tlia[self.CONTEXT],
            self.KV_data_avg.V_tlia[self.CONTEXT],
        )

        return TrialState(
            trial_info={
                "context_entities": entities,
                "context_relations": RELATIONS,
                "n_keys": resolved_condition.n_keys,
                "n_values": resolved_condition.n_values,
                "erasure_iteration": resolved_condition.iteration,
            },
            tensor_data=KV_data_mosaic,
        )

    def build_trial_queries(self, trial_state: TrialState) -> Mapping[str, Any]:
        """Systematically generate all queries.
        """
        return self.prompt_runtime.build_query_specs(
            entity_symbols_to_names = trial_state.trial_info["context_entities"]
        )

    def run_query(self, trial_state: TrialState, query: Any) -> np.ndarray:
        """Run a forward pass, patching in the mosaic activations at the context.
        """
        prompt = self.prompt_runtime.render_prompt_as_tokens(
            context_entities  = None,
            context_relations = None,
            query_entities    = query['query_entities'],
            query_relations   = query['query_relations'],
        )
        return self.backend.forward_with_KV_patching(
            prompt_ids = prompt,
            cache_data = trial_state.tensor_data,
            patch_end  = self.CTXT_END
        )[1]

    def serialize_result(
            self,
            trial_state: TrialState,
            query: Any,
            logits: np.ndarray
        ) -> Dict[str, Any]:
        """Record logits and probabilities for all answer tokens.
        """
        return get_answer_token_logits_and_probs(
            answer_tokens = query['answer_tokens'],
            logits = logits,
            tokenizer = self.tokenizer
        )
