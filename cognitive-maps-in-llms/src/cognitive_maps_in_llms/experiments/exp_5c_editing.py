
from pathlib import Path
from typing import Any, Iterable, Mapping, Dict

import torch
import numpy as np

from llm_neuroscience_tools.backends.base import LMBackend
from llm_neuroscience_tools.prompts.relational import RelationalPromptRuntime
from llm_neuroscience_tools.base_experiment import TrialState, BaseExperiment

from .utils_erasure import load_erasure_conditions
from .utils import get_answer_token_logits_and_probs


RELATIONS = {"R_AB": "FWD", "R_BC": "FWD"}
ERASURE_CONDITIONS = [
    "FIDELITY_95",
    "FIDELITY_95_SITES_X2",
]
EDITING_LOCATIONS = [
    'ALL',
    'A_KEYS',
    'A_VALUES',
    'B_KEYS',
    'B_VALUES',
    'C_KEYS',
    'C_VALUES',
    'NONE',
]


class Experiment_5C_Editing(BaseExperiment):

    def __init__(
        self,
        backend: LMBackend,
        prompt_runtime: RelationalPromptRuntime,
        n_trials: int,
        erasure_trajectory: Path | str
    ):
        """Store handles to experiment components.

        Extra args:
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

        # Load the dependency
        self.masks_by_condition = load_erasure_conditions(
            erasure_dir = erasure_trajectory,
            condition_names = ERASURE_CONDITIONS
        )

    def define_trial_axes(self) -> Mapping[str, Iterable[Any]]:
        """The trial grid is defined by:
            (1) One axis over the random seed used to sample entities,
            (2) One axis that specifies what "level of erasure" to use.
            (3) One axis that replaces keys and/or values expected to
                affect *particular* (name, role) or (role, name) associations
        """
        return {
            'seed': range(self.n_trials),
            'condition': ERASURE_CONDITIONS,
            'editing_locs': EDITING_LOCATIONS
        }

    def build_trial_state(self, trial_key: Dict[str, Any]) -> TrialState:

        seed         = trial_key['seed']
        condition    = trial_key['condition']
        editing_locs = trial_key['editing_locs']

        # (1) Record KV caches on two prompts
        entities_donor, entities_control = (
            self.prompt_runtime.sample_nonoverlapping_entities(seed)
        )
        prompt_donor = self.prompt_runtime.render_prompt_as_tokens(
            context_entities  = entities_donor,
            context_relations = RELATIONS
        )
        prompt_control = self.prompt_runtime.render_prompt_as_tokens(
            context_entities  = entities_control,
            context_relations = RELATIONS
        )
        KV_data_donor, _ = self.backend.forward_with_KV_recording(prompt_donor)
        KV_data_control, _ = self.backend.forward_with_KV_recording(prompt_control)

        # (2) Resolve which tokens and cache components to edit
        if editing_locs == 'ALL':
            token_indices_to_edit_cache = list(
                range(self.CTXT_START, self.CTXT_END + 1)
            )
            edit_keys = edit_values = True
        elif editing_locs == 'NONE':
            token_indices_to_edit_cache = []
            edit_keys = edit_values = False
        elif editing_locs == 'A_KEYS':
            token_indices_to_edit_cache = list(range(
                self.prompt_runtime.landmarks['A1'],
                self.prompt_runtime.landmarks['B1']
            ))
            edit_keys = True
            edit_values = False
        elif editing_locs == 'A_VALUES':
            token_indices_to_edit_cache = [self.prompt_runtime.landmarks['A1']]
            edit_keys = False
            edit_values = True
        elif editing_locs == 'B_KEYS':
            token_indices_to_edit_cache = list(range(
                self.prompt_runtime.landmarks['B1'],
                self.prompt_runtime.landmarks['C1']
            ))
            edit_keys = True
            edit_values = False
        elif editing_locs == 'B_VALUES':
            token_indices_to_edit_cache = [
                self.prompt_runtime.landmarks['B1'],
                self.prompt_runtime.landmarks['B2'],
            ]
            edit_keys = False
            edit_values = True
        elif editing_locs == 'C_KEYS':
            token_indices_to_edit_cache = list(range(
                self.prompt_runtime.landmarks['C1'],
                self.CTXT_END + 1
            ))
            edit_keys = True
            edit_values = False
        elif editing_locs == 'C_VALUES':
            token_indices_to_edit_cache = [self.prompt_runtime.landmarks['C1']]
            edit_keys = False
            edit_values = True
        else:
            raise ValueError(f"Unknown editing location: {editing_locs!r}")

        # (3) Select sites and copy them from donor to control
        resolved_condition = self.masks_by_condition[condition]
        KV_data_mosaic = KV_data_control.clone()
        n_keys = n_values = 0

        # Compute token positions relative to the context segment
        token_indices_to_edit_context =\
            [t - self.CTXT_START for t in token_indices_to_edit_cache]

        if edit_keys:
            key_mask_selected = resolved_condition.key_mask[token_indices_to_edit_context]
            n_keys = int(key_mask_selected.sum())
            K_mask_sli = torch.as_tensor(  # s indexes tokens to edit
                key_mask_selected,
                dtype=torch.bool,
                device=KV_data_control.K_tlia.device,
            )
            KV_data_mosaic.K_tlia[token_indices_to_edit_cache] = torch.where(
                K_mask_sli.unsqueeze(-1),
                KV_data_donor.K_tlia[token_indices_to_edit_cache],
                KV_data_control.K_tlia[token_indices_to_edit_cache],
            )
        if edit_values:
            value_mask_selected = resolved_condition.value_mask[token_indices_to_edit_context]
            n_values = int(value_mask_selected.sum())
            V_mask_sli = torch.as_tensor(  # s indexes tokens to edit
                value_mask_selected,
                dtype=torch.bool,
                device=KV_data_control.V_tlia.device,
            )
            KV_data_mosaic.V_tlia[token_indices_to_edit_cache] = torch.where(
                V_mask_sli.unsqueeze(-1),
                KV_data_donor.V_tlia[token_indices_to_edit_cache],
                KV_data_control.V_tlia[token_indices_to_edit_cache],
            )

        return TrialState(
            trial_info={
                'context_entities_control': entities_control,
                'context_entities_donor':   entities_donor,
                'context_relations':        RELATIONS,
                'n_keys':                   n_keys,
                'n_values':                 n_values,
                'erasure_iteration':        resolved_condition.iteration},
            tensor_data=KV_data_mosaic,
        )

    def build_trial_queries(self, trial_state: TrialState) -> Mapping[str, Any]:
        """Systematically generate all queries.
        """
        return self.prompt_runtime.build_query_specs(
            entity_symbols_to_names = {
                'A': trial_state.trial_info["context_entities_control"]['A'],
                'B': trial_state.trial_info["context_entities_control"]['B'],
                'C': trial_state.trial_info["context_entities_control"]['C'],
                'D': trial_state.trial_info["context_entities_donor"]['A'],
                'E': trial_state.trial_info["context_entities_donor"]['B'],
                'F': trial_state.trial_info["context_entities_donor"]['C']
            }
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
