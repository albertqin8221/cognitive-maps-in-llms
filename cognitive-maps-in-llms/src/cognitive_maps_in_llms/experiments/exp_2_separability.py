from typing import Any, Iterable, Mapping, Dict

import torch
import numpy as np

from llm_neuroscience_tools.backends.base import LMBackend
from llm_neuroscience_tools.prompts.relational import RelationalPromptRuntime
from llm_neuroscience_tools.base_experiment import TrialState, BaseExperiment

from .utils import (
    get_answer_token_logits_and_probs,
    build_context_patching_mask
)


RELATIONS = {'R_AB': 'FWD', 'R_BC': 'FWD'}


class Experiment_2_Separability(BaseExperiment):

    def __init__(self,
                 backend: LMBackend,
                 prompt_runtime: RelationalPromptRuntime,
                 n_trials: int):
        """Store handles to experiment components.
        """
        self.backend        = backend
        self.prompt_runtime = prompt_runtime
        self.tokenizer      = prompt_runtime.tokenizer
        self.n_trials       = n_trials

        # Store some useful landmarks
        self.A1 = self.prompt_runtime.landmarks['A1']
        self.B1 = self.prompt_runtime.landmarks['B1']
        self.B2 = self.prompt_runtime.landmarks['B2']
        self.C1 = self.prompt_runtime.landmarks['C1']
        self.patching_mask = build_context_patching_mask(backend, prompt_runtime)

    def define_trial_axes(self) -> Mapping[str, Iterable[Any]]:
        """The trial grid is defined by:
            (1) One axis over the random seed used to sample entities,
            (2) One axis over the patching location.
        """
        return {
            'seed': range(self.n_trials),
            'patch_location': ['A1', 'B1', 'B2', 'C1', 'CTRL']
        } 

    def build_trial_state(self, trial_key: Dict[str, Any]) -> TrialState:
        """Record activity on two prompts: 
        Acceptor: A -> B -> C
        Donor:    D -> E -> F

        Then, build a mosaic activation by replacing the activations of the 
        control tensor with those of the donor tensor at exactly one entity
        token.
        """

        seed           = trial_key['seed']
        patch_location = trial_key['patch_location']

        # (1) Record activity on A - B - C and D - E - F prompts
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
        Z_tla_donor, _    = self.backend.forward_with_recording(prompt_donor)
        Z_tla_control, _ = self.backend.forward_with_recording(prompt_control)

        # (2) Construct a mosaic tensor
        Z_tla_mosaic = Z_tla_control
        if patch_location == 'A1':
            Z_tla_mosaic[self.A1, ...] = Z_tla_donor[self.A1, ...]
        elif patch_location == 'B1':
            Z_tla_mosaic[self.B1, ...] = Z_tla_donor[self.B1, ...]
        elif patch_location == 'B2':
            Z_tla_mosaic[self.B2, ...] = Z_tla_donor[self.B2, ...]
        elif patch_location == 'C1':
            Z_tla_mosaic[self.C1, ...] = Z_tla_donor[self.C1, ...]
        elif patch_location == 'CTRL':
            pass
        else:
            raise ValueError
        return TrialState(
            trial_info={
                'context_entities_control': entities_control,
                'context_entities_donor':   entities_donor, 
                'context_relations':        RELATIONS},
            tensor_data=Z_tla_mosaic,
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
        return self.backend.forward_with_patching(
            prompt_ids = prompt,
            Z_tla = trial_state.tensor_data,
            mask_tla = self.patching_mask
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
