from typing import Any, Iterable, Mapping, Dict

import numpy as np

from llm_neuroscience_tools.backends.base import LMBackend
from llm_neuroscience_tools.prompts.relational import RelationalPromptRuntime
from llm_neuroscience_tools.base_experiment import TrialState, BaseExperiment

from .utils import (
    get_answer_token_logits_and_probs,
    build_context_patching_mask,
)


RELATIONS = {'R_AB': 'FWD', 'R_BC': 'FWD'}


class Experiment_3_Additivity(BaseExperiment):

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

        # Extract some useful landmarks
        self.A1 = self.prompt_runtime.landmarks['A1']
        self.C1 = self.prompt_runtime.landmarks['C1']
        self.patching_mask = build_context_patching_mask(backend, prompt_runtime)

    def define_trial_axes(self) -> Mapping[str, Iterable[Any]]:
        """The trial grid is defined by:
            (1) One axis over the random seed used to sample entities,
            (2) One axis that specifies whether to apply the swap vector.
        """
        return {
            'seed': range(self.n_trials),
            'condition': ['CTRL', 'SWAP']
        } 

    def build_trial_state(self, trial_key: Dict[str, Any]) -> TrialState:
        """Swap positions MAKE THIS DOCSTRING BETTER...
        """

        seed      = trial_key['seed']
        condition = trial_key['condition']

        # (1) Construct three prompts
        entities_control, entities_donor_A = \
            self.prompt_runtime.sample_nonoverlapping_entities(seed)
        entities_donor_C = {
            'A': entities_donor_A['C'],
            'B': entities_donor_A['B'],
            'C': entities_donor_A['A'],
        }

        # (2) Record activation tensors for each prompt
        prompt_control = self.prompt_runtime.render_prompt_as_tokens(
            context_entities  = entities_control,
            context_relations = RELATIONS 
        )
        Z_tla_control, _ = self.backend.forward_with_recording(prompt_control)
        prompt_donor_A = self.prompt_runtime.render_prompt_as_tokens(
            context_entities  = entities_donor_A,
            context_relations = RELATIONS 
        )
        Z_tla_donor_A, _ = self.backend.forward_with_recording(prompt_donor_A)
        prompt_donor_C = self.prompt_runtime.render_prompt_as_tokens(
            context_entities  = entities_donor_C,
            context_relations = RELATIONS 
        )
        Z_tla_donor_C, _ = self.backend.forward_with_recording(prompt_donor_C)

        # (3) Construct the swapping vector
        Delta_la = Z_tla_donor_C[self.C1, ...] - Z_tla_donor_A[self.A1, ...]

        # (4) Apply the swapping vector
        Z_tla_mosaic = Z_tla_control
        if condition == 'CTRL':
            pass
        elif condition == 'SWAP':
            Z_tla_mosaic[self.A1, ...] += Delta_la
            Z_tla_mosaic[self.C1, ...] -= Delta_la
        else:
            raise ValueError(f"Unknown condition: {condition!r}")

        return TrialState(
            trial_info={
                'context_entities_control': entities_control,
                'context_entities_donor_A': entities_donor_A, 
                'context_entities_donor_C': entities_donor_C, 
                'context_relations': RELATIONS},
            tensor_data=Z_tla_mosaic,
        )
    
    def build_trial_queries(self, trial_state: TrialState) -> Mapping[str, Any]:
        """Systematically generate all queries. 
        """
        return self.prompt_runtime.build_query_specs(
            entity_symbols_to_names=trial_state.trial_info["context_entities_control"]
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
            answer_tokens=query['answer_tokens'],
            logits=logits,
            tokenizer=self.tokenizer,
        )
