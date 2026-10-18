from typing import Any, Iterable, Mapping, Dict

import numpy as np

from llm_neuroscience_tools.backends.base import LMBackend
from llm_neuroscience_tools.prompts.relational import RelationalPromptRuntime
from llm_neuroscience_tools.base_experiment import TrialState, BaseExperiment

from .utils import (
    get_answer_token_logits_and_probs,
    build_context_patching_mask,
)


class Experiment_S1_Positions_RoPE(BaseExperiment):

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
        self.patching_mask = build_context_patching_mask(backend, prompt_runtime)

    def define_trial_axes(self) -> Mapping[str, Iterable[Any]]:
        """The trial grid defined by:
            (1) One axis over the random seed used to sample entities,
            (2) One axis over the patching location.
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

        # (1) Record activity on a single prompt
        RELATIONS = {'R_AB': 'FWD', 'R_BC': 'FWD'}
        entities  = self.prompt_runtime.sample_entities(seed)
        prompt    = self.prompt_runtime.render_prompt_as_tokens(
            context_entities  = entities,
            context_relations = RELATIONS 
        )
        Z_tla_mosaic, _ = self.backend.forward_with_recording(prompt)

        # (2) Swap the two rows in the activation tensor Z_tla corresponding to 
        # the A1 and B1 token positions.
        if condition == 'SWAP':
            Z_tla_mosaic[[self.A1, self.B1]] = Z_tla_mosaic[[self.B1, self.A1]]
        elif condition == 'CTRL':
            pass
        else:
            raise ValueError

        return TrialState(
            trial_info={
                'context_entities':  entities, 
                'context_relations': RELATIONS},
            tensor_data=Z_tla_mosaic,
        )
    
    def build_trial_queries(self, trial_state: TrialState) -> Mapping[str, Any]:
        """Systematically generate all queries. 
        """
        return self.prompt_runtime.build_query_specs(
            entity_symbols_to_names=trial_state.trial_info["context_entities"]
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
