from typing import Any, Iterable, Mapping, Dict

import torch
import numpy as np

from llm_neuroscience_tools.backends.base import LMBackend
from llm_neuroscience_tools.prompts.relational import RelationalPromptRuntime
from llm_neuroscience_tools.base_experiment import TrialState, BaseExperiment

from .utils import get_answer_token_logits_and_probs


class Experiment_S2_Positions_Indirect_Swap(BaseExperiment):

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
        self.A1_B1_DIST = self.B1 - self.A1
        self.CTXT_START = self.prompt_runtime.landmarks['CTXT_START']

        # Use a space as a dummy token
        self.dummy_token_ID = self.tokenizer.encode(" ", add_special_tokens=False)[0]

        # This patching mask should be used for the +DELTA prompt!!
        self.patching_mask = self._get_patching_mask()

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

        # 1). Construct a prompt
        RELATIONS = {'R_AB': 'FWD', 'R_BC': 'FWD'}
        entities  = self.prompt_runtime.sample_entities(seed)
        prompt    = self.prompt_runtime.render_prompt_as_tokens(
            context_entities  = entities,
            context_relations = RELATIONS 
        )

        # 2). Construct +delta, +2delta versions of the prompt
        prompt_plus_delta = self._insert_dummy_tokens(
            prompt_ids    = prompt,
            insertion_pos = self.CTXT_START,
            n_repeats     = self.A1_B1_DIST
        )
        prompt_plus_2delta = self._insert_dummy_tokens(
            prompt_ids    = prompt,
            insertion_pos = self.CTXT_START,
            n_repeats     = 2*self.A1_B1_DIST
        )

        # 3). Construct the activation tensor
        Z_tla_mosaic, _ = self.backend.forward_with_recording(prompt_plus_delta)
        if condition == 'CTRL':
            pass
        elif condition == 'SWAP':
            # Also record on +0 and +2DELTA prompts
            Z_tla_0, _      = self.backend.forward_with_recording(prompt)
            Z_tla_2delta, _ = self.backend.forward_with_recording(prompt_plus_2delta)
            
            # Patch, at the corresponding B1 tokens, from the +0 to the +DELTA prompt 
            Z_tla_mosaic[self.B1+self.A1_B1_DIST, :] = Z_tla_0[self.B1, :]

            # Patch, at the corresponding A1 tokens, from the +2DELTA to the +DELTA prompt
            Z_tla_mosaic[self.A1+self.A1_B1_DIST, :] = Z_tla_2delta[self.A1+2*self.A1_B1_DIST, :]
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
        prompt = self._insert_dummy_tokens(  # add DELTA dummy tokens
            prompt_ids    = prompt,
            insertion_pos = self.CTXT_START,
            n_repeats     = self.A1_B1_DIST
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
    
    def _get_patching_mask(self):
        """Returns a patching mask that is True at all context tokens.
        This patching mask is to be used for the +DELTA prompt.
        """
         
        # Run a dummy forward pass
        entities = self.prompt_runtime.sample_entities(0)
        prompt   = self.prompt_runtime.render_prompt_as_tokens(
            context_entities  = entities,
            context_relations = {'R_AB': 'FWD', 'R_BC': 'FWD'}
        )
        prompt = self._insert_dummy_tokens(  # add DELTA dummy tokens
            prompt_ids    = prompt,
            insertion_pos = self.CTXT_START,
            n_repeats     = self.A1_B1_DIST
        )
        Z_tla, _ = self.backend.forward_with_recording(prompt)

        # Construct the patching mask
        CTXT_START = self.prompt_runtime.landmarks['CTXT_START']+self.A1_B1_DIST
        CTXT_END   = self.prompt_runtime.landmarks['CTXT_END']+self.A1_B1_DIST
        mask_tla = torch.zeros_like(Z_tla, dtype=torch.bool)
        mask_tla[CTXT_START:CTXT_END+1, ...] = True
        
        return mask_tla
    
    def _insert_dummy_tokens(
        self,
        prompt_ids: list[int],
        insertion_pos: int,
        n_repeats: int,
    ) -> list[int]:
        """Insert repeated dummy tokens at a chosen position in a prompt.

        Args:
            prompt_ids: Tokenized prompt as a list of token IDs.
            insertion_pos: Position at which to insert the dummy token. Dummy tokens
                are inserted immediately before ``prompt_ids[insertion_pos]``.
            n_repeats: Number of copies of the dummy token to insert.

        Returns:
            Tokenized prompt with the dummy tokens inserted.
        """
        prompt_ids_with_insertion = (
            prompt_ids[:insertion_pos]
            + n_repeats * [self.dummy_token_ID]
            + prompt_ids[insertion_pos:]
        )
        return prompt_ids_with_insertion
