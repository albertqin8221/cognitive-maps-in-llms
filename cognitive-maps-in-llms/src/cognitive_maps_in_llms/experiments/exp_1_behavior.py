from typing import Any, Iterable, Mapping, Dict

import numpy as np

from llm_neuroscience_tools.backends.base import LMBackend
from llm_neuroscience_tools.prompts.relational import RelationalPromptRuntime
from llm_neuroscience_tools.base_experiment import TrialState, BaseExperiment

from .utils import get_answer_token_logits_and_probs


class Experiment_1_Behavior(BaseExperiment):

    RELATION_PATTERNS = {
        "A --fwd--> B --fwd--> C": {"R_AB": "FWD", "R_BC": "FWD"},
        "A --rev--> B --fwd--> C": {"R_AB": "REV", "R_BC": "FWD"},
        "A --fwd--> B --rev--> C": {"R_AB": "FWD", "R_BC": "REV"},
        "A --rev--> B --rev--> C": {"R_AB": "REV", "R_BC": "REV"},
    }

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

    def define_trial_axes(self) -> Mapping[str, Iterable[Any]]:
        """The trial grid is defined by:
            (1) One axis over the random seed used to sample entities,
            (2) One axis over the relation pattern.
        """
        return {
            'seed': range(self.n_trials),
            'structure': tuple(self.RELATION_PATTERNS)
        } 

    def build_trial_state(self, trial_key: Dict[str, Any]) -> TrialState:
        """For behavior, we only need to draw a set of names. 
        """
        entities = self.prompt_runtime.sample_entities(seed=trial_key['seed'])
        relations = self.RELATION_PATTERNS[trial_key['structure']]
        return TrialState(
            trial_info={
                'context_entities': entities, 
                'context_relations': relations},
            tensor_data=None,
        )
    
    def build_trial_queries(self, trial_state: TrialState) -> Mapping[str, Any]:
        """Systematically generate all queries. 
        """
        return self.prompt_runtime.build_query_specs(
            entity_symbols_to_names = trial_state.trial_info["context_entities"]
        )
    
    def run_query(self, trial_state: TrialState, query: Any) -> np.ndarray:
        """Construct a prompt and run a forward pass.
        """
        prompt = self.prompt_runtime.render_prompt_as_tokens(
            context_entities  = trial_state.trial_info['context_entities'], 
            context_relations = trial_state.trial_info['context_relations'],
            query_entities    = query['query_entities'],
            query_relations   = query['query_relations'],
        )
        return self.backend.forward(prompt)
    
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
