from typing import Any, Iterable, Mapping, Dict

import numpy as np

from llm_neuroscience_tools.backends.base import LMBackend
from llm_neuroscience_tools.prompts.relational import RelationalPromptRuntime
from llm_neuroscience_tools.base_experiment import TrialState, BaseExperiment

from .utils import (
    get_answer_token_logits_and_probs,
    build_context_patching_mask,
)


class Experiment_4_Retrospective_Prospective(BaseExperiment):

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
        """Store handles to experiment components."""
        self.backend        = backend
        self.prompt_runtime = prompt_runtime
        self.tokenizer      = prompt_runtime.tokenizer
        self.n_trials       = n_trials

        # Extract the token span corresponding to C and the B-C relation
        target_entity_landmark = "C1"
        target_relation_symbol = "R_BC"
        if target_entity_landmark not in self.prompt_runtime.landmarks:
            raise ValueError(
                f"Target entity landmark {target_entity_landmark!r} is not "
                "defined by the prompt runtime."
            )
        self.entity_tokens = [
            self.prompt_runtime.landmarks[target_entity_landmark]
        ]
        self.relation_tokens = [
            token_idx
            for landmark, token_idx in self.prompt_runtime.landmarks.items()
            if (
                landmark == target_relation_symbol
                or landmark.startswith(f"{target_relation_symbol}.")
            )
        ]
        if not self.relation_tokens:
            raise ValueError(
                f"Target relation symbol {target_relation_symbol!r} has no "
                "token landmarks in the prompt runtime."
            )
        if min(self.relation_tokens) >= self.entity_tokens[0]:
            raise ValueError(
                f"Target relation symbol {target_relation_symbol!r} must occur "
                f"before target entity landmark {target_entity_landmark!r}."
            )
        self.entity_relation_span_tokens = range(
            min(self.relation_tokens), 
            self.entity_tokens[0] + 1
        )
        self.patching_mask = build_context_patching_mask(backend, prompt_runtime)

    def define_trial_axes(self) -> Mapping[str, Iterable[Any]]:
        """The trial grid defined by:
            (1) One axis over the random seed used to sample entities,
            (2) One axis over the structure of the donor and control contexts,
            (3) One axis over the region the activation tensor to patch.
        """
        return {
            'seed': range(self.n_trials),
            'structure_pair': (
                {
                    "donor": "A --rev--> B --fwd--> C",
                    "control": "A --fwd--> B --fwd--> C",
                },
                {
                    "donor": "A --fwd--> B --fwd--> C",
                    "control": "A --rev--> B --fwd--> C",
                },
                {
                    "donor": "A --fwd--> B --rev--> C",
                    "control": "A --rev--> B --rev--> C",
                },
                {
                    "donor": "A --rev--> B --rev--> C",
                    "control": "A --fwd--> B --rev--> C",
                },
            ),
            'condition': ('CTRL', 'ENTITY', 'RELATION', 'ENTITY_RELATION_SPAN')
        }

    def build_trial_state(self, trial_key: Dict[str, Any]) -> TrialState:

        seed           = trial_key['seed']
        structure_pair = trial_key['structure_pair']
        condition      = trial_key['condition']

        # (1) Build the control and donor prompts and record
        entities = self.prompt_runtime.sample_entities(seed)
        relations_control = self.RELATION_PATTERNS[structure_pair["control"]]
        relations_donor   = self.RELATION_PATTERNS[structure_pair["donor"]]
        prompt_control = self.prompt_runtime.render_prompt_as_tokens(
            context_entities  = entities,
            context_relations = relations_control
        )
        prompt_donor    = self.prompt_runtime.render_prompt_as_tokens(
            context_entities  = entities,
            context_relations = relations_donor
        )
        Z_tla_donor, _ = self.backend.forward_with_recording(prompt_donor)
        Z_tla_control, _ = self.backend.forward_with_recording(prompt_control)

        # (2) Build the mosaic tensor
        Z_tla_mosaic = Z_tla_control
        if condition == 'CTRL':
            pass
        elif condition == 'ENTITY':
            Z_tla_mosaic[self.entity_tokens, ...] = \
                Z_tla_donor[self.entity_tokens, ...]
        elif condition == 'RELATION':
            Z_tla_mosaic[self.relation_tokens, ...] = \
                Z_tla_donor[self.relation_tokens, ...]
        elif condition == 'ENTITY_RELATION_SPAN':
            Z_tla_mosaic[self.entity_relation_span_tokens, ...] = \
                Z_tla_donor[self.entity_relation_span_tokens, ...]
        else:
            raise ValueError(f"Unknown condition: {condition!r}")

        return TrialState(
            trial_info={
                'context_entities':  entities, 
                'context_relations': relations_control,
                'context_relations_donor': relations_donor
            },
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
