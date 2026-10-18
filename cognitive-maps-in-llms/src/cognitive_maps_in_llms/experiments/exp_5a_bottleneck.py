from pathlib import Path
from typing import Any, Iterable, Mapping, Dict

import numpy as np

from llm_neuroscience_tools.backends.base import LMBackend
from llm_neuroscience_tools.prompts.relational import RelationalPromptRuntime
from llm_neuroscience_tools.base_experiment import TrialState, BaseExperiment

from cognitive_maps_in_llms.tasks.key_value_erasure_tools import load_average_cache
from cognitive_maps_in_llms.tasks.layer_split import resolve_dividing_layer

from .utils import get_answer_token_logits_and_probs


RELATIONS = {'R_AB': 'FWD', 'R_BC': 'FWD'}


class Experiment_5A_Relational_Bottleneck(BaseExperiment):

    def __init__(self,
                 backend: LMBackend,
                 prompt_runtime: RelationalPromptRuntime,
                 n_trials: int,
                 recordings: Path | str,
                 dividing_layer_fraction: float = 0.5):
        """Store handles to experiment components.

        Extra args:
            recordings (Path | str): Directory containing
                ``recordings_metadata.json``, ``K_gstlia.npy``, and
                ``V_gstlia.npy`` for the current prompt/model preset pair.
            dividing_layer_fraction (float): Fractional model depth separating
                the first and second parts of the key-value mosaic.
        """
        self.backend        = backend
        self.prompt_runtime = prompt_runtime
        self.tokenizer      = prompt_runtime.tokenizer
        self.n_trials       = n_trials

        # Extract some useful landmarks
        self.CTXT_START = self.prompt_runtime.landmarks['CTXT_START']
        self.CTXT_END   = self.prompt_runtime.landmarks['CTXT_END']
        self.CONTEXT    = slice(self.CTXT_START, self.CTXT_END+1)

        # Load the average KV cache
        self.KV_data_avg = load_average_cache(
            recordings,
            backend,
            prompt_runtime,
        )
        self.DIVIDING_LAYER = resolve_dividing_layer(
            self.KV_data_avg.K_tlia.shape[1],
            dividing_layer_fraction,
        )

    def define_trial_axes(self) -> Mapping[str, Iterable[Any]]:
        """The trial grid is defined by:
            (1) One axis over the random seed used to sample entities,
            (2) One axis that specifies the kind of bottleneck to apply.
        """
        return {
            'seed': range(self.n_trials),
            'condition': ['K_THEN_V', 'V_THEN_K', 'K_ONLY', 'CTRL']
        }

    def build_trial_state(self, trial_key: Dict[str, Any]) -> TrialState:
        """
        Build mosaic key-value memories designed to induce various kinds of
        "relational bottlenecks".
        """

        seed      = trial_key['seed']
        condition = trial_key['condition']

        # (1) Record the KV cache on a single prompt
        entities  = self.prompt_runtime.sample_entities(seed)
        prompt    = self.prompt_runtime.render_prompt_as_tokens(
            context_entities  = entities,
            context_relations = RELATIONS
        )
        KV_data_WT, _ = self.backend.forward_with_KV_recording(prompt)
        KV_data_mosaic = KV_data_WT.clone()

        # (2) Construct the mosaics

        # Erase keys after the division; values before it
        if condition == 'K_THEN_V':
            KV_data_mosaic.K_tlia[self.CONTEXT, self.DIVIDING_LAYER:, ...] = \
                self.KV_data_avg.K_tlia[self.CONTEXT, self.DIVIDING_LAYER:, ...]
            KV_data_mosaic.V_tlia[self.CONTEXT, :self.DIVIDING_LAYER, ...] = \
                self.KV_data_avg.V_tlia[self.CONTEXT, :self.DIVIDING_LAYER, ...]

        # Erase keys before the division; values after it
        elif condition == 'V_THEN_K':
            KV_data_mosaic.K_tlia[self.CONTEXT, :self.DIVIDING_LAYER, ...] = \
                self.KV_data_avg.K_tlia[self.CONTEXT, :self.DIVIDING_LAYER, ...]
            KV_data_mosaic.V_tlia[self.CONTEXT, self.DIVIDING_LAYER:, ...] = \
                self.KV_data_avg.V_tlia[self.CONTEXT, self.DIVIDING_LAYER:, ...]

        # Erase all values
        elif condition == "K_ONLY":
            KV_data_mosaic.V_tlia[self.CONTEXT, ...] = \
                self.KV_data_avg.V_tlia[self.CONTEXT, ...]

        # Otherwise, don't do anything
        elif condition == 'CTRL':
            pass
        else:
            raise ValueError(f"Unknown condition: {condition!r}")

        return TrialState(
            trial_info={
                'context_entities':  entities,
                'context_relations': RELATIONS},
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
