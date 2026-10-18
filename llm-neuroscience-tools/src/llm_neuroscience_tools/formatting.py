import warnings

import torch
import numpy as np
import pandas as pd


class OutputFormatter:
    """Contains a variety of methods to retrieve pretty / interpretable 
    outputs from raw logits.
    """

    def __init__(self, tokenizer):
        """Use the caller's tokenizer to interpret prompts and model outputs."""
        if tokenizer is None:
            raise ValueError("OutputFormatter requires a tokenizer.")
        self.tokenizer = tokenizer

    def get_logit(self, token: str, logits: np.ndarray) -> float:
        """Return the logit associated with a single token string."""
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        return float(logits[token_id].item())

    def get_prob(self, token: str, logits: np.ndarray) -> float:
        """Return the softmax probability associated with a single token string."""
        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        return float(probabilities[token_id].item())
    
    def get_pretty_top_k(self, logits: np.ndarray, k: int = 10) -> str:
        """Given the output logits with shape [N_vocab,], renders the top k tokens.

        Returns a rendered table as a string.
        """
        sorted_output_ids = np.argsort(logits)[::-1]        

        # construct columns to pretty print
        ranks_col  = []
        tokens_col = []
        logits_col = []
        for i in range(k):
            ranks_col.append(i+1)
            tokens_col.append(repr(self.tokenizer.decode(sorted_output_ids[i])).replace(' ', '_'))
            logits_col.append(logits[sorted_output_ids[i]])

        # render in a pretty way with pandas
        dataframe = pd.DataFrame({
            'rank':         ranks_col,
            'token':        tokens_col,
            'output_logit': logits_col
        })

        s = dataframe.to_string(index=False, justify='center', col_space=8)
        lines = s.splitlines()
        underline = "-" * len(lines[0])
        pretty = "\n".join([lines[0], underline, *lines[1:]])
            
        return pretty

    def get_prompt_text(self, prompt_ids: list[int]) -> str:
        """Given a tokenized prompt (list of ints), return the decoded text."""
        return self.tokenizer.decode(prompt_ids)

    def get_pretty_prompt(self, 
                          prompt_ids: list[int], 
                          landmarks: None | dict = None) -> str:
        """Given a tokenized prompt (list of ints), renders the prompt as a 
        pretty table. 

        Returns: The rendered table as a string.
        """

        # construct the table
        tokens_col  = []
        tags_col    = []
        indices_col = []
        for i, tok_id in enumerate(prompt_ids):
            # for each entry like ['A1': i], copy 'A1' to `these_tags`
            if landmarks:
                these_tags = [key for key, pos in landmarks.items() if pos == i]
                tags_col.append(', '.join(these_tags) if len(these_tags) != 0 else '_')
            else:
                tags_col.append('_')

            # update the other columns
            tokens_col.append(repr(self.tokenizer.decode(tok_id)).replace(' ', '_'))
            indices_col.append(i)

        # render the table in a pretty way with pandas
        dataframe = pd.DataFrame({
            'tok_index': indices_col,
            'token': tokens_col,
            'token_ID': prompt_ids,
            'tag': tags_col
        })
        s = dataframe.to_string(index=False, justify='center', col_space=12)
        lines = s.splitlines()
        underline = "-" * len(lines[0])
        pretty = "\n".join([lines[0], underline, *lines[1:]])
        
        return pretty

    def get_context_logits(self,
                           context: dict[str, str],
                           logits: np.ndarray) -> dict:
        """Deprecated: use `get_logit()` instead."""
        warnings.warn(
            "get_context_logits() is deprecated; use get_logit() or a "
            "small wrapper over it instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return {
            symb: self.get_logit(token, logits)
            for symb, token in context.items()
        }
    
    def get_context_probs(self,
                          context: dict[str, str],
                          logits: np.ndarray) -> dict:
        """Deprecated: use `get_prob()` instead."""
        warnings.warn(
            "get_context_probs() is deprecated; use get_prob() or a "
            "small wrapper over it instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return {
            symb: self.get_prob(token, logits)
            for symb, token in context.items()
        }

    def get_pretty_context_logits(self, context: dict[str, str], logits: np.ndarray) -> str:
        """Deprecated: pretty-print labeled token logits/probabilities."""
        warnings.warn(
            "get_pretty_context_logits() is deprecated; prefer using "
            "get_logit()/get_prob() and formatting locally.",
            DeprecationWarning,
            stacklevel=2,
        )
        context_logits = {
            symb: self.get_logit(token, logits)
            for symb, token in context.items()
        }
        context_probs = {
            symb: self.get_prob(token, logits)
            for symb, token in context.items()
        }

        symbols_col = []
        names_col   = []
        logits_col  = []
        probs_col = []
        for symb in sorted(context.keys()):
            names_col.append(context[symb])
            logits_col.append(context_logits[symb])
            probs_col.append(f'{context_probs[symb]:.3f}')
            symbols_col.append(symb)

        dataframe = pd.DataFrame({
            'symbol': symbols_col,
            'name': names_col,
            'logit': logits_col,
            'probability': probs_col
        })
        s = dataframe.to_string(index=False, justify='center', col_space=12)
        lines = s.splitlines()
        underline = "-" * len(lines[0])
        pretty = "\n".join([lines[0], underline, *lines[1:]])

        return pretty
