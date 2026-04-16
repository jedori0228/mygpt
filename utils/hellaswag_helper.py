import os
import json
import torch
from torch.nn import functional as F
from typing import Callable

class hellaswag_helper:

    def __init__(self, HellaSwagRawInputBaseDir: str, encode_func: Callable):

        self.HellaSwagRawInputBaseDir = HellaSwagRawInputBaseDir
        self.enc = encode_func

        self.TotalExample = dict()
        for split in ['val']:
            with open(f"{self.HellaSwagRawInputBaseDir}/hellaswag_{split}.jsonl", "r") as f:
                self.TotalExample[split] = len(f.readlines())

    def iterate_examples(self, split: str):
        # there are 10,042 examples in total in val
        with open(f"{self.HellaSwagRawInputBaseDir}/hellaswag_{split}.jsonl", "r") as f:
            for line in f:
                example = json.loads(line)
                yield example

    def render_example(self, example: dict):

        ctx = example["ctx"]
        label = example["label"]
        endings = example["endings"]

        # data needed to reproduce this eval on the C size
        data = {
            "label": label,
            "ctx_tokens": None,
            "ending_tokens": [],
        }

        # gather up all the tokens
        ctx_tokens = self.enc(ctx)
        data["ctx_tokens"] = ctx_tokens
        tok_rows = []
        mask_rows = []
        for end in endings:
            end_tokens = self.enc(" " + end) # note: prepending " " because GPT-2 tokenizer
            tok_rows.append(ctx_tokens + end_tokens)
            mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
            data["ending_tokens"].append(end_tokens)

        # have to be careful during the collation because the number of tokens in each row can differ
        max_len = max(len(row) for row in tok_rows)
        tokens = torch.zeros((4, max_len), dtype=torch.long)
        mask = torch.zeros((4, max_len), dtype=torch.long)
        for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
            tokens[i, :len(tok_row)] = torch.tensor(tok_row)
            mask[i, :len(mask_row)] = torch.tensor(mask_row)

        return data, tokens, mask, label
    
    def get_most_likely_row(self, tokens, mask, logits):

        # tokens: (Batch=Choices, T=MaxTokens)
        # mask: (Batch=Choices, T=MaxTokens)
        # logits: (Batch=Choices, T=MaxTokens, Vocabs)

        # tokens:
        # Q Q Q A A 0 0 
        # Q Q Q A A A A
        # Q Q Q A A A 0

        # mask:
        # 0 0 0 1 1 0 0 
        # 0 0 0 1 1 1 1
        # 0 0 0 1 1 1 0

        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous() # remove the last token, we don't predict from the last token
        shift_tokens = (tokens[..., 1:]).contiguous() # the truths
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        # Back to tokens: (Batch=Choices, T=MaxTokens)
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        # shift_losses is the cost to predict its next token, so we are interested in the last "Q" token position
        # so we shift the mast left to include the last Q
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, we don't predict the first token of the answer
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred_norm = avg_loss.argmin().item()
        return pred_norm