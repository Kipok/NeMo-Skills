import os
from typing import Dict, List

import numpy as np
import scipy
import torch
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.core import Dataset
from nemo.utils import logging
from omegaconf import OmegaConf


class OutcomeRewardModelDataset(Dataset):
    """This class works only with jsonl files. It assumes each line of the json file is a dictionary
    with the prompt, along with the response (response only, no prompt), and the status denoting whether the response is
    chosen or rejected. This Dataset will combine the prompt with the corresponding response, and then tokenize it. It
    will also create a score field that has 1 if the sample is chosen and 0 if rejected. It also returns the labels for
    each, which is the response tokens with -100 for the prompt part.
    """

    def __init__(
        self,
        cfg,
        tokenizer,
        name,
        data_prefix,
        documents,
        data,
        seq_length,
        seed,
        drop_last=True,
    ):
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.data = data
        self.drop_last = drop_last
        self.seq_length = seq_length
        self.tokenizer = tokenizer

        self.reset_position_ids = cfg.data.get("reset_position_ids", False)
        self.reset_attention_mask = cfg.data.get("reset_attention_mask", False)
        self.eod_mask_loss = cfg.data.get("eod_mask_loss", False)
        self.eos_id = tokenizer.eos_id

        np_rng = np.random.default_rng(seed=seed)
        np_rng.shuffle(self.data)

        self.nograd_length = 32

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < len(self.data)

    def __len__(self):
        return len(self.data)

    def encode(self, text, append_eod=False):
        text_ids = self.tokenizer.text_to_ids(text)

        if len(text_ids) > 0 and append_eod:
            text_ids.append(self.tokenizer.eos_id)

        return text_ids, len(text_ids)

    def __getitem__(self, idx):
        """Returns a sample = prompt + response, their respective lengths, and labels."""
        payload = self.data[idx]
        prompt, prompt_len = self.encode(payload["prompt"])
        sample, sample_len = self.encode(payload["prompt"] + payload["response"])
        labels = ([-100] * prompt_len) + sample[prompt_len:]
        # Separate the response from the prompt
        response = sample[prompt_len:]
        preference = 1 if payload["preference"] == "chosen" else 0

        assert (
            sample[0:prompt_len] == prompt
        ), "the tokenizer for OutcomeRewardModel has merged tokens between prompt and response"

        if sample_len > self.seq_length:
            logging.warning(
                f"WARNING: Tokenized text exceeds max seq length ({sample_len} vs {self.seq_length})."
                + f"The example will be ignored."
            )
            # Truncate the sample and labels to the first nograd_length tokens
            sample_len = self.nograd_length
            sample = sample[: self.nograd_length]
            prompt_len = self.nograd_length // 2
            prompt = prompt[:prompt_len]
            response = sample[prompt_len:]
            labels = torch.ones_like(torch.LongTensor(sample)) * (-100)

        output = {
            "prompt_tokens": torch.LongTensor(prompt),
            "response_tokens": torch.LongTensor(response),
            "sample_length": sample_len,
            "sample_labels": torch.LongTensor(labels),
            "preference": preference,
        }
        return output


def custom_collate(batch, eos_id, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False):
    sample_tokens = [torch.cat((item["prompt_tokens"], item["response_tokens"]), dim=0) for item in batch]
    sample_lengths = torch.LongTensor([item["sample_length"] for item in batch])
    sample_labels = [item["sample_labels"] for item in batch]
    sample_preference = torch.tensor([item["preference"] for item in batch])

    sample_tokens = torch.nn.utils.rnn.pad_sequence(sample_tokens, batch_first=True, padding_value=eos_id)
    sample_labels = torch.nn.utils.rnn.pad_sequence(sample_labels, batch_first=True, padding_value=-100)

    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        sample_tokens,
        eos_id,
        reset_position_ids,
        reset_attention_mask,
        eod_mask_loss,
    )
    assert attention_mask.ndim == 4, "attention_mask is incorrect shape for outcome-based custom_collate"
    if attention_mask.shape[0] == 1:
        # using .expand() here causes errors from pin_memory=True, so need to use .repeat()
        # attention_mask = attention_mask.expand(len(batch), *((-1,) * (len(attention_mask.shape) - 1)))
        attention_mask = attention_mask.repeat(len(batch), *((1,) * (len(attention_mask.shape) - 1)))

    output = {
        "samples": sample_tokens,
        "sample_length": sample_lengths,
        "sample_labels": sample_labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "preference": sample_preference,
    }
    return output
