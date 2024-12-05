import os
from typing import Dict, List

import numpy as np
import scipy
import torch
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
