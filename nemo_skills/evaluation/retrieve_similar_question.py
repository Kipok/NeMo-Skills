# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import glob
import json
import logging
import pdb
import sys
from collections import Counter
from itertools import zip_longest
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import MISSING
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from nemo_skills.evaluation.metrics import MathEval, read_predictions
from nemo_skills.utils import get_help_message, nested_dataclass, setup_logging

LOG = logging.getLogger(__file__)


def unroll_files(prediction_jsonl_files):
    files = []
    for file_pattern in prediction_jsonl_files:
        matched_files = sorted(glob.glob(file_pattern, recursive=True))
        files.extend(matched_files)
    return files


def top_k_similarity(train_embs, test_embs, top_k):
    # Compute cosine-similarities
    cosine_scores = util.cos_sim(test_embs, train_embs)
    # Find the top-k most similar train_embs for each test_emb
    top_k_indices = torch.topk(cosine_scores, k=top_k, dim=1).indices

    return top_k_indices


def bert_encode(model, data, batch_size=32, device=None):
    return model.encode(data, batch_size=batch_size, show_progress_bar=True, device=device)


def read_dataset_question(r_path):
    with open(r_path, 'r', encoding='utf-8') as file:
        return [json.loads(line)['question'] for line in file]


def read_dataset(r_path):
    with open(r_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


@nested_dataclass
class LLMDecontaminatorConfig:
    """Top-level parameters for the script"""

    # list of files to use for llm contamination detection.
    # Can specify multiple patterns separated by space
    # e.g. "path/to/file1.jsonl path/to/file2.jsonl" or with regex
    # "train_folder/output-rs*.jsonl"
    train_jsonl_files: Any = MISSING

    # "test_folder/output-rs*.jsonl"
    test_jsonl_files: Any = MISSING

    ### the model used to compute embedding, default is sentence transformer
    model: str = 'multi-qa-MiniLM-L6-cos-v1'

    ### the number of devices to coompute embedding
    device: Any = MISSING

    ### for each question in the test data set, we retreive the top_k similar questions from the train_jsonl_files
    top_k: int = 3

    batch_size: int = 32

    def __post_init__(self):
        """Building data_file from dataset/split_name if not provided directly."""
        if isinstance(self.train_jsonl_files, str):
            self.train_jsonl_files = self.train_jsonl_files.split(" ")

        # if isinstance(self.test_jsonl_files, str):
        #     self.test_jsonl_files = self.test_jsonl_files.split(" ")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_llm_decontaminator_detect_conifg", node=LLMDecontaminatorConfig)


@hydra.main(version_base=None, config_name="base_llm_decontaminator_detect_conifg")
def compute_embedding(cfg: LLMDecontaminatorConfig):
    cfg = LLMDecontaminatorConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    file_handles = unroll_files(cfg.train_jsonl_files)

    model = SentenceTransformer(cfg.model)
    test_cases = read_dataset_question(cfg.test_jsonl_files)
    test_embs = bert_encode(model, test_cases, batch_size=cfg.batch_size, device=cfg.device)
    db_embedding = collections.defaultdict(list)
    db_questions = collections.defaultdict(list)
    for file_path in file_handles:
        train_cases = read_dataset_question(file_path)
        train_embs = bert_encode(model, train_cases, batch_size=cfg.batch_size, device=cfg.device)
        top_k_indices = top_k_similarity(train_embs, test_embs, cfg.top_k)

        for i, test_case in enumerate(test_cases):
            #### for each file in the training, we will find the top_k similar examples
            top_k_embedding_per_file = [train_embs[index] for index in top_k_indices[i]]
            top_k_questions_per_file = [train_cases[index] for index in top_k_indices[i]]
            db_embedding[i].extend(top_k_embedding_per_file)
            db_questions[i].extend(top_k_questions_per_file)

    test_file = read_dataset(cfg.test_jsonl_files)
    for i, test_case in enumerate(test_cases):
        test_embs_i = test_embs[i].reshape(1, len(test_embs[i]))
        db_embedding[i] = np.array(db_embedding[i])
        #### we will select top_k similar examples from top_k * len(training files)
        top_k_indices = top_k_similarity(db_embedding[i], test_embs_i, cfg.top_k)

        similar_exmaple = []
        for index in top_k_indices.tolist()[0]:
            #### get source file index, suppose we have 10 files, and top_k = 2
            #### then index belong [0, 19], so index // top_k belong [0, 9]
            train_file_index = index // cfg.top_k
            ##### db_questions[i] stores top_k*10 similar question, and we can directly retrevie the question
            pair = {'source': file_handles[train_file_index], 'question': db_questions[i][index]}
            similar_exmaple.append(pair)
        test_file[i]['top_k_similar_example'] = similar_exmaple

    #### we will write back to the orginal files
    with open(cfg.test_jsonl_files, 'w', encoding='utf-8') as file:
        for entry in test_file:
            #### currently we only support top_1 as the candate
            entry['candidate'] = entry['top_k_similar_example'][0]['question']
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')


HELP_MESSAGE = get_help_message(LLMDecontaminatorConfig)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        compute_embedding()
