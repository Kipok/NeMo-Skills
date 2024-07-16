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

import json
import logging
import sys
from collections import Counter
from itertools import zip_longest
from typing import Any

import hydra
from omegaconf import MISSING

from nemo_skills.evaluation.metrics import MathEval, read_predictions
from nemo_skills.utils import get_help_message, nested_dataclass, setup_logging, unroll_files

LOG = logging.getLogger(__file__)


@nested_dataclass
class FillMajorityAnswerConfig:
    """Top-level parameters for the script"""

    # list of files to use for majority voting.
    # Can specify multiple patterns separated by space
    # e.g. "path/to/file1.jsonl path/to/file2.jsonl" or with regex
    # "test_folder/output-rs*.jsonl"
    prediction_jsonl_files: Any = MISSING

    # if set to True will error if any responses/data is missing
    allow_incomplete: bool = False

    # minimum number of majority votes to use the answer.
    # -1 means use half of the votes, which is a good default value
    min_votes: int = -1

    # will be used to fill up when not enough votes are available for the majority
    default_answer: str = "no_answer"

    # will not use any negative answers as this likely indicates bad problems
    # (at least for GSM8K domain). If running with other data, where negative answers
    # are common, should be set to False
    drop_negative_answers: bool = False

    def __post_init__(self):
        """Building data_file from dataset/split_name if not provided directly."""
        if isinstance(self.prediction_jsonl_files, str):
            self.prediction_jsonl_files = self.prediction_jsonl_files.split(" ")

        if self.min_votes < 0:
            self.min_votes = len(self.prediction_jsonl_files) // 2


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_fill_majority_answer_conifg", node=FillMajorityAnswerConfig)


@hydra.main(version_base=None, config_name="base_fill_majority_answer_config")
def fill_majority_answer(cfg: FillMajorityAnswerConfig):
    cfg = FillMajorityAnswerConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    file_handles = [open(file, "rt", encoding="utf-8") for file in unroll_files(cfg.prediction_jsonl_files)]
    # currently majority is only defined for math evals
    evaluator = MathEval()

    majority_answers = []
    all_predictions = []
    for idx, predictions in enumerate(zip_longest(*file_handles)):
        data = read_predictions(predictions, evaluator, cfg.allow_incomplete)
        all_predictions.append(data)
        # TODO: currently majority does not take into account equivalent answers written in a different way
        valid_answers_and_results = [
            (elem['predicted_answer'], elem['is_correct']) for elem in data if elem['predicted_answer'] is not None
        ]
        if len(valid_answers_and_results) == 0:
            majority_answers.append(cfg.default_answer)
            continue
        (majority_answer, _), num_votes = Counter(valid_answers_and_results).most_common(1)[0]

        if num_votes <= cfg.min_votes:
            majority_answers.append(cfg.default_answer)
            continue

        if cfg.drop_negative_answers:
            try:
                majority_answer = float(majority_answer)
            except ValueError:
                majority_answers.append(cfg.default_answer)
                continue
            if majority_answer < 0:
                majority_answers.append(cfg.default_answer)
                continue

        majority_answers.append(majority_answer)

    for file_handle in file_handles:
        file_handle.close()

    # writing the majority answers back to the files
    file_handles = [open(file, "wt", encoding="utf-8") for file in unroll_files(cfg.prediction_jsonl_files)]
    for idx, predictions in enumerate(all_predictions):
        for lidx, handle in enumerate(file_handles):
            predictions[lidx]["expected_answer"] = majority_answers[idx]
            handle.write(json.dumps(predictions[lidx]) + "\n")

    for file_handle in file_handles:
        file_handle.close()


HELP_MESSAGE = get_help_message(FillMajorityAnswerConfig)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        fill_majority_answer()
