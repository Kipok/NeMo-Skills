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

import logging
import sys
from itertools import zip_longest
from typing import Any

import hydra
from omegaconf import MISSING

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

    # for idx, lines in enumerate(zip_longest(*file_handles)):
    #     data = []
    #     for line in lines:
    #         if not line:  # could have missing predictions
    #             if not cfg.allow_incomplete:
    #                 raise RuntimeError("Some data is missing!")
    #             data.append(evaluator.fill_up_missing())
    #             continue
    #         line_dict = json.loads(line)
    #         if not line_dict:
    #             if not allow_incomplete:
    #                 raise RuntimeError("Some data is missing!")
    #             data.append(evaluator.fill_up_missing())
    #             continue
    #         if evaluator.is_incomplete(line_dict):
    #             if not allow_incomplete:
    #                 raise RuntimeError("Some data is missing!")
    #             data.append(evaluator.fill_up_missing())
    #             continue
    #         data.append(line_dict)

    #     evaluator.update(data, aggregation_mode)

    # for file_handle in file_handles:
    #     file_handle.close()

    # return evaluator.get_metrics()

    #     # TODO: currently majority does not take into account equivalent answers written in a different way
    #     valid_answers_and_results = [
    #         (ans, is_correct) for ans, is_correct in zip(pred_answers, evaluations) if ans is not None
    #     ]
    #     if len(valid_answers_and_results) == 0:
    #         total_no_answer += 1
    #         majority_answers.append("None123")
    #     else:
    #         majority_result = Counter(valid_answers_and_results).most_common(1)[0]
    #         try:
    #             majority_answer = float(majority_result[0][0])
    #         except:
    #             majority_answer = -1
    #         if majority_result[1] <= 10 or majority_answer < 0:
    #             majority_answers.append("None123")  # need to not match any of the answers
    #         else:
    #             if majority_answer != int(majority_answer):
    #                 print(majority_result, idx)
    #             majority_answers.append(int(majority_answer))
    #         total_correct += majority_result[0][1]

    #     file_handles = [open(file, "wt", encoding="utf-8") for file in unroll_files(prediction_jsonl_files)]
    #     for idx, cur_data in enumerate(all_data):
    #         if idx == max_samples:
    #             break
    #         for lidx, handle in enumerate(file_handles):
    #             line = cur_data[lidx]
    #             if not line:  # could have missing predictions
    #                 continue
    #             line_dict = json.loads(line)
    #             if not line_dict:
    #                 handle.write(line)
    #                 continue
    #             line_dict["expected_answer"] = majority_answers[idx]
    #             handle.write(json.dumps(line_dict) + "\n")

    #     for file_handle in file_handles:
    #         file_handle.close()


HELP_MESSAGE = get_help_message(FillMajorityAnswerConfig)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        fill_majority_answer()
