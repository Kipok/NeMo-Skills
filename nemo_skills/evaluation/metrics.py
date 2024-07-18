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
import re
import sys
from collections import Counter
from itertools import zip_longest
from pathlib import Path

# adding nemo_skills to python path to avoid requiring installation
sys.path.append(str(Path(__file__).absolute().parents[1]))

from nemo_skills.utils import unroll_files

LOG = logging.getLogger(__file__)


class MathEval:
    def __init__(self):
        self.reset()

    def fill_up_missing(self):
        return {'predicted_answer': None, 'is_correct': False}

    def is_incomplete(self, elem):
        return 'is_correct' not in elem or 'predicted_answer' not in elem

    def update(self, predictions, aggregation_mode):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
            aggregation_mode (str): "best", "majority", "first", etc. Might vary by benchmark.
        """
        # this shouldn't do any heavy calculation, but just read the metric from existing json entry
        # all the heavy lifting should be done in the evaluation script
        self.total += 1
        if aggregation_mode == "best":
            self.total_correct += any([elem['is_correct'] for elem in predictions])
            if all([elem['predicted_answer'] is None for elem in predictions]):
                self.total_no_answer += 1
        elif aggregation_mode == "majority":
            # TODO: currently majority does not take into account equivalent answers written in a different way
            valid_answers_and_results = [
                (elem['predicted_answer'], elem['is_correct'])
                for elem in predictions
                if elem['predicted_answer'] is not None
            ]
            if len(valid_answers_and_results) == 0:
                self.total_no_answer += 1
            else:
                majority_result = Counter(valid_answers_and_results).most_common(1)[0][0]
                self.total_correct += majority_result[1]
        elif aggregation_mode == "first":
            self.total_correct += predictions[0]['is_correct']
            self.total_no_answer += predictions[0]['predicted_answer'] is None
        else:
            raise ValueError(f"Unsupported mode {aggregation_mode}")

    def get_metrics(self):
        return {
            "num_entries": self.total,
            "correct_answer": self.total_correct / self.total * 100.0,
            "wrong_answer": (self.total - self.total_correct - self.total_no_answer) / self.total * 100.0,
            "no_answer": self.total_no_answer / self.total * 100.0,
        }

    def reset(self):
        self.total_correct = 0
        self.total_no_answer = 0
        self.total = 0


class CodeEval:
    def __init__(self):
        self.reset()

    def fill_up_missing(self):
        return {'is_correct': False, 'is_correct-plus': False}

    def is_incomplete(self, elem):
        return 'is_correct' not in elem or 'is_correct-plus' not in elem

    def update(self, predictions, aggregation_mode):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
            aggregation_mode (str): "best", "first", etc. Might vary by benchmark.
        """
        # this shouldn't do any heavy calculation, but just read the metric from existing json entry
        # all the heavy lifting should be done in the evaluation script
        self.total += 1
        if aggregation_mode == "best":
            self.total_correct += any([elem['is_correct'] for elem in predictions])
            self.total_correct_plus += any([elem['is_correct-plus'] for elem in predictions])
        elif aggregation_mode == "first":
            self.total_correct += predictions[0]['is_correct']
            self.total_correct_plus += predictions[0]['is_correct-plus']
        else:
            raise ValueError(f"Unsupported mode {aggregation_mode}")

    def get_metrics(self):
        return {
            "num_entries": self.total,
            "passing_base_tests": self.total_correct / self.total * 100.0,
            "passing_plus_tests": self.total_correct_plus / self.total * 100.0,
        }

    def reset(self):
        self.total_correct = 0
        self.total_correct_plus = 0
        self.total = 0


class IFEval:
    def __init__(self):
        self.reset()

    def fill_up_missing(self):
        return {'loose_eval': {'follow_all_instructions': False}, 'strict_eval': {'follow_all_instructions': False}}

    def is_incomplete(self, elem):
        incomplete = 'loose_eval' not in elem or 'strict_eval' not in elem
        if incomplete:
            return False
        return (
            'follow_all_instructions' not in elem['loose_eval'] or 'follow_all_instructions' not in elem['strict_eval']
        )

    def update(self, predictions, aggregation_mode):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
            aggregation_mode (str): "best", "first", etc. Might vary by benchmark.
        """
        # this shouldn't do any heavy calculation, but just read the metric from existing json entry
        # all the heavy lifting should be done in the evaluation script
        self.total += 1
        if aggregation_mode == "best":
            self.total_correct_loose += any([elem['loose_eval']['follow_all_instructions'] for elem in predictions])
            self.total_correct_strict += any([elem['strict_eval']['follow_all_instructions'] for elem in predictions])
        elif aggregation_mode == "first":
            self.total_correct_loose += predictions[0]['loose_eval']['follow_all_instructions']
            self.total_correct_strict += predictions[0]['strict_eval']['follow_all_instructions']
        else:
            raise ValueError(f"Unsupported mode {aggregation_mode}")

    def get_metrics(self):
        return {
            "num_entries": self.total,
            "strict_accuracy": self.total_correct_strict / self.total * 100.0,
            "loose_accuracy": self.total_correct_loose / self.total * 100.0,
        }

    def reset(self):
        self.total_correct_loose = 0
        self.total_correct_strict = 0
        self.total = 0


class ArenaEval:
    def __init__(self):
        self.reset()

    def _get_judge_score(self, judgment):
        # adapted from https://github.com/lm-sys/arena-hard-auto/blob/main/gen_judgment.py
        pattern = re.compile('\[\[([AB<>=]+)\]\]')
        matches = pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None
        elif len(set(matches)) == 1:
            return matches[0].strip("\n")
        else:
            return None

    def fill_up_missing(self):
        return {'loose_eval': {'follow_all_instructions': False}, 'strict_eval': {'follow_all_instructions': False}}

    def is_incomplete(self, elem):
        incomplete = 'loose_eval' not in elem or 'strict_eval' not in elem
        if incomplete:
            return False
        return (
            'follow_all_instructions' not in elem['loose_eval'] or 'follow_all_instructions' not in elem['strict_eval']
        )

    def update(self, predictions, aggregation_mode):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
            aggregation_mode (str): "best", "first", etc. Might vary by benchmark.
        """
        # this shouldn't do any heavy calculation, but just read the metric from existing json entry
        # all the heavy lifting should be done in the evaluation script
        self.total += 1
        self.scores.append([])
        if aggregation_mode == "best":
            judge_scores = [self._get_judge_score(elem['judgements'][0]) for elem in predictions]
            # adding the best score out of all the generations
            possible_scores = ['A>>B', 'A>B', 'A=B', 'B>A', 'B>>A']
            for possible_score in possible_scores:
                # picking the best available score
                if any([score == possible_score for score in judge_scores]):
                    self.scores[-1].append(possible_score)
                    best_id = judge_scores.index(possible_score)
                    self.lengths += len(predictions[best_id]['generation'])
                    break
            else:
                self.scores[-1].append(None)  # in case judge didn't generate a valid score

            judge_scores = [self._get_judge_score(elem['judgements'][1]) for elem in predictions]
            # second score is grading swapped answers, so we iterate from the end
            for possible_score in possible_scores[::-1]:
                # picking the best available score
                if any([score == possible_score for score in judge_scores]):
                    self.scores[-1].append(possible_score)
                    best_id = judge_scores.index(possible_score)
                    self.lengths += len(predictions[best_id]['generation'])
                    break
            else:
                self.scores[-1].append(None)  # in case judge didn't generate a valid score
        elif aggregation_mode == "first":
            self.lengths += len(predictions[0]['generation'])
            self.scores[-1] = [self._get_judge_score(judgement) for judgement in predictions[0]['judgements']]
        else:
            raise ValueError(f"Unsupported mode {aggregation_mode}")

    def get_metrics(self):
        # run the score aggregation using arena-hard logic
        # currently needs sklearn, which is not ideal, but let's just error-out if it's not installed
        try:
            from nemo_skills.evaluation.arena_utils import get_aggregate_score
        except ImportError:
            raise ImportError(
                "Please install scikit-learn to be able to bootstrap battle results and calculate accurate elo scores"
            )

        metrics = {'num_entries': self.total}
        metrics.update(get_aggregate_score(self.scores))
        metrics['avg_response_length'] = self.lengths / self.total
        return metrics

    def reset(self):
        self.scores = []  # list of lists
        self.lengths = 0
        self.total = 0


def read_predictions(predictions, evaluator, allow_incomplete=False):
    data = []
    for prediction in predictions:
        if not prediction:  # could have missing predictions
            if not allow_incomplete:
                raise RuntimeError("Some data is missing!")
            data.append(evaluator.fill_up_missing())
            continue
        prediction_dict = json.loads(prediction)
        if not prediction_dict:
            if not allow_incomplete:
                raise RuntimeError("Some data is missing!")
            data.append(evaluator.fill_up_missing())
            continue
        if evaluator.is_incomplete(prediction_dict):
            if not allow_incomplete:
                raise RuntimeError("Some data is missing!")
            data.append(evaluator.fill_up_missing())
            continue
        data.append(prediction_dict)

    return data


def compute_metrics(
    prediction_jsonl_files,
    evaluator,
    allow_incomplete=False,
    max_samples=-1,
    aggregation_mode='first',
):
    file_handles = [open(file, "rt", encoding="utf-8") for file in unroll_files(prediction_jsonl_files)]

    evaluator.reset()
    for idx, predictions in enumerate(zip_longest(*file_handles)):
        if idx == max_samples:
            break
        data = read_predictions(predictions, evaluator, allow_incomplete)
        evaluator.update(data, aggregation_mode)

    for file_handle in file_handles:
        file_handle.close()

    return evaluator.get_metrics()
