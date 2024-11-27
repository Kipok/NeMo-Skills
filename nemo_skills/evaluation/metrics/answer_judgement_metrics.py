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

from collections import Counter

from nemo_skills.evaluation.metrics.base import BaseMetrics
from nemo_skills.evaluation.metrics.utils import is_correct_judgement


class AnswerJudgementMetrics(BaseMetrics):
    def __init__(self):
        self.reset()

    def fill_up_missing(self):
        return {'judgement': "Judgement: No", 'expected_judgement': "Judgement: No"}

    def is_incomplete(self, elem):
        return 'judgement' not in elem or 'expected_judgement' not in elem

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
            is_correct = any(
                [
                    is_correct_judgement(elem['judgement']) == is_correct_judgement(elem['expected_judgement'])
                    for elem in predictions
                ]
            )
            self.total_correct += is_correct
            if not is_correct:
                if is_correct_judgement(predictions[0]['judgement']):
                    self.fp_count += 1
                else:
                    self.fn_count += 1
        elif aggregation_mode == "majority":
            answers = [is_correct_judgement(elem['judgement']) for elem in predictions]
            majority_judgement = Counter(answers).most_common(1)[0]
            is_correct = majority_judgement == is_correct_judgement(predictions[0]['expected_judgement'])
            self.total_correct += is_correct
            if not is_correct:
                if majority_judgement:
                    self.fp_count += 1
                else:
                    self.fn_count += 1
        elif aggregation_mode == "first":
            is_correct = is_correct_judgement(predictions[0]['judgement']) == is_correct_judgement(
                predictions[0]['expected_judgement']
            )
            self.total_correct += is_correct
            if not is_correct:
                if is_correct_judgement(predictions[0]['judgement']):
                    self.fp_count += 1
                else:
                    self.fn_count += 1
        else:
            raise ValueError(f"Unsupported mode {aggregation_mode}")

    def get_metrics(self):
        return {
            "num_entries": self.total,
            "correct_judgements": self.total_correct / self.total * 100.0,
            "false_positives": self.fp_count / self.total * 100.0,
            "false_negatives": self.fn_count / self.total * 100.0,
        }

    def reset(self):
        self.total_correct = 0
        self.fp_count = 0
        self.fn_count = 0
        self.total = 0
