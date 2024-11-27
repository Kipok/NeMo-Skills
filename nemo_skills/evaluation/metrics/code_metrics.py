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

from nemo_skills.evaluation.metrics.base import BaseMetrics


class CodeMetrics(BaseMetrics):
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
