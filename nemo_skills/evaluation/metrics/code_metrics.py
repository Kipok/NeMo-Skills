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

from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics


class CodeMetrics(BaseMetrics):
    def __init__(self):
        self.reset()

    def fill_up_missing(self):
        return {'is_correct': False, 'is_correct-plus': False}

    def is_incomplete(self, elem):
        return 'is_correct' not in elem or 'is_correct-plus' not in elem

    def _update_perf_dict(self, perf_dict, correct, correct_plus):
        perf_dict["total_correct"] += correct
        perf_dict["total_correct_plus"] += correct_plus

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        self.total += 1

        if len(predictions) > 1:
            correct = any([elem['is_correct'] for elem in predictions])
            correct_plus = any([elem['is_correct-plus'] for elem in predictions])
            self._update_perf_dict(self.agg_mode_dict["best"], correct, correct_plus)
        else:
            correct = predictions[0]['is_correct']
            correct_plus = predictions[0]['is_correct-plus']
            self._update_perf_dict(self.agg_mode_dict["greedy"], correct, correct_plus)

    def get_metrics(self):
        metrics_dict = {}
        for agg_mode, agg_metric_dict in self.agg_mode_dict.items():
            metrics_dict[agg_mode] = {"num_entries": self.total}

            metrics_dict[agg_mode]["passing_base_tests"] = (agg_metric_dict["total_correct"] / self.total) * 100.0
            metrics_dict[agg_mode]["passing_plus_tests"] = (agg_metric_dict["total_correct_plus"] / self.total) * 100.0

        return metrics_dict

    def reset(self):
        self.total = 0
        self.agg_mode_dict = defaultdict(lambda: defaultdict(int))
