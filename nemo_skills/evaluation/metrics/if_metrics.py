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


class IFMetrics(BaseMetrics):
    # loosely adapted from
    # https://github.com/google-research/google-research/blob/master/instruction_following_eval/evaluation_main.py

    required_keys = ['follow_instruction_list', 'instruction_id_list']

    def __init__(self):
        self.reset()

    def fill_up_missing(self):
        return {
            'loose_eval': {key: [] for key in self.required_keys},
            'strict_eval': {key: [] for key in self.required_keys},
        }

    def is_incomplete(self, elem):
        incomplete = 'loose_eval' not in elem or 'strict_eval' not in elem
        if incomplete:
            return True

        if any([key not in elem['loose_eval'] for key in self.required_keys]):
            return True

        if any([key not in elem['strict_eval'] for key in self.required_keys]):
            return True

        return False

    def _update_single_stat(self, stats_dict, elems):
        """Will update using the pass@k strategy (just pass a single-element list to get greedy)."""
        # has to be the same across all elements as they are solutions for the same question
        instruction_id_list = elems[0]['instruction_id_list']
        # computing "pass@k" score
        follow_instruction_list = elems[0]['follow_instruction_list']
        for elem in elems:
            follow_instruction_list = [
                follow_instruction_list[i] or elem['follow_instruction_list'][i]
                for i in range(len(follow_instruction_list))
            ]

        stats_dict['prompt']['total'] += 1
        if all(follow_instruction_list):
            stats_dict['prompt']['correct'] += 1

        stats_dict['instruction']['total'] += len(instruction_id_list)
        stats_dict['instruction']['correct'] += sum(follow_instruction_list)

        for instruction_id, followed_or_not in zip(instruction_id_list, follow_instruction_list):
            instruction_id = instruction_id.split(":")[0]
            stats_dict['tier0']['total'][instruction_id] += 1
            if followed_or_not:
                stats_dict['tier0']['correct'][instruction_id] += 1

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        # this shouldn't do any heavy calculation, but just read the metric from existing json entry
        # all the heavy lifting should be done in the evaluation script
        if aggregation_mode == "best":
            self._update_single_stat(self.strict_stats, [pred['strict_eval'] for pred in predictions])
            self._update_single_stat(self.loose_stats, [pred['loose_eval'] for pred in predictions])
        elif aggregation_mode == "first":
            self._update_single_stat(self.strict_stats, [predictions[0]['strict_eval']])
            self._update_single_stat(self.loose_stats, [predictions[0]['loose_eval']])
        else:
            raise ValueError(f"Unsupported mode {aggregation_mode}")

    def get_metrics(self):
        prompt_total = self.strict_stats['prompt']['total']
        inst_total = self.strict_stats['instruction']['total']
        prompt_strict = self.strict_stats['prompt']['correct'] / prompt_total * 100.0
        inst_strict = self.strict_stats['instruction']['correct'] / inst_total * 100.0
        prompt_loose = self.loose_stats['prompt']['correct'] / prompt_total * 100.0
        inst_loose = self.loose_stats['instruction']['correct'] / inst_total * 100.0
        return {
            "num_prompts": prompt_total,
            "num_instructions": inst_total,
            "average_score": (prompt_strict + inst_strict + prompt_loose + inst_loose) / 4,
            "prompt_strict_accuracy": prompt_strict,
            "instruction_strict_accuracy": inst_strict,
            "prompt_loose_accuracy": prompt_loose,
            "instruction_loose_accuracy": inst_loose,
        }

    def reset(self):
        # the original code also has a deeper breakdown into tier1 scores,
        # but that's probably too much for us to track at this stage
        self.strict_stats = {
            "prompt": {"total": 0, "correct": 0},
            "instruction": {"total": 0, "correct": 0},
            "tier0": {"total": defaultdict(int), "correct": defaultdict(int)},
        }
        self.loose_stats = {
            "prompt": {"total": 0, "correct": 0},
            "instruction": {"total": 0, "correct": 0},
            "tier0": {"total": defaultdict(int), "correct": defaultdict(int)},
        }
