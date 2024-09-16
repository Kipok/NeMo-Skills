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
import os
import re
import warnings
from itertools import chain
from math import isclose
from typing import List

import tqdm
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from tqdm.contrib.concurrent import process_map

from nemo_skills.code_execution import CODE_OUTPUT_SEPARATORS, CODE_SEPARATORS

LOG = logging.getLogger(__file__)

PATTERN_ANS = re.compile(r"\\boxed\{([^}]*)\}")
PATTERN_CODE = re.compile(CODE_SEPARATORS[0])

PATTERN_PYTHON_CODE = re.compile("```[pP]ython")


class BaseFilter(BaseParallelProcessor):
    def __init__(self, **kwargs):
        if 'in_memory_chunksize' not in kwargs:
            kwargs['in_memory_chunksize'] = 100000000
        if 'chunksize' not in kwargs:
            kwargs['chunksize'] = 100000
        super().__init__(**kwargs)

    def finalize(self, metrics: List):
        LOG.info("Number of entries after processing: %d", self.number_of_entries)

        if not metrics:
            return

        if 'num_removed' in metrics[0]:
            num_removed_entries = sum(metric.get('num_removed', 0) for metric in metrics)
            LOG.info("Number of removed entries: %d", num_removed_entries)

        if 'num_modified' in metrics[0]:
            num_modified_entries = sum(metric.get('num_modified', 0) for metric in metrics)
            LOG.info("Number of modified entries: %d", num_modified_entries)


class DropMultiBoxed(BaseFilter):

    def __init__(self, solution_key: str = "generation", **kwargs):
        super().__init__(**kwargs)
        self.solution_key = solution_key

    def process_dataset_entry(self, data_entry) -> List:
        if len(PATTERN_ANS.findall(data_entry[self.solution_key])) > 1:
            return [DataEntry(data=None, metrics=dict(num_removed=1))]
        return [DataEntry(data=data_entry, metrics=dict(num_removed=0))]


class DropUselessCode(BaseFilter):

    def __init__(self, solution_key: str = "generation", **kwargs):
        super().__init__(**kwargs)
        self.solution_key = solution_key

    def process_dataset_entry(self, data_entry) -> List:
        ans_match = PATTERN_ANS.search(data_entry[self.solution_key])
        code_match = PATTERN_CODE.search(data_entry[self.solution_key])
        if not ans_match or not code_match or ans_match.start() > code_match.start():
            return [DataEntry(data=None, metrics=dict(num_removed=1))]

        return [DataEntry(data=data_entry, metrics=dict(num_removed=0))]


class DropBrokenCode(BaseFilter):
    def __init__(self, solution_key: str = "generation", **kwargs):
        super().__init__(**kwargs)
        self.solution_key = solution_key

    def process_dataset_entry(self, data_entry) -> List:
        generation = data_entry[self.solution_key]
        code_start_indices = [match.start() for match in re.finditer(CODE_SEPARATORS[0], generation)]
        code_end_indices = [match.start() for match in re.finditer(CODE_SEPARATORS[1], generation)]
        code_out_start_indices = [match.start() for match in re.finditer(CODE_OUTPUT_SEPARATORS[0], generation)]
        code_out_end_indices = [match.start() for match in re.finditer(CODE_OUTPUT_SEPARATORS[1], generation)]

        num_code_occs = set(
            [len(code_start_indices), len(code_end_indices), len(code_out_start_indices), len(code_out_end_indices)]
        )
        if len(num_code_occs) != 1:
            return [DataEntry(data=None, metrics=dict(num_removed=1))]

        if not len(code_end_indices):
            return [DataEntry(data=data_entry, metrics=dict(num_removed=0))]

        for code_start_idx, code_end_idx, code_out_start_idx, code_out_end_idx in zip(
            code_start_indices, code_end_indices, code_out_start_indices, code_out_end_indices
        ):
            if not (code_start_idx < code_end_idx < code_out_start_idx < code_out_end_idx):
                return [DataEntry(data=None, metrics=dict(num_removed=1))]

        return [DataEntry(data=data_entry, metrics=dict(num_removed=0))]


class DropIncorrectCodeBlocks(BaseFilter):
    def __init__(self, solution_key: str = "generation", **kwargs):
        super().__init__(**kwargs)
        self.solution_key = solution_key

    def process_dataset_entry(self, data_entry) -> List:
        if len(PATTERN_PYTHON_CODE.findall(data_entry[self.solution_key])) != 1:
            return [DataEntry(data=None, metrics=dict(num_removed=1))]
        return [DataEntry(data=data_entry, metrics=dict(num_removed=0))]


class MajorityFilter(BaseFilter):
    def __init__(
        self,
        min_majority_votes: int = 0,
        min_majority_percentage: int = 0.0,
        drop_negative_answers: bool = False,
        drop_noninteger_answers: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_majority_votes = min_majority_votes
        self.min_majority_percentage = min_majority_percentage
        self.drop_negative_answers = drop_negative_answers
        self.drop_noninteger_answers = drop_noninteger_answers

    def process_dataset_entry(self, data_entry) -> List:
        majority_votes = data_entry.get("majority_votes", None)
        total_votes = data_entry.get("total_votes", None)
        if majority_votes is None or total_votes is None:
            return [DataEntry(data=data_entry, metrics=dict(num_removed=0))]
        if majority_votes < self.min_majority_votes or majority_votes < total_votes * self.min_majority_percentage:
            return [DataEntry(data=None, metrics=dict(num_removed=1))]

        return [DataEntry(data=data_entry, metrics=dict(num_removed=0))]


class TrimSolutions(BaseFilter):

    def __init__(self, solution_key: str = "generation", **kwargs):
        super().__init__(**kwargs)
        self.solution_key = solution_key

    def process_dataset_entry(self, data_entry) -> List:
        output_lines = data_entry[self.solution_key].split("\n")

        stop_idx = 0
        for idx, soln_line in enumerate(output_lines):
            if PATTERN_ANS.findall(soln_line):
                stop_idx = idx
                break

        if stop_idx < len(output_lines) - 1 and (
            "\\end{align" in output_lines[stop_idx + 1]
            or "\]" in output_lines[stop_idx + 1]
            or "$$" in output_lines[stop_idx + 1]
        ):
            stop_idx = stop_idx + 1

        trimmed_output = "\n".join(output_lines[: stop_idx + 1])
        is_modified = trimmed_output != data_entry[self.solution_key]
        data_entry[self.solution_key] = trimmed_output

        return [DataEntry(data=data_entry, metrics=dict(num_modified=int(is_modified)))]


class CodeTextFilter(BaseParallelProcessor):
    def __init__(self, filter_type, solution_key='generation', **kwargs):
        if 'in_memory_chunksize' not in kwargs:
            kwargs['in_memory_chunksize'] = 100000000
        if 'chunksize' not in kwargs:
            kwargs['chunksize'] = 100000
        super().__init__(**kwargs)
        self.text_filter_type = filter_type
        self.solution_key = solution_key

    def process_dataset_entry(self, grouped_samples: List):
        code_solns = []
        text_solns = []
        for sample in grouped_samples:
            if CODE_SEPARATORS[0] in sample[self.solution_key]:
                code_solns.append(sample)
            else:
                text_solns.append(sample)

        filtered_predictions = []
        if self.text_filter_type is None:
            filtered_predictions.extend(code_solns)
            filtered_predictions.extend(text_solns)
        elif self.text_filter_type == 'all':
            filtered_predictions.extend(code_solns)
        elif self.text_filter_type == 'majority_code':
            filtered_predictions.extend(code_solns)
            if len(code_solns) <= len(grouped_samples) // 2:
                filtered_predictions.extend(text_solns)
        elif self.text_filter_type == 'majority_text':
            if len(code_solns) > len(grouped_samples) // 2:
                filtered_predictions.extend(code_solns)
            else:
                filtered_predictions.extend(text_solns)
        elif self.text_filter_type == 'any_code':
            if code_solns:
                filtered_predictions.extend(code_solns)
            else:
                filtered_predictions.extend(text_solns)
        else:
            raise NotImplementedError(f"Filtering method {self.text_filter_type} not implemented")
        num_removed = len(grouped_samples) - len(filtered_predictions)

        return [DataEntry(data=filtered_predictions, metrics=dict(num_removed=num_removed))]

    def process(self):
        self.prepare()
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)
        metrics = []

        with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
            for manifest_chunk in self._chunk_manifest():
                # this will unroll all inner lists
                data = chain(
                    *process_map(
                        self.process_dataset_entry,
                        manifest_chunk,
                        max_workers=self.max_workers,
                        chunksize=self.chunksize,
                    )
                )
                for data_entry in tqdm.tqdm(data):
                    metrics.append(data_entry.metrics)
                    if data_entry.data is None:
                        continue
                    json.dump(data_entry.data, fout, ensure_ascii=False)
                    self.number_of_entries += 1
                    fout.write("\n")

        self.finalize(metrics)

    def finalize(self, metrics: List):
        LOG.info("Number of entries after processing: %d", self.number_of_entries)

        if not metrics:
            return

        if 'num_removed' in metrics[0]:
            num_removed_entries = sum(metric.get('num_removed', 0) for metric in metrics)
            LOG.info("Number of removed entries: %d", num_removed_entries)
