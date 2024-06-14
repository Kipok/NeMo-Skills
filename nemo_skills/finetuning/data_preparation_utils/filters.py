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
import os
import re
from collections import Counter
from itertools import chain
from typing import List

import tqdm
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from tqdm.contrib.concurrent import process_map

from nemo_skills.code_execution import CODE_OUTPUT_SEPARATORS, CODE_SEPARATORS
from nemo_skills.synthetic_arithmetic.solve_expression import solve_expression

PATTERN_ANS = re.compile(r"\\boxed\{([^}]*)\}")
PATTERN_CODE = re.compile(CODE_SEPARATORS[0])


class DropMultiBoxed(BaseParallelProcessor):

    def __init__(self, solution_key: str = "generation", **kwargs):
        super().__init__(**kwargs)
        self.solution_key = solution_key

    def process_dataset_entry(self, data_entry) -> List:
        if len(PATTERN_ANS.findall(data_entry[self.solution_key])) > 1:
            return [DataEntry(data=None)]
        return [DataEntry(data=data_entry)]


class DropUselessCode(BaseParallelProcessor):

    def __init__(self, solution_key: str = "generation", **kwargs):
        super().__init__(**kwargs)
        self.solution_key = solution_key

    def process_dataset_entry(self, data_entry) -> List:
        ans_match = PATTERN_ANS.search(data_entry[self.solution_key])
        code_match = PATTERN_CODE.search(data_entry[self.solution_key])
        if not ans_match or not code_match or ans_match.start() > code_match.start():
            return [DataEntry(data=None)]

        return [DataEntry(data=data_entry)]


class DropBrokenCode(BaseParallelProcessor):
    def __init__( self, solution_key: str = "generation", **kwargs):
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
            return [DataEntry(data=None)]

        if not len(code_end_indices):
            return [DataEntry(data=data_entry)]

        for code_start_idx, code_end_idx, code_out_start_idx, code_out_end_idx in zip(
            code_start_indices, code_end_indices, code_out_start_indices, code_out_end_indices
        ):
            if not (code_start_idx < code_end_idx < code_out_start_idx < code_out_end_idx):
                return [DataEntry(data=None)]

        return [DataEntry(data=data_entry)]


class TrimSolutions(BaseParallelProcessor):

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
        data_entry[self.solution_key] = trimmed_output

        return [DataEntry(data=data_entry)]


class SplitArithmetic(BaseParallelProcessor):

    def __init__(self, remove_incorrect: bool = False, solution_key: str = "generation", **kwargs):
        super().__init__(**kwargs)
        self.solution_key = solution_key
        self.remove_incorrect = remove_incorrect

    def process_dataset_entry(self, data_entry: str) -> str:
        """
        Extends short arithmetic expressions solutions to step-by-step ones
        For example `1 + 2 + 3 + 4 = 10` -> `1 + 2 + 3 + 4 = 3 + 3 + 4 = 6 + 4 = 10`.
        """
        text = data_entry[self.solution_key]
        new_text = []
        last_end = 0

        for expression, start in self.extract_expressions(text):
            end = start + len(expression)
            parts = expression.split("=")

            if len(parts) != 2:
                new_text.append(text[last_end:end])
                last_end = end
                continue
            expr, ans = parts

            try:
                solution_steps = solve_expression(expr)
            except:
                new_text.append(text[last_end:end])
                last_end = end
                continue

            solution = []
            for step in solution_steps[:-1]:
                solution.append(re.sub(r"(-\d+)", r"(\1)", step))
            solution.append(solution_steps[-1].strip())
            solution = " = ".join(solution)
            solution = re.sub(r"\s+", " ", solution)

            try:
                if eval(solution_steps[-1]) == eval(ans):
                    new_text.append(text[last_end:start] + solution)
                else:
                    new_text.append(text[last_end:end])
                    if self.remove_incorrect:  # skipping solutions with broken math
                        return [DataEntry(data=None)]

                last_end = end
            except KeyboardInterrupt:
                raise
            except:
                new_text.append(text[last_end:end])
                last_end = end

        new_text.append(text[last_end:])
        data_entry[self.solution_key] = "".join(new_text)

        return [DataEntry(data=data_entry)]

    def get_op_counts(self, counter):
        return sum(counter.get(op, 0) for op in "+-/*")

    def extract_expressions(self, text: str):
        start = 0
        cur_expr = []
        for idx, c in enumerate(text):
            prev_len = len(cur_expr)
            if c.isspace():
                if cur_expr:
                    cur_expr.append(c)
            elif c == '.':
                if cur_expr and cur_expr[-1].isdigit():
                    cur_expr.append(c)
                elif cur_expr:
                    result = ''.join(cur_expr)
                    yield result.rstrip(), start
            elif c.isdigit():
                cur_expr.append(c)
            elif c == '=' and not cur_expr:
                continue
            elif c in '+-/*=()':
                cur_expr.append(c)
            else:
                result = ''.join(cur_expr)
                counter = Counter(result)
                if self.get_op_counts(counter) >= 2:
                    yield result.rstrip(), start
                cur_expr = []
            if prev_len == 0 and len(cur_expr) > 0:
                start = idx
        result = ''.join(cur_expr)
        counter = Counter(result)
        if self.get_op_counts(counter) >= 2:
            yield result.rstrip(), start


class CodeTextFilter(BaseParallelProcessor):
    def __init__(self, filter_type, filter_key='generation', **kwargs):
        super().__init__(**kwargs)
        self.text_filter_type = filter_type
        self.filter_key = filter_key

    def process_dataset_entry(self, groupped_samples: List):
        code_solns = []
        text_solns = []
        for sample in groupped_samples:
            if CODE_SEPARATORS[0] in sample[self.filter_key]:
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
            if len(code_solns) <= len(groupped_samples) // 2:
                filtered_predictions.extend(text_solns)
        elif self.text_filter_type == 'majority_text':
            if len(code_solns) > len(groupped_samples) // 2:
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

        return [DataEntry(data=filtered_predictions)]

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
