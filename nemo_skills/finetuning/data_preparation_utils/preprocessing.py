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
import random
from collections import defaultdict
from itertools import chain, zip_longest
from typing import Dict, Optional

import tqdm
from sdp.processors.base_processor import BaseProcessor

from nemo_skills.inference.prompt.utils import Prompt, get_prompt_config
from nemo_skills.utils import unroll_files

LOG = logging.getLogger(__file__)


class ReadData(BaseProcessor):

    def __init__(
        self,
        prediction_jsonl_files: Optional[str] = None,
        preprocessed_dataset_files: Optional[str] = None,
        skip_first: int = 0,
        add_correct: bool = True,
        add_incorrect: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prediction_jsonl_files = prediction_jsonl_files
        self.preprocessed_dataset_files = preprocessed_dataset_files
        self.skip_first = skip_first
        self.add_correct = add_correct
        self.add_incorrect = add_incorrect

        if isinstance(self.prediction_jsonl_files, str):
            self.prediction_jsonl_files = self.prediction_jsonl_files.split(" ")

        if isinstance(self.preprocessed_dataset_files, str):
            self.preprocessed_dataset_files = self.preprocessed_dataset_files.split(" ")

        if self.prediction_jsonl_files is None and self.preprocessed_dataset_files is None:
            raise ValueError("Either `prediction_jsonl_files` or `preprocessed_dataset_files` should be provided")

        if not self.add_correct and not self.add_incorrect:
            raise ValueError("At least one of `add_correct` and `add_incorrect` should be True")

    def _read_preprocessed_data(self) -> int:
        samples = []
        questions = set()
        for file_path in self.preprocessed_dataset_files:
            with open(file_path, "rt", encoding="utf-8") as file_handle:
                for line in tqdm.tqdm(file_handle):
                    sample = json.loads(line)
                    questions.add(sample["question"])
                    # for backward compatibility
                    if "generation" not in sample and "generated_solution" in sample:
                        sample["generation"] = sample.pop("generated_solution")
                    samples.append(sample)

        return samples

    def _read_raw_data(self) -> int:
        samples = []
        questions = set()
        file_handles = [
            open(manifest, "rt", encoding="utf-8") for manifest in unroll_files(self.prediction_jsonl_files)
        ]
        for idx, lines in tqdm.tqdm(enumerate(zip_longest(*file_handles))):
            if idx < self.skip_first:
                continue

            seen_predictions = {}
            for lidx, file_line in enumerate(lines):
                # if different files have different number of lines
                if file_line is None:
                    continue
                line_dict = json.loads(file_line)
                # can be empty for incomplete generations
                if not line_dict:
                    continue

                questions.add(line_dict["question"])
                if line_dict["question"] not in seen_predictions:
                    seen_predictions[line_dict["question"]] = set()

                # skipping any incomplete generations
                if "is_correct" not in line_dict:
                    LOG.warning("Found incomplete generations (is_correct field is missing) - skipping")
                    continue

                if not self.add_correct and line_dict["is_correct"]:
                    continue

                if not self.add_incorrect and not line_dict["is_correct"]:
                    continue

                # for backward compatibility
                if "generation" not in line_dict and "generated_solution" in line_dict:
                    line_dict["generation"] = line_dict.pop("generated_solution")

                if line_dict["generation"] in seen_predictions[line_dict["question"]]:
                    continue

                seen_predictions[line_dict["question"]].add(line_dict["generation"])
                line_dict['filename'] = file_handles[lidx].name
                samples.append(line_dict)

        for handle in file_handles:
            handle.close()

        return samples

    def process(self):
        samples = []
        if self.prediction_jsonl_files:
            samples.extend(self._read_raw_data())
        if self.preprocessed_dataset_files:
            samples.extend(self._read_preprocessed_data())
        with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
            for sample in samples:
                fout.write(json.dumps(sample) + "\n")


class GroupSamples(BaseProcessor):
    def __init__(self, group_key='question', **kwargs):
        super().__init__(**kwargs)
        self.group_key = group_key

    def process(self):
        samples = defaultdict(list)
        with open(self.input_manifest_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                sample = json.loads(line)
                samples[sample[self.group_key]].append(sample)

        with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
            for groupped_samples in samples.values():
                fout.write(json.dumps(groupped_samples) + "\n")


class ShuffleAndDownsampleData(BaseProcessor):
    def __init__(
        self,
        random_seed: int,
        do_shuffle: bool,
        num_samples: Optional[int] = None,
        sampling_method: str = 'random',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_method = sampling_method
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.do_shuffle = do_shuffle

    def process(self):
        groupped_samples = []
        with open(self.input_manifest_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                samples = json.loads(line)
                groupped_samples.append(samples)

        random.seed(self.random_seed)
        if self.sampling_method == "random":
            output_instances = list(chain(*groupped_samples))
            if self.do_shuffle:
                random.shuffle(output_instances)
            output_instances = output_instances[: self.num_samples]
        elif self.sampling_method == "fair":
            soln_counter = 0
            output_instances = []
            while self.num_samples is not None:
                for quesn_idx in range(len(groupped_samples)):
                    if len(output_instances) == self.num_samples:
                        break
                    if len(groupped_samples[quesn_idx]) > soln_counter:
                        output_instances.append(groupped_samples[quesn_idx][soln_counter])
                soln_counter += 1
                if len(output_instances) == self.num_samples:
                    break
            if self.do_shuffle:
                random.shuffle(output_instances)
        else:
            raise NotImplementedError(f"Sampling method {self.sampling_method} not implemented")

        with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
            for instance in output_instances:
                fout.write(json.dumps(instance) + "\n")


class WriteFinalSftManifest(BaseProcessor):
    def __init__(self, prompt_type: str, chat_format: bool = False, metadata: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.prompt_type = prompt_type
        self.chat_format = chat_format
        self.metadata = metadata
        if not self.metadata:
            self.metadata = {}

    def process(self):
        with (
            open(self.input_manifest_file, "rt", encoding="utf-8") as fin,
            open(self.output_manifest_file, "wt", encoding="utf-8") as fout,
        ):
            prompt_config = get_prompt_config(self.prompt_type)
            prompt = Prompt(config=prompt_config)
            # only looping over the correct samples (unless asked for incorrect)
            for line in fin:
                elem = json.loads(line)
                if self.chat_format:
                    elem['conversations'] = [
                        {'value': elem['question'], 'from': 'User', 'canonical_form': ''},
                        {'value': elem.pop("generation"), 'from': 'Assistant', 'canonical_form': ''},
                    ]
                    elem['system'] = prompt_config.system
                    elem['mask'] = 'User'
                    elem['type'] = None
                else:
                    elem["input"] = prompt.build_string(input_dict={"question": elem['question']})
                    elem["output"] = elem.pop("generation")
                elem.update(self.metadata)
                fout.write(json.dumps(elem) + "\n")
