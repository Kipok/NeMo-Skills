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

import subprocess
from pathlib import Path

# tuple of dataset name, available splits and prepared sft files
DATASETS = [
    ('algebra222', ['test'], []),
    ('asdiv', ['test'], []),
    ('gsm-hard', ['test', 'test_rounded'], []),
    ('mawps', ['test'], []),
    ('svamp', ['test'], []),
    ('tabmwp', ['train', 'validation', 'test'], []),
    ('gsm8k', ['train', 'train_full', 'validation', 'test'], ['validation-sft', 'validation-sft-chat']),
    ('gsm-plus', ['test', 'test_rounded'], []),
    ('gsm-ic-2step', ['test'], []),
    ('gsm-ic-mstep', ['test'], []),
    ('functional', ['test'], []),
    ('math', ['train', 'train_full', 'validation', 'test'], ['validation-sft', 'validation-sft-chat']),
    ('human-eval', ['test'], []),
    ('mbpp', ['test'], []),
    ('mmlu', ['test', 'dev', 'val'], []),
    ('ifeval', ['test'], []),
    ('math-odyssey', ['test'], []),
    ('aime-2024', ['test'], []),
]


def test_data_scripts():
    subprocess.run(
        f'python {Path(__file__).absolute().parents[1] / "nemo_skills" / "dataset" / "prepare.py"}',
        shell=True,
        check=True,
    )

    # checking that all expected files are created
    expected_files = []
    for dataset, splits, sft_files in DATASETS:
        for split in splits:
            expected_files.append(f"{dataset}/{split}.jsonl")
        for sft_file in sft_files:
            expected_files.append(f"{dataset}/{sft_file}.jsonl")

    for file in expected_files:
        assert (Path(__file__).absolute().parents[1] / "datasets" / file).exists()
