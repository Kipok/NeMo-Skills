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

# running most things through subprocess since that's how it's usually used
import subprocess

import pytest
from test_datasets import DATASETS

DATA_TO_TEST = []
for prompt_template in ['default-base', 'llama3-base', 'llama3-instruct', 'nemotron-instruct']:
    for dataset, splits, _ in DATASETS:
        for split in splits:
            DATA_TO_TEST.append((dataset, split, prompt_template))


@pytest.mark.parametrize("dataset,split_name,prompt_template", DATA_TO_TEST)
def test_generation_dryrun_default(dataset, split_name, prompt_template):
    """Testing the default prompts for each dataset."""
    cmd = (
        "python nemo_skills/inference/generate.py "
        f"    ++output_file=./test.jsonl "
        f"    ++prompt_template={prompt_template} "
        f"    ++dataset={dataset} "
        f"    ++split_name={split_name} "
        f"    ++server.server_type=nemo "
        f"    ++dry_run=True "
    )
    subprocess.run(cmd, shell=True, check=True)


# @pytest.mark.parametrize(
#     "dataset,split_name,prompt_template,prompt_config",
#     ['local', 'piston'],
# )
# def test_generation_dryrun_specific(dataset, split_name, prompt_template):
#     """Testing a couple of specific prompts."""
#     cmd = (
#         "python nemo_skills/inference/generate.py "
#         f"    ++output_file=./test.jsonl "
#         f"    ++prompt_config={prompt_config} "
#         f"    ++prompt_template={prompt_template} "
#         f"    ++dataset={dataset} "
#         f"    ++split_name={split_name} "
#         f"    ++server.server_type=nemo "
#         f"    ++dry_run=True "
#     )
#     subprocess.run(cmd, shell=True, check=True)
