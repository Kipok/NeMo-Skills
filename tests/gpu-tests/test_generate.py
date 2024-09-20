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

# needs to define NEMO_SKILLS_TEST_TRTLLM_MODEL to run these tests
# needs to define NEMO_SKILLS_TEST_NEMO_MODEL to run these tests
# you'd also need 2+ GPUs to run this test
# the metrics are assuming llama3-8b-base as the model and will fail for other models

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).absolute().parents[1]))
from nemo_skills.evaluation.metrics import compute_metrics

# TODO: retrieval test


@pytest.mark.gpu
def test_vllm_generate_greedy():
    model_path = os.getenv('NEMO_SKILLS_TEST_HF_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")

    cmd = (
        f"python -m nemo_skills.pipeline.generate "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type vllm "
        f"    --output_dir /tmp/nemo-skills-tests/vllm-generate-greedy "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    ++input_file=/nemo_run/code/nemo_skills/dataset/math/test.jsonl "
        f"    ++prompt_config=generic/math "
        f"    ++prompt_template=llama3-instruct "
        f"    ++batch_size=8 "
        f"    ++max_samples=10 "
    )
    subprocess.run(cmd, shell=True, check=True)

    # no evaluation by default - checking just the number of lines and that there is no is_correct key
    with open(f"/tmp/nemo-skills-tests/vllm-generate-greedy/generation/output.jsonl") as fin:
        lines = fin.readlines()
    assert len(lines) == 10
    for line in lines:
        data = json.loads(line)
        assert 'is_correct' not in data
        assert 'generation' in data


@pytest.mark.gpu
def test_vllm_generate_seeds():
    model_path = os.getenv('NEMO_SKILLS_TEST_HF_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")

    cmd = (
        f"python -m nemo_skills.pipeline.generate "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type vllm "
        f"    --output_dir /tmp/nemo-skills-tests/vllm-generate-greedy "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --num_random_seeds 3 "
        f"    --eval_args='++eval_type=math' "
        f"    ++dataset=math "
        f"    ++split=test "
        f"    ++prompt_template=llama3-instruct "
        f"    ++split=test "
        f"    ++batch_size=8 "
        f"    ++max_samples=10 "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that all 3 files are created
    for seed in range(3):
        with open(f"/tmp/nemo-skills-tests/vllm-generate-greedy/generation/output-rs{seed}.jsonl") as fin:
            lines = fin.readlines()
        assert len(lines) == 10
        for line in lines:
            data = json.loads(line)
            assert 'is_correct' in data
            assert 'generation' in data

    # running compute_metrics to check that results are expected
    metrics = compute_metrics(
        [f"/tmp/nemo-skills-tests/vllm-generate-greedy/output-rs*.jsonl"],
        importlib.import_module('nemo_skills.dataset.math').METRICS_CLASS(),
        aggregation_mode="majority",
    )
    # rough check, since exact accuracy varies depending on gpu type
    assert metrics['sympy_correct'] >= 50
    assert metrics['num_entries'] == 10
