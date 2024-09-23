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
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).absolute().parents[1]))
from nemo_skills.evaluation.metrics import compute_metrics


@pytest.mark.gpu
def test_sft():
    model_path = os.getenv('NEMO_SKILLS_TEST_NEMO_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_NEMO_MODEL to run this test")

    cmd = (
        f"python -m nemo_skills.pipeline.train "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --expname test-sft "
        f"    --output_dir /tmp/nemo-skills-tests/train-sft "
        f"    --nemo_model {model_path} "
        f"    --num_nodes 1 "
        f"    --num_gpus 1 "
        f"    --num_training_jobs 1 "
        f"    --training_data /nemo_run/code/tests/data/small-sft-data.test "
        f"    --disable_wandb "
        f"    ++trainer.sft.val_check_interval=7 "
        f"    ++trainer.sft.save_interval=7 "
        f"    ++trainer.sft.limit_val_batches=1 "
        f"    ++trainer.sft.max_steps=15 "
        f"    ++trainer.sft.max_epochs=10 "
        f"    ++model.data.train_ds.add_eos=False "
        f"    ++model.data.train_ds.global_batch_size=10 "
        f"    ++model.data.train_ds.micro_batch_size=2 "
        f"    ++model.optim.lr=1e-6 "
        f"    ++model.optim.sched.warmup_steps=0 "
        f"    ++model.tensor_model_parallel_size=1 "
        f"    ++model.pipeline_model_parallel_size=1 "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that the final model can be used for evaluation
    cmd = (
        f"python -m nemo_skills.pipeline.eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model /tmp/nemo-skills-tests/train-sft/model-averaged-nemo "
        f"    --server_type nemo "
        f"    --output_dir /tmp/nemo-skills-tests/train-sft/evaluation "
        f"    --benchmarks gsm8k:0 "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    ++prompt_template=llama3-instruct "
        f"    ++split=test "
        f"    ++batch_size=8 "
        f"    ++max_samples=10 "
    )
    subprocess.run(cmd, shell=True, check=True)

    metrics = compute_metrics(
        [f"/tmp/nemo-skills-tests/train-sft/evaluation/eval-results/gsm8k/output-greedy.jsonl"],
        importlib.import_module('nemo_skills.dataset.gsm8k').METRICS_CLASS(),
    )
    # only checking the total, since model is tiny
    assert metrics['num_entries'] == 10


@pytest.mark.gpu
def test_dpo():
    model_path = os.getenv('NEMO_SKILLS_TEST_NEMO_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_NEMO_MODEL to run this test")

    model_path = os.getenv('NEMO_SKILLS_TEST_NEMO_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_NEMO_MODEL to run this test")

    cmd = (
        f"python -m nemo_skills.pipeline.train "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --expname test-dpo "
        f"    --output_dir /tmp/nemo-skills-tests/test-dpo "
        f"    --nemo_model {model_path} "
        f"    --num_nodes 1 "
        f"    --num_gpus 1 "
        f"    --num_training_jobs 1 "
        f"    --training_data /nemo_run/code/tests/data/small-dpo-data.test "
        f"    --disable_wandb "
        f"    --training_algo dpo "
        f"    ++trainer.dpo.val_check_interval=1 "
        f"    ++trainer.dpo.save_interval=1 "
        f"    ++trainer.dpo.limit_val_batches=1 "
        f"    ++trainer.dpo.max_steps=3 "
        f"    ++trainer.dpo.max_epochs=10 "
        f"    ++model.data.train_ds.add_eos=False "
        f"    ++model.global_batch_size=1 "
        f"    ++model.micro_batch_size=1 "
        f"    ++model.optim.lr=1e-6 "
        f"    ++model.optim.sched.warmup_steps=0 "
        f"    ++model.tensor_model_parallel_size=1 "
        f"    ++model.pipeline_model_parallel_size=1 "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that the final model can be used for evaluation
    cmd = (
        f"python -m nemo_skills.pipeline.eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model /tmp/nemo-skills-tests/test-dpo/model-averaged-nemo "
        f"    --server_type nemo "
        f"    --output_dir /tmp/nemo-skills-tests/test-dpo/evaluation "
        f"    --benchmarks gsm8k:0 "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    ++prompt_template=llama3-instruct "
        f"    ++split=test "
        f"    ++batch_size=8 "
        f"    ++max_samples=10 "
    )
    subprocess.run(cmd, shell=True, check=True)

    metrics = compute_metrics(
        [f"/tmp/nemo-skills-tests/test-dpo/evaluation/eval-results/gsm8k/output-greedy.jsonl"],
        importlib.import_module('nemo_skills.dataset.gsm8k').METRICS_CLASS(),
    )
    # only checking the total, since model is tiny
    assert metrics['num_entries'] == 10
