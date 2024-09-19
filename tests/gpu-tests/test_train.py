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

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).absolute().parents[1]))
from nemo_skills.evaluation.metrics import MathMetrics, compute_metrics


@pytest.mark.gpu
def test_sft():
    model_path = os.getenv('NEMO_SKILLS_TEST_NEMO_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_NEMO_MODEL to run this test")

    cmd = (
        f"python -m nemo_skills.pipeline.train "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --expname test-train "
        f"    --output_dir /tmp/nemo-skills-tests/train-sft "
        f"    --nemo_model {model_path} "
        f"    --num_nodes 1 "
        f"    --num_gpus 1 "
        f"    --num_training_jobs 1 "
        f"    --training_data /nemo_run/code/nemo_skills/dataset/math/test.jsonl "
    )

    """
    python nemo_skills/pipeline/train.py \
        --cluster local \
        --expname test \
        --output_dir /exps/checkpoints/test \
        --nemo_model /nemo_models/llama3-tiny \
        --num_nodes 1 \
        --num_gpus 1 \
        --num_training_jobs 1 \
        --training_data /data/data.jsonl \
        ++model.data.train_ds.add_eos=False \
        ++model.data.train_ds.global_batch_size=10 \
        ++model.data.train_ds.micro_batch_size=1 \
        ++trainer.sft.val_check_interval=1 \
        ++trainer.sft.save_interval=1 \
        ++trainer.sft.limit_val_batches=1 \
        ++trainer.sft.max_steps=3 \
        ++trainer.sft.max_epochs=10 \
        ++model.optim.lr=5e-6 \
        ++model.optim.sched.warmup_steps=0 \
        ++model.tensor_model_parallel_size=1 \
        ++model.pipeline_model_parallel_size=1
    """

    cmd = f""" \
python {Path(__file__).absolute().parents[2]}/datasets/gsm8k/prepare.py --split validation && \
export NEMO_SKILLS_DATA={Path(__file__).absolute().parents[2] / 'datasets'} && \
export NEMO_SKILLS_RESULTS={output_path} && \
python pipeline/run_pipeline.py \
      --expname test-sft \
      --nemo_model {model_path} \
      --num_nodes 1 \
      --num_gpus 1 \
      --disable_wandb \
      --extra_eval_args "+prompt=openmathinstruct/sft ++max_samples=4 --benchmarks gsm8k:1 math:0 --num_jobs 1 --num_gpus 1" \
      ++model.data.train_ds.file_path=/data/gsm8k/validation-sft.jsonl \
      ++trainer.sft.max_steps=15 \
      ++trainer.sft.val_check_interval=10 \
      ++trainer.sft.limit_val_batches=2 \
      ++model.data.train_ds.global_batch_size=4 \
      ++model.tensor_model_parallel_size=1 \
      ++model.pipeline_model_parallel_size=1 \
      ++model.optim.lr=1e-6 \
"""
    subprocess.run(cmd, shell=True)

    # only checking the total, since model is tiny
    for gen_file in ['gsm8k/output-greedy.jsonl', 'gsm8k/output-rs0.jsonl', 'math/output-greedy.jsonl']:
        metrics = compute_metrics([f"{output_path}/nemo-skills-exps/results/test-sft/{gen_file}"], MathMetrics())
        assert metrics['num_entries'] == 4


@pytest.mark.gpu
def test_dpo():
    model_path = os.getenv('NEMO_SKILLS_TEST_NEMO_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_NEMO_MODEL to run this test")
    output_path = os.getenv('NEMO_SKILLS_TEST_OUTPUT', '/tmp')

    cmd = f""" \
python {Path(__file__).absolute().parents[2]}/datasets/gsm8k/prepare.py --split validation && \
export NEMO_SKILLS_DATA={Path(__file__).absolute().parent} && \
export NEMO_SKILLS_RESULTS={output_path} && \
python pipeline/run_pipeline.py \
      --expname test-dpo \
      --nemo_model {model_path} \
      --num_nodes 1 \
      --num_gpus 1 \
      --disable_wandb \
      --stages dpo prepare_eval eval \
      --extra_eval_args "+prompt=openmathinstruct/sft ++max_samples=4 --benchmarks gsm8k:0 --num_jobs 1 --num_gpus 1" \
      ++model.data.data_prefix.train='[/data/small-dpo-data.test]' \
      ++model.data.data_prefix.validation='[/data/small-dpo-data.test]' \
      ++model.data.data_prefix.test='[/data/small-dpo-data.test]' \
      ++trainer.dpo.max_steps=15 \
      ++trainer.dpo.max_epochs=10 \
      ++trainer.dpo.val_check_interval=10 \
      ++trainer.dpo.limit_val_batches=2 \
      ++model.global_batch_size=4 \
      ++model.tensor_model_parallel_size=1 \
      ++model.pipeline_model_parallel_size=1 \
      ++model.optim.lr=1e-6 \
"""
    subprocess.run(cmd, shell=True)

    # only checking the total, since model is tiny
    metrics = compute_metrics(
        [f"{output_path}/nemo-skills-exps/results/test-dpo/gsm8k/output-greedy.jsonl"], MathMetrics()
    )
    assert metrics['num_entries'] == 4
