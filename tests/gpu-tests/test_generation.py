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


@pytest.mark.gpu
def test_trtllm_run_eval():
    model_path = os.getenv('NEMO_SKILLS_TEST_TRTLLM_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_TRTLLM_MODEL to run this test")

    cmd = (
        f"python -m nemo_skills.pipeline.eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type trtllm "
        f"    --output_dir /tmp/nemo-skills-tests/trtllm-eval "
        f"    --benchmarks gsm8k:0 "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    ++prompt_template=llama3-instruct "
        f"    ++split_name=test "
        f"    ++batch_size=8 "
        f"    ++max_samples=20 "
    )
    subprocess.run(cmd, shell=True, check=True)

    # running compute_metrics to check that results are expected
    metrics_calculator = importlib.import_module('nemo_skills.dataset.gsm8k').METRICS_CLASS()
    metrics = compute_metrics(
        [f"/tmp/nemo-skills-tests/trtllm-eval/eval-results/gsm8k/output-greedy.jsonl"], metrics_calculator
    )
    # rough check, since exact accuracy varies depending on gpu type
    assert int(metrics['sympy_correct']) >= 50
    assert metrics['num_entries'] == 20


@pytest.mark.gpu
def test_vllm_run_eval():
    # this test expects llama3-instruct to properly check accuracy
    # will run a bunch of benchmarks, but is still pretty fast
    # mmlu/IFMetrics will be cut to 400 samples to save time
    # could cut everything, but human-eval/mbpp don't work with partial gens
    model_path = os.getenv('NEMO_SKILLS_TEST_HF_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")

    cmd = (
        f"python -m nemo_skills.pipeline.eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type vllm "
        f"    --output_dir /tmp/nemo-skills-tests/vllm-eval "
        f"    --benchmarks math:0 human-eval:0 mbpp:0 ifeval:0 mmlu:0 "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    ++prompt_template=llama3-instruct "
        f"    ++split_name=test "
        f"    ++batch_size=400 "
        f"    ++max_samples=400 "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that summarize results works (just that there are no errors, but can inspect the output as well)
    subprocess.run(
        "python -m nemo_skills.pipeline.summarize_results /tmp/nemo-skills-tests/vllm-eval",
        shell=True,
        check=True,
    )

    # running compute_metrics to check that results are expected
    metrics = compute_metrics(
        ["/tmp/nemo-skills-tests/vllm-eval/eval-results/algebra222/output-greedy.jsonl"],
        importlib.import_module('nemo_skills.dataset.math').METRICS_CLASS(),
    )

    assert metrics['sympy_correct'] >= 30
    assert metrics['num_entries'] == 400

    metrics = compute_metrics(
        ["/tmp/nemo-skills-tests/vllm-eval/eval-results/human-eval/output-greedy.jsonl"],
        importlib.import_module('nemo_skills.dataset.human-eval').METRICS_CLASS(),
    )
    assert metrics['passing_base_tests'] >= 50
    assert metrics['passing_plus_tests'] >= 50
    assert metrics['num_entries'] == 164

    metrics = compute_metrics(
        ["/tmp/nemo-skills-tests/vllm-eval/eval-results/mbpp/output-greedy.jsonl"],
        importlib.import_module('nemo_skills.dataset.mbpp').METRICS_CLASS(),
    )
    assert metrics['passing_base_tests'] >= 50
    assert metrics['passing_plus_tests'] >= 50
    assert metrics['num_entries'] == 378

    metrics = compute_metrics(
        ["/tmp/nemo-skills-tests/vllm-eval/eval-results/ifeval/output-greedy.jsonl"],
        importlib.import_module('nemo_skills.dataset.ifeval').METRICS_CLASS(),
    )
    assert metrics['prompt_strict_accuracy'] >= 60
    assert metrics['instruction_strict_accuracy'] >= 70
    assert metrics['prompt_loose_accuracy'] >= 60
    assert metrics['instruction_loose_accuracy'] >= 70
    assert metrics['num_prompts'] == 400
    assert metrics['num_instructions'] == 601

    metrics = compute_metrics(
        ["/tmp/nemo-skills-tests/vllm-eval/eval-results/mmlu/output-greedy.jsonl"],
        importlib.import_module('nemo_skills.dataset.mmlu').METRICS_CLASS(),
    )
    assert metrics['sympy_correct'] >= 50
    assert metrics['num_entries'] == 400


@pytest.mark.gpu
def test_trtllm_run_eval_retrieval():
    model_path = os.getenv('LLAMA3_8B_BASE_TRTLLM')
    if not model_path:
        pytest.skip("Define LLAMA3_8B_BASE_TRTLLM to run this test")
    output_path = os.getenv('NEMO_SKILLS_TEST_OUTPUT', '/tmp')

    cmd = f""" \
python pipeline/run_eval.py \
    --model_path {model_path} \
    --server_type trtllm \
    --output_dir {output_path} \
    --benchmarks math:0 \
    --num_gpus 1 \
    --num_nodes 1 \
    +prompt=openmathinstruct/reference \
    ++prompt.few_shot_examples.retrieval_field=question  \
    ++prompt.few_shot_examples.retrieval_file=/code/datasets/math/train_full.jsonl \
    ++prompt.few_shot_examples.num_few_shots=5 \
    ++split_name=test \
    ++batch_size=8 \
    ++max_samples=20 \
"""
    subprocess.run(cmd, shell=True)

    # double checking that code was actually executed
    with open(f"{output_path}/math/output-greedy.jsonl") as fin:
        data = [json.loads(line) for line in fin]

    for elem in data:
        assert '<llm-code>' not in elem['generation']
        assert elem['error_message'] == '<not_executed>'

    # running compute_metrics to check that results are expected
    metrics = compute_metrics([f"{output_path}/math/output-greedy.jsonl"], EVALUATOR_MAP['math']())
    assert (int(metrics['sympy_correct']), int(metrics['no_answer'])) == (10, 15)
    assert metrics['num_entries'] == 20


@pytest.mark.gpu
def test_trtllm_run_labeling():
    model_path = os.getenv('LLAMA3_8B_BASE_TRTLLM')
    if not model_path:
        pytest.skip("Define LLAMA3_8B_BASE_TRTLLM to run this test")
    output_path = os.getenv('NEMO_SKILLS_TEST_OUTPUT', '/tmp')

    cmd = f""" \
python pipeline/run_labeling.py \
    --model_path {model_path} \
    --server_type trtllm \
    --output_dir {output_path} \
    --num_gpus 1 \
    --num_nodes 1 \
    +prompt=openmathinstruct/base \
    ++prompt.few_shot_examples.examples_type=gsm8k_only_code \
    ++prompt.few_shot_examples.num_few_shots=5 \
    ++dataset=gsm8k \
    ++split_name=train_full \
    ++inference.temperature=0.5 \
    ++skip_filled=False \
    ++batch_size=8 \
    ++max_samples=20 \
"""
    subprocess.run(cmd, shell=True)

    # double checking that code was actually executed
    with open(f"{output_path}/output-rs0.jsonl") as fin:
        data = [json.loads(line) for line in fin]

    for elem in data:
        assert '<llm-code>' in elem['generation']
        assert elem['error_message'] != '<not_executed>'

    # running compute_metrics to check that results are expected
    metrics = compute_metrics([f"{output_path}/output-rs0.jsonl"], EVALUATOR_MAP['gsm8k']())
    assert (int(metrics['sympy_correct']), int(metrics['no_answer'])) == (30, 15)
    assert metrics['num_entries'] == 20


@pytest.mark.gpu
def test_nemo_run_eval():
    model_path = os.getenv('LLAMA3_8B_BASE_NEMO')
    if not model_path:
        pytest.skip("Define LLAMA3_8B_BASE_NEMO to run this test")
    output_path = os.getenv('NEMO_SKILLS_TEST_OUTPUT', '/tmp')

    cmd = f""" \
python pipeline/run_eval.py \
    --model_path {model_path} \
    --server_type nemo \
    --output_dir {output_path} \
    --benchmarks gsm8k:0 \
    --num_gpus 1 \
    --num_nodes 1 \
    +prompt=openmathinstruct/base \
    ++prompt.few_shot_examples.examples_type=gsm8k_only_code \
    ++prompt.few_shot_examples.num_few_shots=5 \
    ++prompt.context_type=reference_solution \
    ++split_name=test \
    batch_size=8 \
    max_samples=20 \
"""
    subprocess.run(cmd, shell=True)

    # double checking that code was actually executed
    with open(f"{output_path}/gsm8k/output-greedy.jsonl") as fin:
        data = [json.loads(line) for line in fin]

    for elem in data:
        assert '<llm-code>' in elem['generation']
        assert elem['error_message'] != '<not_executed>'

    # running compute_metrics to check that results are expected
    metrics = compute_metrics([f"{output_path}/gsm8k/output-greedy.jsonl"], EVALUATOR_MAP['gsm8k']())
    assert (int(metrics['sympy_correct']), int(metrics['no_answer'])) == (90, 5)
    assert metrics['num_entries'] == 20
