# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import json
from pathlib import Path

import pytest

from nemo_skills.evaluation.metrics.if_metrics import IFMetrics

IFEVAL_FIXTURE = Path(__file__).parent / "data" / "eval_outputs" / "eval-results" / "ifeval" / "output-rs0.jsonl-test"


@pytest.fixture
def prediction() -> dict:
    """One record in the shape the ifeval evaluator writes and summarize_results reads."""
    with open(IFEVAL_FIXTURE, encoding="utf-8") as fin:
        return json.loads(fin.readline())


def test_incorrect_sample_resets_both_eval_modes(prediction):
    """--max_seq_len grades an over-long sample incorrect through this method.

    The IFEval keys live inside strict_eval and loose_eval, which is where
    update() reads them from; there is no top-level follow_all_instructions.
    """
    incorrect = IFMetrics().get_incorrect_sample(prediction)

    for eval_key in ("strict_eval", "loose_eval"):
        assert incorrect[eval_key]["follow_all_instructions"] is False
        assert not any(incorrect[eval_key]["follow_instruction_list"])
        assert len(incorrect[eval_key]["follow_instruction_list"]) == len(
            prediction[eval_key]["follow_instruction_list"]
        )


def test_incorrect_sample_scores_zero(prediction):
    """What the caller is after: the sample must now score as a miss."""
    incorrect = IFMetrics().get_incorrect_sample(prediction)

    for eval_key in ("strict_eval", "loose_eval"):
        score = IFMetrics()._get_score_dict(incorrect[eval_key])
        assert score["prompt"] is False
        assert score["instruction"] == 0


def test_incorrect_sample_leaves_the_original_alone(prediction):
    """Every sibling calculator returns a copy; compute_metrics reassigns data[i]."""
    before = json.loads(json.dumps(prediction))

    IFMetrics().get_incorrect_sample(prediction)

    assert prediction == before
