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
# See the License for the specific lang

import json

from nemo_skills.evaluation.metrics.answer_judgement_metrics import AnswerJudgementMetrics
from nemo_skills.evaluation.metrics.arena_metrics import ArenaMetrics
from nemo_skills.evaluation.metrics.code_metrics import CodeMetrics
from nemo_skills.evaluation.metrics.if_metrics import IFMetrics
from nemo_skills.evaluation.metrics.lean4_metrics import Lean4Metrics
from nemo_skills.evaluation.metrics.math_metrics import MathMetrics
from nemo_skills.evaluation.metrics.mtbench_metrics import MtBenchMetrics

def read_predictions(predictions, evaluator, allow_incomplete=False):
    data = []
    for prediction in predictions:
        if not prediction:  # could have missing predictions
            if not allow_incomplete:
                raise RuntimeError("Some data is missing!")
            data.append(evaluator.fill_up_missing())
            continue
        prediction_dict = json.loads(prediction)
        if not prediction_dict:
            if not allow_incomplete:
                raise RuntimeError("Some data is missing!")
            data.append(evaluator.fill_up_missing())
            continue
        if evaluator.is_incomplete(prediction_dict):
            if not allow_incomplete:
                raise RuntimeError("Some data is missing!")
            data.append(evaluator.fill_up_missing())
            continue
        data.append(prediction_dict)

    return data


def is_correct_judgement(judgement):
    if 'Judgement:' not in judgement:
        return False  # improper judgement format, so have to judge as false
    verdict = judgement.split('Judgement:')[-1].strip()
    return verdict.lower() == 'yes'



METRICS_MAP = {
    "math": MathMetrics(),
    "lean4-proof": Lean4Metrics(),
    "lean4-statement": Lean4Metrics(),
    "answer-judgement": AnswerJudgementMetrics(),
    "arena": ArenaMetrics(),
    "code": CodeMetrics(),
    "if": IFMetrics(),
    "mt-bench": MtBenchMetrics(),
}

def get_metrics(metric_type: str):


    if metric_type not in METRICS_MAP:
        raise ValueError(
            f"Metric f{metric_type} not found.\nSupported types: {str(METRICS_MAP.keys())}"
        )
    return METRICS_MAP[metric_type]
