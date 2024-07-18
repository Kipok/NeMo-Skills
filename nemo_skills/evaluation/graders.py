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
import re
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

LOG = logging.getLogger(__file__)


def math_grader(cfg):
    from nemo_skills.code_execution.sandbox import get_sandbox

    sandbox = get_sandbox(**cfg.sandbox)
    sandbox.batch_evaluate_results(
        prediction_jsonl_files=cfg.prediction_jsonl_files,
        **cfg.eval_config,
    )


def code_grader(cfg):
    # TODO: need to move it to a separate docker (either our sandbox or separate srun)
    from evalplus.evaluate import evaluate
    from omegaconf import OmegaConf

    from nemo_skills.evaluation.code_utils import preprocess_code

    # processing each generation separately (TODO: evalplus can do it together, but need to figure out the format)
    for jsonl_file in cfg.prediction_jsonl_files:
        with open(jsonl_file) as f:
            samples = [preprocess_code(json.loads(line)) for line in f]
        # all changes will be done with a new key "completion", so it's ok to write to the same file
        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        eval_config = {
            "samples": jsonl_file,
            "base_only": False,
            "parallel": None,
            "i_just_wanna_run": False,
            "test_details": False,
            "min_time_limit": 1,
            "gt_time_limit_factor": 4.0,
            "mini": False,
            "noextreme": False,
            "version": "default",
        }
        eval_config.update(OmegaConf.to_container(cfg.eval_config))
        evaluate(Namespace(**eval_config))
        with open(jsonl_file[:-6] + '_eval_results.json', 'rt', encoding="utf-8") as fin:
            evalplus_grades = json.load(fin)
        # adding is_correct key to allow compute_metrics to work
        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                sample['is_correct'] = evalplus_grades['eval'][sample['task_id']][0]['base_status'] == "pass"
                sample['is_correct-plus'] = (
                    sample['is_correct'] and evalplus_grades['eval'][sample['task_id']][0]['plus_status'] == "pass"
                )
                f.write(json.dumps(sample) + "\n")

        # moving eval file as otherwise evalplus does not want to recompute metrics if it's present..
        shutil.move(jsonl_file[:-6] + '_eval_results.json', jsonl_file[:-6] + '_eval_results-saved.json')


def if_grader(cfg):
    for jsonl_file in cfg.prediction_jsonl_files:
        parent_dir = Path(jsonl_file).absolute().parent
        cmd = (
            'cd /opt/benchmarks/google-research && python -m instruction_following_eval.evaluation_main '
            f'--input_data={jsonl_file} '
            f'--input_response_data={jsonl_file} '
            f'--output_dir={parent_dir} '
        )
        subprocess.run(cmd, shell=True, check=True)
        # fusing eval metrics back into the generation file
        with open(jsonl_file, "rt", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f]

        with open(parent_dir / 'eval_results_loose.jsonl', 'rt', encoding="utf-8") as f:
            eval_results = [json.loads(line) for line in f]
        for sample, eval_result in zip(samples, eval_results):
            sample['loose_eval'] = eval_result

        with open(parent_dir / 'eval_results_strict.jsonl', 'rt', encoding="utf-8") as f:
            eval_results = [json.loads(line) for line in f]
        for sample, eval_result in zip(samples, eval_results):
            sample['strict_eval'] = eval_result

        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        # removing metric files to avoid reusing them
        (parent_dir / 'eval_results_loose.jsonl').unlink()
        (parent_dir / 'eval_results_strict.jsonl').unlink()


def arena_grader(cfg):
    def get_score(judgment):
        pattern = re.compile('\[\[([AB<>=]+)\]\]')
        # adapted from https://github.com/lm-sys/arena-hard-auto/blob/main/gen_judgment.py
        matches = pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None
        elif len(set(matches)) == 1:
            return matches[0].strip("\n")
        else:
            return None

    # currently only support api models for simplicity
    def write_judgements(data_file, judge_model='gpt-4-1106-preview', base_url=None, batch_size=10):
        data_file = Path(data_file).absolute()
        parent_dir = Path(__file__).absolute().parent
        cmd = (
            f'{sys.executable} {Path(__file__).absolute().parents[2]}/nemo_skills/inference/generate_solutions.py '
            f'+prompt=openai/arena-judge '
            f'++server.server_type=openai '
            f'++server.model={judge_model} '
            f'++data_file={data_file} '
            f'++output_file={parent_dir}/judgement.jsonl '
            f'{"++server.base_url=" + base_url if base_url else ""} '
            f'++batch_size={batch_size} '
            f'++inference.tokens_to_generate=2048 '  # TODO: this should ideally be as large as possible, but need a tokenizer
        )
        # TODO: number_of_judgment_attempts ?
        subprocess.run(cmd, shell=True, check=True)

        # fusing judge responses back into the generation file
        with open(parent_dir / 'judgement.jsonl', 'rt', encoding="utf-8") as f:
            judgements = [json.loads(line)['generation'] for line in f]

        with open(str(data_file)[:-4], 'rt', encoding='utf-8') as fin:
            samples = [json.loads(line) for line in fin]

        for sample, judgement in zip(samples, judgements):
            sample['judgements'].append(judgement)
            sample['judge_scores'].append(get_score(judgement))

        # writing back to the original file without -tmp
        with open(str(data_file)[:-4], "wt", encoding="utf-8") as fout:
            for sample in samples:
                fout.write(json.dumps(sample) + "\n")

        (parent_dir / 'judgement.jsonl').unlink()

    for jsonl_file in cfg.prediction_jsonl_files:
        # preparing input generation file by adding baseline answers
        with open(jsonl_file, 'rt', encoding='utf-8') as fin:
            samples = [json.loads(line) for line in fin]

        # TODO: caching?
        # need to create a tmp file to not override the generation key
        with open(jsonl_file + '-tmp', "wt", encoding="utf-8") as fout:
            for sample in samples:
                sample['answer_1'] = sample.pop('generation')
                sample['answer_2'] = sample['baseline_answer']
                fout.write(json.dumps(sample) + "\n")

        write_judgements(data_file=jsonl_file + '-tmp', **cfg.eval_config)

        # switching the order of answers
        with open(jsonl_file, 'rt', encoding='utf-8') as fin:
            samples = [json.loads(line) for line in fin]

        with open(jsonl_file + '-tmp', "wt", encoding="utf-8") as fout:
            for sample in samples:
                sample['answer_1'] = sample['baseline_answer']
                sample['answer_2'] = sample.pop('generation')
                fout.write(json.dumps(sample) + "\n")

        write_judgements(data_file=jsonl_file + '-tmp', **cfg.eval_config)

        Path(jsonl_file + '-tmp').unlink()
