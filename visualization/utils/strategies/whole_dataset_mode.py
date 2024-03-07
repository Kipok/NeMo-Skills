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
import os
from typing import Dict, List

import dash_bootstrap_components as dbc
from dash import html
from flask import current_app
from omegaconf import OmegaConf
import requests

from settings.constants import (
    GREEDY,
    METRICS,
    OUTPUT,
    OUTPUT_PATH,
    PARAMETERS_FILE_NAME,
    RESULTS_PATH,
    WHOLE_DATASET_MODE,
)
from settings.templates import compute_metrics_template
from utils.common import get_available_models, get_examples, run_subprocess
from utils.strategies.base_strategy import ModeStrategies

from nemo_skills.evaluation.evaluate_results import (
    EvaluateResultsConfig,
    evaluate_results,
)
from nemo_skills.inference.generate_solutions import (
    GenerateSolutionsConfig,
    InferenceConfig,
    generate_solutions,
)
from nemo_skills.inference.prompt.utils import PromptConfig


class WholeDatasetModeStrategy(ModeStrategies):
    mode = WHOLE_DATASET_MODE

    def __init__(self):
        super().__init__()

    def get_utils_input_layout(self) -> List[dbc.AccordionItem]:
        inference_condition = lambda name, value: isinstance(value, (str, int, float))
        return super().get_utils_input_layout(
            inference_condition,
            lambda name, value: True,
            False,
            list(self.config.items()),
        )

    def get_query_input_layout(self) -> List[dbc.AccordionItem]:
        return []

    def run(self, utils: Dict, params: Dict):
        runs_storage = get_available_models()

        run_index = len(runs_storage)
        metrics_directory = RESULTS_PATH.format(run_index)
        output_file = os.path.join(metrics_directory, OUTPUT_PATH.format(OUTPUT, GREEDY))
        save_metrics_file = os.path.join(
            metrics_directory, OUTPUT_PATH.format(METRICS, GREEDY)
        )
        random_seed_start = (
            utils['start_random_seed']
            if params['range_random_mode']
            else utils['random_seed']
        )
        random_seed_end = (
            utils['end_random_seed']
            if params['range_random_mode']
            else utils['random_seed'] + 1
        )
        generate_solutions_config = GenerateSolutionsConfig(
            output_file=output_file,
            sandbox=self.config['sandbox'],
            server=self.config['server'],
            data_file=self.config['data_file'],
            **{
                key: value
                for key, value in utils.items()
                if key in current_app.config['data_explorer']
            },
        )
        generate_solutions_config.prompt = PromptConfig(
            **{
                key: value
                for key, value in utils.items()
                if key in current_app.config['data_explorer']['prompt']
            }
        )
        generate_solutions_config.prompt.context = utils['context_templates']
        generate_solutions_config.prompt.examples = get_examples().get(
            utils['examples_type'] if utils['examples_type'] else "", []
        )

        generate_solutions_config.inference = InferenceConfig(
            **{
                key: value
                for key, value in utils.items()
                if key in current_app.config['data_explorer']['inference']
            }
        )
        for random_seed in range(random_seed_start, random_seed_end):
            output_file = (
                output_file
                if not params['range_random_mode']
                else os.path.join(
                    metrics_directory,
                    OUTPUT_PATH.format(OUTPUT, "rs" + str(random_seed)),
                )
            )
            save_metrics_file = (
                save_metrics_file
                if not params['range_random_mode']
                else os.path.join(
                    metrics_directory,
                    OUTPUT_PATH.format(METRICS, "rs" + str(random_seed)),
                )
            )
            generate_solutions_config.output_file = output_file
            generate_solutions_config.inference.random_seed = random_seed
            try:
                logging.info("Generate solutions")
                generate_solutions(OmegaConf.structured(generate_solutions_config))
                evaluate_results_config = EvaluateResultsConfig(
                    prediction_jsonl_files=output_file,
                    sabdbox=self.config['sandbox'],
                )
                logging.info("Evaluate results")
                evaluate_results(OmegaConf.structured(evaluate_results_config))

                compute_metrics_command = compute_metrics_template.format(
                    prediction_jsonl_files=output_file,
                    save_metrics_file=save_metrics_file,
                )
            except requests.exceptions.ConnectionError as e:
                return self._get_connection_error_message()
            except Exception as e:
                return html.Div(f"Something went wrong\n{e}")

            logging.info("Compute metrics")
            _, errors, success = run_subprocess(compute_metrics_command)
            if not success:
                return html.Div(f"Something went wrong\n{errors}")

        runs_storage[run_index] = {
            "utils": utils,
            "examples": get_examples().get(utils["examples_type"], []),
        }

        with open(PARAMETERS_FILE_NAME, "w") as f:
            f.write(json.dumps(runs_storage))

        return html.Pre(
            f'Done. Results are in folder\n{"/".join(output_file.split("/")[:-1])}'
        )

    def get_prompt(self, utils: Dict, question: str) -> str:
        return super().get_prompt(utils, "***your question***")
