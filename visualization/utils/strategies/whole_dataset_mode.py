import json
import logging
import os
from typing import Dict, List

from settings.constants import (
    GREEDY,
    METRICS,
    OUTPUT_PATH,
    OUTPUT,
    ONE_TEST_MODE,
    PARAMETERS_FILE_NAME,
    RESULTS_PATH,
)
from dash import html
import dash_bootstrap_components as dbc

from settings.templates import (
    compute_metrics_template,
    evaluate_results_template,
    generate_solution_template,
)
from utils.common import (
    examples,
    get_available_models,
    run_subprocess,
)
from visualization.utils.strategies.base_strategy import ModeStrategies


class WholeDatasetModeStrategy(ModeStrategies):
    mode = ONE_TEST_MODE

    def __init__(self):
        super().__init__()

    def get_utils_input_layout(self) -> List[dbc.AccordionItem]:
        inference_condition = lambda name, value: not isinstance(value, dict)
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
        output_file = (
            os.path.join(metrics_directory, OUTPUT_PATH.format(OUTPUT, GREEDY))
            if utils["output_file"] == '???'
            else utils["output_file"]
        )
        save_metrics_file = (
            os.path.join(metrics_directory, OUTPUT_PATH.format(METRICS, GREEDY))
            if "save_metrics" not in utils or not utils["save_metrics"]
            else utils["save_metrics"]
        )
        utils['context_type'] = "empty"
        utils['num_few_shots'] = 0
        filled_examples = [
            utils['template'].format(
                context=utils['context_templates'].format(**example_dict),
                **example_dict,
            )
            for example_dict in examples.get(utils['examples_type'], [])
        ]

        utils['template'].replace(
            '{{context}}',
            utils['delimiter'].join(filled_examples) + '{{context}}',
        )
        random_seed_start = (
            utils['start_random_seed']
            if params['range_random_mode']
            else utils['random_seed']
        )
        random_seed_end = (
            utils['random_seed']
            if params['range_random_mode']
            else utils['random_seed'] + 1
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
            generate_solution_command = generate_solution_template.format(
                output_file=output_file,
                sandbox_host=self.config['server']['host'],
                **{
                    key: (
                        value.replace('\n', '\\n')
                        # .replace("'", "\\'")
                        .replace(')', '\)')
                        .replace('(', '\(')
                        .replace('}', '\}')
                        .replace('{', '\{')
                        if isinstance(value, str)
                        else value
                    )
                    for key, value in utils.items()
                    if key != 'output_file'
                },
                **self.config['server'],
            )
            if self.config['data_file']:
                generate_solution_command += f"++data_file={utils['data_file']}"

            evaluate_results_command = evaluate_results_template.format(
                prediction_jsonl_files=output_file,
                sandbox_host=self.config['server']['host'],
                ssh_server=self.config['server']['ssh_server'],
                ssh_key_path=self.config['server']['ssh_key_path'],
            )

            compute_metrics_command = compute_metrics_template.format(
                prediction_jsonl_files=output_file,
                save_metrics_file=save_metrics_file,
            )

            for command, log_message in [
                (generate_solution_command, "Generate solutions"),
                (evaluate_results_command, "Evaluate results"),
                (compute_metrics_command, "Compute metrics"),
            ]:
                logging.info(log_message)
                _, success = run_subprocess(command)
                if not success:
                    return html.Div("Something went wrong")

        runs_storage[run_index] = {
            "utils": utils,
            "examples": examples.get(utils["examples_type"], []),
        }

        with open(PARAMETERS_FILE_NAME, "w") as f:
            f.write(json.dumps(runs_storage))

        return html.Pre(
            f'Done. Results are in folder\n{"/".join(output_file.split("/")[:-1])}'
        )

    def get_prompt(self, utils: Dict, question: str) -> str:
        return super().get_prompt(utils, "***your question***")
