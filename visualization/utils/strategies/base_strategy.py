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

import logging
from copy import deepcopy
from dataclasses import asdict
from typing import Callable, Dict, Iterable, List, Tuple, Union

import dash_bootstrap_components as dbc
import requests
from dash import html
from flask import current_app
from layouts import (
    get_input_group_layout,
    get_results_content_layout,
    get_selector_layout,
    get_single_prompt_output_layout,
    get_switch_layout,
    get_text_area_layout,
)
from settings.constants import QUERY_INPUT_TYPE, UNDEFINED
from utils.common import get_examples

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.inference.generate_solutions import InferenceConfig
from nemo_skills.inference.prompt.utils import PromptConfig, context_templates, get_prompt
from nemo_skills.inference.server.model import get_model


class ModeStrategies:
    def __init__(self):
        self.sandbox = None
        self.config = deepcopy(current_app.config['data_explorer'])

    def sandbox_init(self):
        if self.sandbox is None:
            self.sandbox = get_sandbox(
                **self.config['sandbox'],
            )

    def get_utils_input_layout(
        self,
        inference_condition: Callable[[str, Union[str, int, float, bool]], bool],
        prompt_condition: Callable[[str, Union[str, int, float, bool]], bool],
        disabled: bool = False,
        additional_config_values: List[Tuple[str, Union[str, int, float, bool]]] = [],
    ) -> List[dbc.AccordionItem]:
        self.config['prompt']['context_templates'] = context_templates[self.config["prompt"]["context_type"]]
        input_group_layout = html.Div(
            (
                [
                    get_input_group_layout(name, value, dbc.Input)
                    for name, value in list(self.config["inference"].items()) + additional_config_values
                    if inference_condition(name, value)
                ]
                + [
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText(param_name),
                            get_selector_layout(self.config['types'][param_name], param_name, value),
                        ],
                        style={"margin-bottom": "15px"},
                    )
                    for param_name, value in self.config["prompt"].items()
                    if prompt_condition(param_name, value) and param_name in self.config['types'].keys()
                ]
                + [
                    get_input_group_layout(param_name, value, dbc.Textarea)
                    for param_name, value in self.config["prompt"].items()
                    if prompt_condition(param_name, value) and 'type' not in param_name
                ]
            ),
            id="utils_group",
        )
        utils_group_layout = [
            dbc.AccordionItem(
                html.Div(
                    [
                        get_switch_layout(
                            id="range_random_seed_mode",
                            labels=["use random seed range"],
                            disabled=[disabled],
                        ),
                        input_group_layout,
                    ]
                ),
                title="Utils",
            )
        ]
        self.config['prompt'].pop("context_templates")
        return utils_group_layout

    def get_few_shots_input_layout(self) -> List[dbc.AccordionItem]:
        examples_type = self.config["prompt"]["examples_type"]
        size = len(
            get_examples().get(
                examples_type,
                [],
            )
        )
        return [
            dbc.AccordionItem(
                self.get_few_shots_div_layout(size),
                title="Few shots",
                id="few_shots_group",
            )
        ]

    def get_query_input_layout(
        self, key_values: List[Tuple[str, str]], is_prompt_search: bool = True
    ) -> List[dbc.AccordionItem]:
        switch_layout = [
            get_switch_layout(
                {
                    "type": "view_mode",
                    "id": QUERY_INPUT_TYPE,
                },
                ["view mode"],
            )
        ]
        search_layout = [self._get_search_prompt_layout()] if is_prompt_search else []
        query_input = [
            html.Div(
                self.get_query_input_children_layout(key_values),
                id="query_input_children",
            )
        ]
        return [
            dbc.AccordionItem(
                html.Div(
                    switch_layout + search_layout + query_input,
                ),
                title="Input",
                id="query_input_content",
            )
        ]

    def get_query_input_children_layout(
        self, key_values: Iterable[Tuple[str, str]], view_mode: bool = False
    ) -> List[dbc.InputGroup]:
        return [
            dbc.InputGroup(
                [
                    dbc.InputGroupText(key),
                    get_text_area_layout(key, value, view_mode),
                ],
                className="mb-3",
            )
            for key, value in key_values
        ]

    def get_few_shots_div_layout(self, size: int) -> html.Div:
        return html.Div(
            [
                html.Div(
                    [
                        dbc.Pagination(
                            id="few_shots_pagination",
                            max_value=size,
                            active_page=1,
                        ),
                        dbc.Button(
                            "add example",
                            id="add_example_button",
                            outline=True,
                            size="sm",
                            color="primary",
                            className="me-1",
                        ),
                        dbc.Button(
                            "delete current example",
                            id="del_example_button",
                            outline=True,
                            size="sm",
                            color="primary",
                            className="me-1",
                        ),
                        get_switch_layout(
                            id={
                                "type": "view_mode",
                                "id": "few_shots_input",
                            },
                            labels=["view mode"],
                        ),
                    ]
                ),
                dbc.Container(id="few_shots_pagination_content"),
            ],
            id="few_shots_div",
        )

    def run(self, utils: Dict, params: Dict) -> html.Div:
        self.sandbox_init()
        llm = get_model(
            **self.config['server'],
            sandbox=self.sandbox,
        )

        logging.info(f"query to process: {params['prompts'][0]}")

        inference_cfg = InferenceConfig(
            temperature=utils['temperature'],
            top_k=utils['top_k'],
            top_p=utils['top_p'],
            random_seed=utils['random_seed'],
        )
        try:
            outputs = llm(
                prompts=params['prompts'],
                stop_phrases=[utils["delimiter"]],
                **asdict(inference_cfg),
            )
        except requests.exceptions.ConnectionError as e:
            return self._get_connection_error_message()
        except Exception as e:
            logging.error(f"error during run prompt: {e}")
            logging.error(f"error type: {type(e)}")
            return html.Div(f"Got error\n{e}")

        logging.info(f"query's answer: {outputs[0]}")

        try:
            color = (
                'green'
                if self.sandbox.is_output_correct(outputs[0]['predicted_answer'], params["expected_answer"])
                else "red"
            )
        except Exception as e:
            color = 'grey'

        return get_results_content_layout(
            outputs[0]['generated_solution'],
            get_single_prompt_output_layout(
                outputs[0]['generated_solution'],
            ),
            style={"border": "2px solid " + color},
            switch_is_active=True,
        )

    def get_prompt(self, utils: Dict, question: str) -> str:
        prompt_config = {key: value for key, value in utils.items() if key in self.config['prompt'].keys()}

        prompt_config['context'] = utils['context_templates']
        prompt_config['examples'] = get_examples().get(utils['examples_type'] if utils['examples_type'] else "", [])

        prompt = get_prompt(
            PromptConfig(**prompt_config),
            input_dict={
                'question': question,
            },
        )
        return prompt

    def _get_search_prompt_layout(self) -> dbc.InputGroup:
        return dbc.InputGroup(
            [
                dbc.InputGroupText("Index of test"),
                dbc.Input(
                    value=1,
                    id="query_search_input",
                    type="number",
                    size="sm",
                ),
                dbc.Button(
                    "Search",
                    id="query_search_button",
                    outline=True,
                    size="sm",
                    color="primary",
                    className="me-1",
                ),
            ],
            className="mb-3",
        )

    def _get_connection_error_message(self):
        return html.Div(
            html.P(
                [
                    "Could not connect to the server. " "Please check that the server is running (look at ",
                    html.A(
                        "inference.md",
                        href="https://github.com/Kipok/NeMo-Skills/blob/main/docs/inference.md",
                    ),
                    " for more information). ",
                    "Also check that you have provided correct host, ssh_key_path and ssh_server parameters",
                ]
            )
        )
