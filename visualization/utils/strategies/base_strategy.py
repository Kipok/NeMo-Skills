from dataclasses import asdict
from flask import current_app
import logging
from typing import Callable, Dict, Iterable, List, Tuple, Union

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.inference.generate_solutions import InferenceConfig
from nemo_skills.inference.prompt.utils import (
    context_templates,
    PromptConfig,
    get_prompt,
)
from nemo_skills.inference.server.model import get_model
import dash_bootstrap_components as dbc
from dash import html

from layouts import (
    get_input_group_layout,
    get_switch_layout,
    get_text_area_layout,
    get_single_prompt_output_layout,
)
from utils.common import (
    examples,
)
from visualization.settings.constants import QUERY_INPUT_TYPE


class ModeStrategies:
    def __init__(self):
        self.sandbox = None
        self.config = current_app.config['prompt_explorer']

    def sandbox_init(self, utils):
        if self.sandbox is None:
            self.sandbox = get_sandbox(
                sandbox_type=self.config['sandbox']['sandbox_type'],
                host=self.config['server']['host'],
                ssh_server=self.config['server']['ssh_server'],
                ssh_key_path=self.config['server']['ssh_key_path'],
            )

    def get_utils_input_layout(
        self,
        inference_condition: Callable[
            [str, Union[str, int, float, bool]], bool
        ],
        prompt_condition: Callable[[str, Union[str, int, float, bool]], bool],
        disabled: bool = False,
        additional_config_values: List[
            Tuple[str, Union[str, int, float, bool]]
        ] = [],
    ) -> List[dbc.AccordionItem]:
        self.config['prompt']['context_templates'] = context_templates[
            self.config["prompt"]["context_type"]
        ]
        input_group_layout = html.Div(
            (
                [
                    get_input_group_layout(name, value, dbc.Input)
                    for name, value in list(self.config["inference"].items())
                    + additional_config_values
                    if inference_condition(name, value)
                ]
                + [
                    get_input_group_layout(param_name, value, dbc.Textarea)
                    for param_name, value in self.config["prompt"].items()
                    if prompt_condition(param_name, value)
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
            examples.get(
                examples_type,
                [{}],
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
        search_layout = (
            [self._get_search_prompt_layout()] if is_prompt_search else []
        )
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
        self.sandbox_init(utils)
        llm = get_model(
            server_type=self.config['server']['server_type'],
            host=self.config['server']['host'],
            sandbox=self.sandbox,
            ssh_server=self.config['server']['ssh_server'],
            ssh_key_path=self.config['server']['ssh_key_path'],
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
        except Exception as e:
            logging.error(f"error during run prompt: {e}")
            return html.Div(f"Got error\n{e}")

        logging.info(f"query's answer: {outputs[0]}")
        color = (
            'green'
            if self.sandbox.is_output_correct(
                outputs[0]['predicted_answer'], params["expected_answer"]
            )
            else "red"
        )
        return html.Div(
            get_single_prompt_output_layout(
                outputs[0]['generated_solution'],
            ),
            style={"border": "2px solid " + color},
        )

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

    def get_prompt(self, utils: Dict, question: str) -> str:
        prompt_config = {
            key: value
            for key, value in utils.items()
            if key in self.config['prompt'].keys()
        }

        prompt = get_prompt(
            PromptConfig(**prompt_config),
            input_dict={
                'question': question,
            },
            context=utils['context_templates'],
            examples=examples.get(
                utils['examples_type'] if utils['examples_type'] else "", []
            ),
        )
        return prompt
