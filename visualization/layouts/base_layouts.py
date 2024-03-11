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

import itertools
from typing import Dict, List, Optional, Union

import dash_bootstrap_components as dbc
from dash import dcc, html
<<<<<<< HEAD
from settings.constants import QUERY_INPUT_TYPE
from utils.common import (
    parse_model_answer,
)
from utils.decoration import design_text_output, highlight_code
=======
from utils.common import get_estimated_height, parse_model_answer
from utils.decoration import design_text_output, highlight_code

from visualization.settings.constants import QUERY_INPUT_TYPE
>>>>>>> 0035808 ([pre-commit.ci] auto fixes from pre-commit.com hooks)


def get_main_page_layout() -> html.Div:
    nav_items = [
        dbc.NavItem(
            dbc.NavLink(
                "Inference",
                id="run_mode_link",
                href="/",
                active=True,
            )
        ),
        dbc.NavItem(dbc.NavLink("Analyze", id="analyze_link", href="/analyze")),
    ]
    return html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dbc.NavbarSimple(
                children=nav_items,
                brand="Data Explorer",
                sticky="top",
                color="blue",
                dark=True,
                class_name="mb-2",
            ),
            html.Div(id='dummy_output', style={'display': 'none'}, children=""),
            dbc.Container(id="page_content"),
        ]
    )


def get_switch_layout(
    id: Union[Dict, str],
    labels: List[str],
    values: Optional[List[str]] = None,
    disabled: List[bool] = [False],
    is_active: bool = False,
    additional_params: Dict = {},
) -> dbc.Checklist:
    if values is None:
        values = labels
    return dbc.Checklist(
        id=id,
        options=[
            {
                "label": label,
                "value": value,
                "disabled": is_disabled,
            }
<<<<<<< HEAD
            for label, value, is_disabled in itertools.zip_longest(
                labels, values, disabled, fillvalue=False
            )
=======
            for label, values, is_disabled in itertools.zip_longest(labels, values, disabled, fillvalue=False)
>>>>>>> 0035808 ([pre-commit.ci] auto fixes from pre-commit.com hooks)
        ],
        value=[values[0]] if is_active else [],
        switch=True,
        **additional_params,
    )


def validation_parameters(name: str, value: Union[str, int, float]) -> Dict[str, str]:
    parameters = {"type": "text"}
    if str(value).replace(".", "", 1).replace("-", "", 1).isdigit():
        parameters["type"] = "number"

    if str(value).isdigit():
        parameters["min"] = 0

    if "." in str(value) and str(value).replace(".", "", 1).isdigit():
        parameters["min"] = 0
        parameters["max"] = 1 if name != "temperature" else 100
        parameters["step"] = 0.1

    return parameters


def get_input_group_layout(
    name: str,
    value: Union[str, int, float, bool],
    input_function: Union[dbc.Input, dbc.Textarea],
) -> dbc.InputGroup:
    if input_function is dbc.Input:
        additional_params = validation_parameters(name, value)
    else:
<<<<<<< HEAD
=======
        height = {'height': get_estimated_height(value)} if value != "" and str(value).strip() != "" else {}
>>>>>>> 0035808 ([pre-commit.ci] auto fixes from pre-commit.com hooks)
        additional_params = {
            "style": {
                'width': '100%',
            }
        }

    return dbc.InputGroup(
        [
            dbc.InputGroupText(name),
            input_function(
<<<<<<< HEAD
                value=get_utils_field_representation(value),
=======
                value=(value if value == "" or str(value).strip() != "" else repr(value)[1:-1]),
>>>>>>> 0035808 ([pre-commit.ci] auto fixes from pre-commit.com hooks)
                id=name,
                **additional_params,
                debounce=True,
            ),
        ],
        className="mb-3",
    )


def get_text_area_layout(key: str, value: str, view_mode: bool = False) -> Union[dbc.Textarea, html.Pre]:
    component = dbc.Textarea
    if view_mode:
        component = html.Pre

    return component(
        **({'children': get_single_prompt_output_layout(value)} if view_mode else {"value": value}),
        id={
            "type": QUERY_INPUT_TYPE,
            "id": key,
        },
        style={
            'width': '100%',
            'border': "1px solid #dee2e6",
        },
    )


def get_single_prompt_output_layout(answer: str) -> List[html.Div]:
    parsed_answers = parse_model_answer(answer)
    return [
        item
        for parsed_answer in parsed_answers
        for item in [
            design_text_output(parsed_answer["explanation"]),
            (
                highlight_code(
                    parsed_answer["code"],
                )
                if parsed_answer["code"]
                else ""
            ),
            (
                design_text_output(
                    parsed_answer["output"],
                    style=(
                        {
                            "border": "1px solid black",
                            "background-color": "#9CADF0",
                            "marginBottom": "10px",
                        }
                        if 'wrong_code_block' not in parsed_answer
                        else {
                            "border": "1px solid red",
                            "marginBottom": "10px",
                        }
                    ),
                )
                if parsed_answer["output"] is not None
                else ""
            ),
        ]
        if item != ""
    ]


def get_results_content_layout(
    text: str, content: str = None, style={}, switch_is_active: bool = False
) -> html.Div:
    return html.Div(
        [
            get_switch_layout(
                {
                    "type": "view_mode",
                    "id": "results_content",
                },
                ["view mode"],
                is_active=switch_is_active,
            ),
            html.Pre(
                content if content else text,
                id="results_content_text",
                style={'margin-bottom': '10px'},
            ),
            dcc.Store(data=text, id="text_store"),
        ],
        style=style,
    )


def get_utils_field_representation(value) -> str:
    return value if value == "" or str(value).strip() != "" else repr(value)[1:-1]
