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
from typing import Dict, Iterable, List, Optional, Union

import dash_bootstrap_components as dbc
from dash import dcc, html
from flask import current_app
from settings.constants import ANSI, CODE, COMPARE, LATEX, MARKDOWN, SEPARATOR_DISPLAY, SEPARATOR_ID, UNDEFINED
from settings.ui_config import UIConfig
from utils.common import parse_model_answer
from utils.decoration import color_text_diff, design_text_output, highlight_code



# def get_main_page_layout() -> html.Div:
#     nav_items = [
#         dbc.NavItem(
#             dbc.NavLink(
#                 "Inference",
#                 id="run_mode_link",
#                 href="/",
#                 active=True,
#                 # color=UIConfig.COLORS["text"],
#             )
#         ),
#         dbc.NavItem(dbc.NavLink("Analyze", id="analyze_link", href="/analyze", style=UIConfig.COLORS["text"])),
#     ]
#     return html.Div(
#         [
#             dcc.Location(id="url", refresh=False),
#             dbc.NavbarSimple(
#                 children=nav_items,
#                 brand="NeMo Inspector",
#                 # brand_style={'font-weight': '600', 'letter-spacing': '0.5px', "font-size": "1.2rem"},
#                 sticky="top",
#                 # color="primary",  # Using Bootstrap's primary instead of plain "blue"
#                 dark=True,
#                 style=UIConfig.STYLES["nav"],
#                 class_name="mb-2 shadow-sm",  # Added subtle shadow
#                 # padding="1rem",  # More comfortable padding
#             ),
#             #     children=nav_items,
#             #     brand="NeMo Inspector",
#             #     sticky="top",
#             #     color="blue",
#             #     dark=True,
#             #     class_name="mb-2",
#             # ),
#             dbc.Container(id="page_content"),
#             dbc.Container(id="js_trigger", style={'display': 'none'}, children=""),
#             dbc.Container(id="js_container"),
#             dbc.Container(id='dummy_output', style={'display': 'none'}, children=""),
#         ]
#     )

def get_main_page_layout() -> html.Div:
    # Define modern dark theme colors and styles
    THEME = {
        "background": "#000000",
        "surface": "#1A1A1A",
        "primary": "#9D5CFF",
        "text": "#FFFFFF",
    }
    
    # Navigation items with modern styling
    nav_items = [
        dbc.NavItem(
            dbc.NavLink(
                "Inference",
                id="run_mode_link",
                href="/",
                active=True,
                style={
                    "color": THEME["text"],
                    # "padding": "0.5rem 1rem",
                    "margin": "0 0.5rem",
                    "transition": "all 0.2s ease-in-out",
                    "border-radius": "6px",
                }
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "Analyze",
                id="analyze_link",
                href="/analyze",
                style={
                    # "padding": "0.5rem 1rem",
                    "margin": "0 0.5rem",
                    "transition": "all 0.2s ease-in-out",
                    "border-radius": "6px",
                }
            )
        ),
    ]

    return html.Div(
        [
            dcc.Location(id="url", refresh=False),
            # Modern navbar with gradient background
            dbc.NavbarSimple(
                children=nav_items,
                brand="NeMo Inspector",
                brand_style={
                    "fontSize": "1.25rem",
                    "fontWeight": "600",
                    "letterSpacing": "0.5px",
                    "color": THEME["text"]
                },
                sticky="top",
                style={
                    "background": f"linear-gradient(to right, {THEME['surface']}, {THEME['background']})",
                    "borderBottom": "1px solid rgba(255,255,255,0.1)",
                    "padding": "1rem 0",
                },
                dark=True,
                class_name="mb-4",
            ),
            # Main content container with proper spacing
            dbc.Container(
                id="page_content",
                style={
                    "minHeight": "calc(100vh - 80px)",  # Full height minus navbar
                    "padding": "2rem 0",
                },
                class_name="px-4",
            ),
            # Hidden containers with consistent styling
            dbc.Container(
                id="js_trigger",
                style={'display': 'none'},
                children="",
                class_name="p-0"
            ),
            dbc.Container(
                id="js_container",
                class_name="p-0"
            ),
            dbc.Container(
                id='dummy_output',
                style={'display': 'none'},
                children="",
                class_name="p-0"
            ),
        ],
        style={
            "backgroundColor": THEME["background"],
            "color": THEME["text"],
            "minHeight": "100vh",
        }
    )

def get_switch_layout(
    id: Union[Dict, str],
    labels: List[str],
    values: Optional[List[str]] = None,
    disabled: List[bool] = [False],
    is_active: bool = False,
    chosen_values: Optional[List[str]] = None,
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
            for label, value, is_disabled in itertools.zip_longest(labels, values, disabled, fillvalue=False)
        ],
        value=(chosen_values if chosen_values else [values[0]] if is_active else []),
        **additional_params,
    )


def get_selector_layout(options: Iterable, id: str, value: str = "") -> dbc.Select:
    if value not in options:
        options = [value] + list(options)
    return dbc.Select(
        id=id,
        options=[
            {
                "label": str(value),
                "value": value,
            }
            for value in options
        ],
        value=str(value),
    )


def get_text_area_layout(
    id: str, value: str, text_modes: List[str] = [], editable: bool = False
) -> Union[dbc.Textarea, html.Pre]:
    if editable and text_modes == []:
        component = dbc.Textarea
        children = {"value": value}
    else:
        component = html.Pre
        children = {"children": get_single_prompt_output_layout(value, text_modes)}
    return component(
        **children,
        id=id,
        style={
            'width': '100%',
            'border': "1px solid #dee2e6",
        },
    )


def get_single_prompt_output_layout(
    answer: str, text_modes: List[str] = [CODE, LATEX, ANSI], compare_to: str = ""
) -> List[html.Div]:
    parsed_answers = (
        parse_model_answer(answer) if CODE in text_modes else [{"explanation": answer, "code": None, "output": None}]
    )
    parsed_compared_answers = (
        (
            parse_model_answer(compare_to)
            if CODE in text_modes
            else [{"explanation": compare_to, "code": None, "output": None}]
        )
        if COMPARE in text_modes
        else parsed_answers
    )

    items = []
    styles = {
        "explanation": {'default': {}, 'wrong': {}},
        "code": {'default': {}, 'wrong': {}},
        "output": {
            "default": {
                "border": "1px solid black",
                "background-color": "#cdd4f1c8",
                "marginBottom": "10px",
                "marginTop": "-6px",
            },
            'wrong': {
                "border": "1px solid red",
                "marginBottom": "10px",
                "marginTop": "-6px",
            },
        },
    }

    functions = {"explanation": design_text_output, "code": highlight_code, "output": design_text_output}

    def check_existence(array: List[Dict[str, str]], i: int, key: str):
        return i < len(array) and key in array[i] and array[i][key]

    for i in range(max(len(parsed_answers), len(parsed_compared_answers))):
        for key in ["explanation", "code", "output"]:
            if check_existence(parsed_answers, i, key) or check_existence(parsed_compared_answers, i, key):
                diff = color_text_diff(
                    parsed_answers[i][key] if check_existence(parsed_answers, i, key) else "",
                    parsed_compared_answers[i][key] if check_existence(parsed_compared_answers, i, key) else "",
                )
                style_type = (
                    'default'
                    if not check_existence(parsed_answers, i, key) or 'wrong_code_block' not in parsed_answers[i][key]
                    else 'wrong'
                )
                style = styles[key][style_type]
                item = functions[key](diff, style=style, text_modes=text_modes)
                items.append(item)
    return items


def get_text_modes_layout(id: str, is_formatted: bool = True):
    return get_switch_layout(
        id={
            "type": "text_modes",
            "id": id,
        },
        labels=[CODE, LATEX, MARKDOWN, ANSI],
        chosen_values=[CODE, LATEX, ANSI] if is_formatted else [],
        additional_params={
            "style": {
                "display": "inline-flex",
                "flex-wrap": "wrap",
            },
            "inputStyle": {'margin-left': '-10px'},
            "labelStyle": {'margin-left': '3px'},
        },
    )


def get_results_content_layout(
    text: str, content: str = None, style: Dict = {}, is_formatted: bool = False
) -> html.Div:
    return html.Div(
        [
            get_text_modes_layout("results_content", is_formatted),
            html.Pre(
                content if content else text,
                id="results_content_text",
                style={'margin-bottom': '10px'},
            ),
            dcc.Store(data=text, id="text_store"),
        ],
        style=style,
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


def get_input_group_layout(name: str, value: Union[str, int, float, bool]) -> dbc.InputGroup:
    input_function = dbc.Textarea
    additional_params = {
        "style": {
            'width': '100%',
        },
        "debounce": True,
    }
    if name.split(SEPARATOR_DISPLAY)[-1] in current_app.config['nemo_inspector']['types'].keys():
        input_function = get_selector_layout
        additional_params = {
            "options": current_app.config['nemo_inspector']['types'][name.split(SEPARATOR_DISPLAY)[-1]],
        }
        if value is None:
            value = UNDEFINED
    elif isinstance(value, bool):
        input_function = get_selector_layout
        additional_params = {"options": [True, False]}
    elif isinstance(value, (float, int)):
        input_function = dbc.Input
        additional_params = validation_parameters(name, value)
        additional_params["debounce"] = True

    return dbc.InputGroup(
        [
            dbc.InputGroupText(name),
            input_function(
                value=get_utils_field_representation(value),
                id=name.replace(SEPARATOR_DISPLAY, SEPARATOR_ID),
                **additional_params,
            ),
        ],
        className="mb-3",
    )


def get_utils_field_representation(value: Union[str, int, float, bool], key: str = "") -> str:
    return (
        UNDEFINED
        if value is None and key.split(SEPARATOR_ID)[-1] in current_app.config['nemo_inspector']['types']
        else value if value == "" or str(value).strip() != "" else repr(value)[1:-1]
    )
