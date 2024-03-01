import itertools
from typing import Dict, List, Optional, Union
from dash import html, dcc
import dash_bootstrap_components as dbc

from utils.common import get_estimated_height, parse_model_answer
from utils.decoration import design_text_output, highlight_code
from visualization.settings.constants import QUERY_INPUT_TYPE


def get_main_page_layout() -> html.Div:
    nav_items = [
        dbc.NavItem(
            dbc.NavLink(
                "Run model",
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
    additional_params: Dict = {},
) -> dbc.Checklist:
    if values is None:
        values = labels
    return dbc.Checklist(
        id=id,
        options=[
            {
                "label": label,
                "value": values,
                "disabled": is_disabled,
            }
            for label, values, is_disabled in itertools.zip_longest(
                labels, values, disabled, fillvalue=False
            )
        ],
        switch=True,
        **additional_params,
    )


def validation_parameters(
    name: str, value: Union[str, int, float]
) -> Dict[str, str]:
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
        height = (
            {'height': get_estimated_height(value)}
            if value != "" and str(value).strip() != ""
            else {}
        )
        additional_params = {
            "style": {
                'width': '100%',
                **height,
            }
        }  # TODO: add js to update height

    return dbc.InputGroup(
        [
            dbc.InputGroupText(name),
            input_function(
                value=(
                    value
                    if value == "" or str(value).strip() != ""
                    else repr(value)[1:-1]
                ),
                id=name,
                **additional_params,
                debounce=True,
            ),
        ],
        className="mb-3",
    )


def get_text_area_layout(
    key: str, value: str, view_mode: bool = False
) -> Union[dbc.Textarea, html.Pre]:
    component = dbc.Textarea
    if view_mode:
        component = html.Pre

    return component(
        **(
            {'children': get_single_prompt_output_layout(value)}
            if view_mode
            else {"value": value}
        ),
        id={
            "type": QUERY_INPUT_TYPE,
            "id": key,
        },
        style={
            'width': '100%',
            'height': get_estimated_height(value),
            'border': "1px solid #dee2e6",
        },  # TODO: add js to update height
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
