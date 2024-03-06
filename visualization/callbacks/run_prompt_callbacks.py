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

from copy import deepcopy
import json
from typing import Dict, List, Optional, Tuple, Union

import dash_bootstrap_components as dbc
from callbacks import app
from dash import ALL, dcc, html, no_update
from dash._callback import NoUpdate
from dash.dependencies import Input, Output, State
from layouts import (
    get_few_shots_by_id_layout,
    get_query_params_layout,
    get_single_prompt_output_layout,
)
from layouts.base_layouts import get_switch_layout
from settings.constants import (
    ANSWER_FIELD,
    QUERY_INPUT_ID,
    QUERY_INPUT_TYPE,
    QUESTION_FIELD,
)
from utils.common import get_examples, get_test_data, get_values_from_input_group
from utils.strategies.strategy_maker import RunPromptStrategyMaker

from nemo_skills.inference.prompt.utils import context_templates


@app.callback(
    [
        Output("few_shots_pagination_content", "children"),
    ],
    [
        Input("few_shots_pagination", "active_page"),
        Input(
            {
                "type": "view_mode",
                "id": "few_shots_input",
            },
            "value",
        ),
    ],
    [
        State('examples_type', "value"),
    ],
)
def change_examples_page(
    page: int,
    view_mode: bool,
    examples_type: str,
) -> Tuple[Tuple[html.Div], int]:
    if not examples_type:
        examples_type = ""
    return [get_few_shots_by_id_layout(page, examples_type, view_mode and len(view_mode))]


@app.callback(
    [
        Output("few_shots_pagination", "max_value", allow_duplicate=True),
        Output("few_shots_pagination", "active_page", allow_duplicate=True),
    ],
    [Input("add_example_button", "n_clicks")],
    [
        State('examples_type', "value"),
    ],
    prevent_initial_call=True,
)
def add_example(
    n_clicks: int,
    examples_type: str,
) -> Tuple[int, int, int]:
    if not examples_type:
        examples_type = ""
    if examples_type not in get_examples():
        get_examples()[examples_type] = []
    last_page = len(get_examples()[examples_type])
    examples_type_keys = (
        list(get_examples().keys())[0]
        if not len(get_examples()[examples_type])
        else examples_type
    )
    get_examples()[examples_type].append(
        {key: "" for key in get_examples()[examples_type_keys][0].keys()}
    )
    return (last_page + 1, last_page + 1)


@app.callback(
    [
        Output("few_shots_pagination", "max_value", allow_duplicate=True),
        Output("few_shots_pagination", "active_page", allow_duplicate=True),
        Output("few_shots_pagination_content", "children", allow_duplicate=True),
    ],
    [Input("del_example_button", "n_clicks")],
    [
        State("few_shots_pagination", "active_page"),
        State('examples_type', "value"),
        State(
            {
                "type": "view_mode",
                "id": "few_shots_input",
            },
            "value",
        ),
    ],
    prevent_initial_call=True,
)
def del_example(
    n_clicks: int, page: int, examples_type: str, view_mode: List[str]
) -> Tuple[
    Union[int, NoUpdate],
    Union[int, NoUpdate],
    Union[Tuple[html.Div], NoUpdate],
    Union[int, NoUpdate],
]:
    if not examples_type:
        examples_type = ""
    if examples_type not in get_examples():
        get_examples()[examples_type] = []
    last_page = len(get_examples()[examples_type])
    if last_page:
        prev_pagination_page = page if page < last_page else page - 1
        get_examples()[examples_type].pop(page - 1)
        return (
            last_page - 1,
            prev_pagination_page,
            get_few_shots_by_id_layout(prev_pagination_page, examples_type, view_mode),
        )
    return (
        no_update,
        no_update,
        no_update,
    )


@app.callback(
    Output(
        "dummy_output",
        'children',
        allow_duplicate=True,
    ),
    Input({"type": "few_shots_input", "id": ALL}, "value"),
    [
        State("few_shots_pagination", "active_page"),
        State('examples_type', "value"),
        State({"type": "few_shots_input", "id": ALL}, "id"),
    ],
    prevent_initial_call=True,
)
def update_examples(
    page_content: Optional[List[str]],
    page: int,
    examples_type: str,
    page_content_ids: List[int],
) -> NoUpdate:
    if not examples_type:
        examples_type = ""
    if examples_type not in get_examples():
        get_examples()[examples_type] = []
    last_page = len(get_examples()[examples_type])
    if last_page:
        get_examples()[examples_type][page - 1 if page else 0] = {
            key["id"]: value for key, value in zip(page_content_ids, page_content)
        }
    return no_update


@app.callback(
    Output("few_shots_div", "children"),
    Input("examples_type", "value"),
)
def update_examples_type(
    examples_type: str,
) -> Union[NoUpdate, dbc.AccordionItem]:
    if not examples_type:
        examples_type = ""
    size = len(
        get_examples().get(
            examples_type,
            [],
        )
    )
    return RunPromptStrategyMaker().get_strategy().get_few_shots_div_layout(size)


@app.callback(
    Output("context_templates", "value"),
    Input("context_type", "value"),
)
def update_context_type(
    context_type: str,
) -> Union[NoUpdate, dbc.AccordionItem]:
    return context_templates.get(context_type, no_update)


@app.callback(
    Output("utils_group", "children"),
    Input("range_random_seed_mode", "value"),
    State("utils_group", "children"),
)
def update_random_seed_mode(
    range_random_mode: List[int],
    utils: List[Dict],
) -> List[Dict]:
    for i, util in enumerate(utils):
        name = util['props']['children'][0]['props']['children']
        if name in ("random_seed", "start_random_seed", "end_random_seed"):
            break

    if range_random_mode:
        utils[i]['props']['children'][0]['props']['children'] = "start_random_seed"
        utils[i]['props']['children'][1]['props']['id'] = "start_random_seed"
        utils.insert(i + 1, deepcopy(utils[i]))
        utils[i + 1]['props']['children'][0]['props']['children'] = "end_random_seed"
        utils[i + 1]['props']['children'][1]['props']['id'] = "end_random_seed"
        utils[i + 1]['props']['children'][1]['props']['value'] += 1
    else:
        utils[i]['props']['children'][0]['props']['children'] = "random_seed"
        utils[i]['props']['children'][1]['props']['id'] = "random_seed"
        utils.pop(i + 1)
    return utils


@app.callback(
    Output("results_content", "children"),
    [Input("run_button", "n_clicks")],
    [
        State("utils_group", "children"),
        State("range_random_seed_mode", "value"),
        State("run_mode_options", "value"),
        State({"type": QUERY_INPUT_TYPE, "id": ALL}, "value"),
        State({"type": QUERY_INPUT_TYPE, "id": ALL}, "id"),
    ],
    prevent_initial_call=True,
)
def get_run_test_results(
    n_clicks: int,
    utils: List[Dict],
    range_random_mode: List[int],
    run_mode: str,
    query_params: List[str],
    query_params_ids: List[Dict],
) -> Union[html.Div, NoUpdate]:
    if n_clicks is None:
        return no_update

    utils = get_values_from_input_group(utils)
    if "examples_type" in utils and utils["examples_type"] is None:
        utils["examples_type"] = ""

    try:
        question_id = query_params_ids.index(
            json.loads(QUERY_INPUT_ID.format(QUERY_INPUT_TYPE, QUESTION_FIELD))
        )
        answer_id = query_params_ids.index(
            json.loads(QUERY_INPUT_ID.format(QUERY_INPUT_TYPE, ANSWER_FIELD))
        )
        question = query_params[question_id]
        expected_answer = query_params[answer_id]
    except ValueError:
        question = ""
        expected_answer = ""

    return (
        RunPromptStrategyMaker(run_mode)
        .get_strategy()
        .run(
            utils,
            {
                "question": question,
                "expected_answer": expected_answer,
                "range_random_mode": range_random_mode,
            },
        )
    )


@app.callback(
    [
        Output("prompt_params_input", "children", allow_duplicate=True),
        Output("results_content", "children", allow_duplicate=True),
    ],
    [
        Input("run_mode_options", "value"),
    ],
    prevent_initial_call=True,
)
def change_mode(run_mode: str) -> Tuple[List[dbc.AccordionItem], None]:
    return get_query_params_layout(run_mode), None


@app.callback(
    [
        Output("query_input_children", "children", allow_duplicate=True),
    ],
    [
        Input("query_search_button", "n_clicks"),
        Input(
            {
                "type": "view_mode",
                "id": QUERY_INPUT_TYPE,
            },
            "value",
        ),
    ],
    [
        State("query_search_input", "value"),
    ],
    prevent_initial_call=True,
)
def prompt_search(
    n_clicks: int, view_mode: str, index: int
) -> Tuple[Union[List[str], NoUpdate]]:
    key_values = get_test_data(index)[0].items()
    return [
        RunPromptStrategyMaker()
        .get_strategy()
        .get_query_input_children_layout(
            key_values,
            view_mode,
        )
    ]


@app.callback(
    Output("results_content", "children", allow_duplicate=True),
    [
        Input("preview_button", "n_clicks"),
    ],
    [
        State("run_mode_options", "value"),
        State("utils_group", "children"),
        State({"type": QUERY_INPUT_TYPE, "id": ALL}, "value"),
        State({"type": QUERY_INPUT_TYPE, "id": ALL}, "id"),
    ],
    prevent_initial_call=True,
)
def preview(
    n_clicks: int,
    run_mode: str,
    utils: List[Dict],
    query_params: List[str],
    query_params_ids: List[int],
) -> html.Pre:
    utils = get_values_from_input_group(utils)
    try:
        question_id = query_params_ids.index(
            json.loads(QUERY_INPUT_ID.format(QUERY_INPUT_TYPE, QUESTION_FIELD))
        )
        question = query_params[question_id]
    except ValueError:
        question = ""

    prompt = RunPromptStrategyMaker(run_mode).get_strategy().get_prompt(utils, question)
    return html.Div(
        [
            get_switch_layout(
                {
                    "type": "view_mode",
                    "id": "preview",
                },
                ["view mode"],
            ),
            html.Pre(
                prompt,
                id="preview_text",
            ),
            dcc.Store(data=prompt, id="preview_store"),
        ]
    )


@app.callback(
    Output("preview_text", "children", allow_duplicate=True),
    [
        Input(
            {
                "type": "view_mode",
                "id": "preview",
            },
            "value",
        ),
    ],
    [State("preview_store", "data")],
    prevent_initial_call=True,
)
def change_preview_mode(view_mode: bool, text: str) -> html.Pre:
    return get_single_prompt_output_layout(text) if view_mode and len(view_mode) else text
