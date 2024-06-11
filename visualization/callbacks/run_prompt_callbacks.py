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
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dash_bootstrap_components as dbc
from callbacks import app
from dash import ALL, html, no_update
from dash._callback import NoUpdate
from dash.dependencies import Input, Output, State
from flask import current_app
from layouts import (
    get_few_shots_by_id_layout,
    get_query_params_layout,
    get_results_content_layout,
    get_single_prompt_output_layout,
    get_utils_field_representation,
)
from settings.constants import (
    FEW_SHOTS_INPUT,
    QUERY_INPUT_TYPE,
    SEPARATOR_DISPLAY,
    SEPARATOR_ID,
    UNDEFINED,
)
from utils.common import (
    extract_query_params,
    get_examples,
    get_test_data,
    get_utils_from_config,
    get_values_from_input_group,
)
from utils.strategies.strategy_maker import RunPromptStrategyMaker

from nemo_skills.inference.prompt.utils import (
    FewShotExamplesConfig,
    PromptConfig,
    context_templates,
    get_prompt_config,
    prompt_types,
)


@app.callback(
    [
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    Input("prompt_params_input", "active_item"),
    State("js_trigger", "children"),
    prevent_initial_call=True,
)
def trigger_js(active_item: str, js_trigger: str) -> Tuple[str, str]:
    return "", js_trigger + " "


@app.callback(
    [
        Output("few_shots_div", "children"),
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    Input("examples_type", "value"),
    State("js_trigger", "children"),
    prevent_initial_call=True,
)
def update_examples_type(examples_type: str, js_trigger: str) -> Union[NoUpdate, dbc.AccordionItem]:
    if not examples_type:
        examples_type = ""
    size = len(
        get_examples().get(
            examples_type,
            [],
        )
    )
    return (
        RunPromptStrategyMaker().get_strategy().get_few_shots_div_layout(size),
        "",
        js_trigger + " ",
    )


@app.callback(
    [
        Output("few_shots_pagination_content", "children"),
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    [
        Input("few_shots_pagination", "active_page"),
        Input(
            {
                "type": "view_mode",
                "id": FEW_SHOTS_INPUT,
            },
            "value",
        ),
        Input('examples_type', "value"),
    ],
    State("js_trigger", "children"),
    prevent_initial_call=True,
)
def change_examples_page(
    page: int,
    view_mode: List[str],
    examples_type: str,
    js_trigger: str,
) -> Tuple[Tuple[html.Div], int]:
    if not examples_type:
        examples_type = ""
    return (
        get_few_shots_by_id_layout(page, examples_type, view_mode and len(view_mode)),
        '',
        js_trigger + '',
    )


@app.callback(
    [
        Output("few_shots_pagination", "max_value", allow_duplicate=True),
        Output("few_shots_pagination", "active_page", allow_duplicate=True),
    ],
    Input("add_example_button", "n_clicks"),
    State('examples_type', "value"),
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
    examples_type_keys = list(get_examples().keys())[0] if not len(get_examples()[examples_type]) else examples_type
    get_examples()[examples_type].append({key: "" for key in get_examples()[examples_type_keys][0].keys()})
    return (last_page + 1, last_page + 1)


@app.callback(
    [
        Output("few_shots_pagination", "max_value", allow_duplicate=True),
        Output("few_shots_pagination", "active_page", allow_duplicate=True),
        Output("few_shots_pagination_content", "children", allow_duplicate=True),
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    Input("del_example_button", "n_clicks"),
    [
        State("few_shots_pagination", "active_page"),
        State('examples_type', "value"),
        State(
            {
                "type": "view_mode",
                "id": FEW_SHOTS_INPUT,
            },
            "value",
        ),
        State("js_trigger", "children"),
    ],
    prevent_initial_call=True,
)
def del_example(
    n_clicks: int,
    page: int,
    examples_type: str,
    view_mode: List[str],
    js_trigger: str,
) -> Tuple[
    Union[int, NoUpdate],
    Union[int, NoUpdate],
    Union[Tuple[html.Div], NoUpdate],
    Union[int, NoUpdate],
]:
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update
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
            '',
            js_trigger + ' ',
        )
    return no_update, no_update, no_update, no_update, no_update


@app.callback(
    Output(
        "dummy_output",
        'children',
        allow_duplicate=True,
    ),
    Input({"type": FEW_SHOTS_INPUT, "id": ALL}, "value"),
    [
        State({"type": FEW_SHOTS_INPUT, "id": ALL}, "id"),
        State("few_shots_pagination", "active_page"),
        State('examples_type', "value"),
        State(
            {
                "type": "view_mode",
                "id": FEW_SHOTS_INPUT,
            },
            "value",
        ),
    ],
    prevent_initial_call=True,
)
def update_examples(
    page_content: Optional[List[str]],
    page_content_ids: List[int],
    page: int,
    examples_type: str,
    view_mode: List[str],
) -> NoUpdate:
    if view_mode and len(view_mode) or not page_content:
        return no_update

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
    [
        Output(
            SEPARATOR_ID.join(field.split(SEPARATOR_DISPLAY)),
            "value",
            allow_duplicate=True,
        )
        for field in get_utils_from_config(
            {"prompt": asdict(PromptConfig(few_shot_examples=FewShotExamplesConfig()))}
        ).keys()
    ]
    + [
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    Input("prompt_type", "value"),
    State("js_trigger", "children"),
    prevent_initial_call=True,
)
def update_prompt_type(prompt_type: str, js_trigger: str) -> Union[NoUpdate, dbc.AccordionItem]:
    if prompt_type not in map(lambda name: name.split('.')[0], prompt_types):
        output_len = len(get_utils_from_config(asdict(PromptConfig(few_shot_examples=FewShotExamplesConfig()))).keys())
        return [no_update] * (output_len + 2)
    prompt_config = get_prompt_config(prompt_type)
    current_app.config['data_explorer']['prompt']['stop_phrases'] = list(prompt_config.stop_phrases)
    return [
        get_utils_field_representation(value, key)
        for key, value in get_utils_from_config(asdict(prompt_config)).items()
    ] + ['', js_trigger + " "]


@app.callback(
    [
        Output("context_template", "value", allow_duplicate=True),
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    Input("context_type", "value"),
    State("js_trigger", "children"),
    prevent_initial_call=True,
)
def update_context_type(context_type: str, js_trigger: str) -> Union[NoUpdate, dbc.AccordionItem]:
    return context_templates.get(context_type, no_update), "", js_trigger + " "


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
    [
        Output("results_content", "children"),
        Output("loading_container", "children", allow_duplicate=True),
    ],
    Input("run_button", "n_clicks"),
    [
        State("utils_group", "children"),
        State("range_random_seed_mode", "value"),
        State("run_mode_options", "value"),
        State({"type": QUERY_INPUT_TYPE, "id": ALL}, "value"),
        State({"type": QUERY_INPUT_TYPE, "id": ALL}, "id"),
        State("loading_container", "children"),
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
    loading_container: str,
) -> Union[Tuple[html.Div, str], Tuple[NoUpdate, NoUpdate]]:
    if n_clicks is None:
        return no_update, no_update

    utils = get_values_from_input_group(utils)
    if "examples_type" in utils and utils["examples_type"] is None:
        utils["examples_type"] = ""

    return (
        RunPromptStrategyMaker(run_mode)
        .get_strategy()
        .run(
            utils,
            {
                **extract_query_params(query_params_ids, query_params),
                "range_random_mode": range_random_mode,
            },
        ),
        loading_container + " ",
    )


@app.callback(
    [
        Output("prompt_params_input", "children", allow_duplicate=True),
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
        Output("results_content", "children", allow_duplicate=True),
    ],
    Input("run_mode_options", "value"),
    [
        State("utils_group", "children"),
        State("js_trigger", "children"),
    ],
    prevent_initial_call=True,
)
def change_mode(run_mode: str, utils: List[Dict], js_trigger: str) -> Tuple[List[dbc.AccordionItem], None]:
    utils = get_values_from_input_group(utils)
    return (
        get_query_params_layout(run_mode, utils.get('data_file', UNDEFINED)),
        "",
        js_trigger + ' ',
        None,
    )


@app.callback(
    [
        Output("query_input_children", "children", allow_duplicate=True),
        Output("query_store", "data", allow_duplicate=True),
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    [
        Input("query_search_button", "n_clicks"),
        Input("data_file", "value"),
        Input("run_mode_options", "value"),
    ],
    [
        State("query_search_input", "value"),
        State("js_trigger", "children"),
    ],
    prevent_initial_call=True,
)
def prompt_search(
    n_clicks: int, data_file: str, run_mode: str, index: int, js_trigger: str
) -> Tuple[Union[List[str], NoUpdate]]:
    query_data = get_test_data(index, data_file)[0]
    return (
        RunPromptStrategyMaker()
        .get_strategy()
        .get_query_input_children_layout(
            query_data,
            [],
        ),
        query_data,
        "",
        js_trigger + " ",
    )


@app.callback(
    [
        Output("query_input_children", "children", allow_duplicate=True),
        Output("query_store", "data", allow_duplicate=True),
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    [
        Input(
            {
                "type": "view_mode",
                "id": QUERY_INPUT_TYPE,
            },
            "value",
        ),
    ],
    [
        State("query_store", "data"),
        State({"type": QUERY_INPUT_TYPE, "id": ALL}, "value"),
        State({"type": QUERY_INPUT_TYPE, "id": ALL}, "id"),
        State("js_trigger", "children"),
    ],
    prevent_initial_call=True,
)
def change_prompt_search_mode(
    view_mode: List[str],
    query_store: Dict[str, str],
    query_params: List[str],
    query_params_ids: List[int],
    js_trigger: str,
) -> Tuple[Union[List[str], NoUpdate]]:
    if view_mode and len(view_mode):
        query_store = extract_query_params(query_params_ids, query_params)

    return (
        RunPromptStrategyMaker()
        .get_strategy()
        .get_query_input_children_layout(
            query_store,
            view_mode,
        ),
        query_store,
        "",
        js_trigger + " ",
    )


@app.callback(
    Output("results_content", "children", allow_duplicate=True),
    Input("preview_button", "n_clicks"),
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

    prompt = (
        RunPromptStrategyMaker(run_mode)
        .get_strategy()
        .get_prompt(utils, extract_query_params(query_params_ids, query_params))
    )
    return get_results_content_layout(str(prompt))


@app.callback(
    Output("results_content_text", "children", allow_duplicate=True),
    Input(
        {
            "type": "view_mode",
            "id": "results_content",
        },
        "value",
    ),
    State("text_store", "data"),
    prevent_initial_call=True,
)
def change_results_content_mode(view_mode: List[str], text: str) -> html.Pre:
    return get_single_prompt_output_layout(text) if view_mode and len(view_mode) else text
