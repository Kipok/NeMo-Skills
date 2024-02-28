from dataclasses import asdict
import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

from dash import ALL, html, no_update
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash._callback import NoUpdate
from dash.dependencies import Input, Output, State

from callbacks import app

from settings.config import ConfigHolder
from settings.constants import (
    COMPLETE_MODE,
    ANSWER_FIELD,
    ONE_TEST_MODE,
    PARAMETERS_FILE_NAME,
    QUERY_INPUT_ID,
    QUERY_INPUT_TYPE,
    QUESTION_FIELD,
    RESULTS_PATH,
    WHOLE_DATASET_MODE,
)
from settings.templates import (
    compute_metrics_template,
    evaluate_results_template,
    generate_solution_template,
)
from utils.common import (
    examples,
    get_available_models,
    get_values_from_input_group,
    run_subprocess,
)
from layouts import (
    get_few_shots_by_id_layout,
    get_query_params_layout,
    get_single_prompt_output_layout,
)
from layouts.run_prompt_page_layouts import (
    get_few_shots_div_layout,
    get_query_input_children_layout,
)

from nemo_skills.inference.server.model import get_model
from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.inference.prompt.utils import (
    context_templates,
    get_prompt,
    PromptConfig,
)
from nemo_skills.inference.generate_solutions import InferenceConfig

sandbox = None


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
    logging.info("change_examples_page")
    if not examples_type:
        examples_type = ""
    return [
        get_few_shots_by_id_layout(
            page, examples_type, view_mode and len(view_mode)
        )
    ]


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
    if examples_type not in examples:
        examples[examples_type] = []
    last_page = len(examples[examples_type])
    examples[examples_type].append(
        {
            key: ""
            for key in examples[
                (
                    list(examples.keys())[0]
                    if not len(examples[examples_type])
                    else examples_type
                )
            ][0].keys()
        }
    )
    return (last_page + 1, last_page + 1)


@app.callback(
    [
        Output("few_shots_pagination", "max_value", allow_duplicate=True),
        Output("few_shots_pagination", "active_page", allow_duplicate=True),
        Output(
            "few_shots_pagination_content", "children", allow_duplicate=True
        ),
    ],
    [Input("del_example_button", "n_clicks")],
    [
        State("few_shots_pagination", "active_page"),
        State('examples_type', "value"),
    ],
    prevent_initial_call=True,
)
def del_example(n_clicks: int, page: int, examples_type: str) -> Tuple[
    Union[int, NoUpdate],
    Union[int, NoUpdate],
    Union[Tuple[html.Div], NoUpdate],
    Union[int, NoUpdate],
]:
    if not examples_type:
        examples_type = ""
    if examples_type not in examples:
        examples[examples_type] = []
    last_page = len(examples[examples_type])
    if last_page:
        prev_pagination_page = page if page < last_page else page - 1
        examples[examples_type].pop(page - 1)
        return (
            last_page - 1,
            prev_pagination_page,
            get_few_shots_by_id_layout(prev_pagination_page, examples_type),
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
    logging.info("update_examples")
    if not examples_type:
        examples_type = ""
    if examples_type not in examples:
        examples[examples_type] = []
    last_page = len(examples[examples_type])
    if last_page:
        examples[examples_type][page - 1 if page else 0] = {
            key["id"]: value
            for key, value in zip(page_content_ids, page_content)
        }
    return no_update


@app.callback(
    Output("few_shots_div", "children"),
    Input("examples_type", "value"),
)
def update_examples_type(
    examples_type: str,
) -> NoUpdate | dbc.AccordionItem:
    if not examples_type:
        examples_type = ""
    return get_few_shots_div_layout(examples_type)


@app.callback(
    Output("context_templates", "value"),
    Input("context_type", "value"),
)
def update_context_type(
    context_type: str,
) -> NoUpdate | dbc.AccordionItem:
    return context_templates.get(context_type, no_update)


@app.callback(
    Output("results_content", "children"),
    [Input("run_button", "n_clicks")],
    [
        State("utils_group", "children"),
        State("range_random_seed_mode", "value"),
        State("run_mode_options", "value"),
        State({"type": "query_input", "id": ALL}, "value"),
        State({"type": "query_input", "id": ALL}, "id"),
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
    logging.info('run_test_results')
    global sandbox
    config = ConfigHolder.get_config()

    utils = get_values_from_input_group(utils)
    if "examples_type" in utils and utils["examples_type"] is None:
        utils["examples_type"] = ""
    if run_mode != WHOLE_DATASET_MODE:
        global sandbox
        if sandbox is None:
            sandbox = get_sandbox(
                sandbox_type=config['sandbox']['sandbox_type'],
                host=config['server']['host'],
                ssh_server=config['server']['ssh_server'],
                ssh_key_path=config['server']['ssh_key_path'],
            )
        llm = get_model(
            server_type=config['server']['server_type'],
            host=config['server']['host'],
            sandbox=sandbox,
            ssh_server=config['server']['ssh_server'],
            ssh_key_path=config['server']['ssh_key_path'],
        )

        prompt_config = {
            key: value
            for key, value in utils.items()
            if key in config['prompt'].keys()
        }
        logging.info(query_params_ids)
        question_id = query_params_ids.index(
            json.loads(QUERY_INPUT_ID.format(QUERY_INPUT_TYPE, QUESTION_FIELD))
        )
        try:
            answer_id = query_params_ids.index(
                json.loads(
                    QUERY_INPUT_ID.format(QUERY_INPUT_TYPE, ANSWER_FIELD)
                )
            )
        except:
            answer_id = -1

        prompts = [
            (
                get_prompt(
                    PromptConfig(**prompt_config),
                    input_dict={
                        'question': query_params[question_id],
                    },
                    context=utils['context_templates'],
                    examples=examples.get(
                        (
                            utils['examples_type']
                            if utils['examples_type']
                            else ""
                        ),
                        [],
                    ),
                )
                if run_mode == ONE_TEST_MODE
                else str(query_params[question_id])
            )
        ]

        logging.info(f"query to process: {prompts[0]}")

        inference_cfg = InferenceConfig(
            temperature=utils['temperature'],
            top_k=utils['top_k'],
            top_p=utils['top_p'],
            random_seed=utils['random_seed'],
        )
        outputs = llm(
            prompts=prompts,
            stop_phrases=(
                [
                    (
                        prompt_config["delimiter"]
                        if run_mode == ONE_TEST_MODE
                        else ConfigHolder.get_config()['prompt']["delimiter"]
                    )
                ]
            ),
            **asdict(inference_cfg),
        )

        logging.info(f"query's answer: {outputs[0]}")
        color = (
            'green'
            if answer_id != -1
            and sandbox.is_output_correct(
                outputs[0]['predicted_answer'], query_params[answer_id]
            )
            else "red"
        )
        return html.Div(
            get_single_prompt_output_layout(
                outputs[0]['generated_solution'],
            ),
            style={"border": "2px solid " + color},
        )
    else:
        runs_storage = get_available_models()

        run_index = len(runs_storage)
        metrics_directory = RESULTS_PATH.format(run_index)
        output_file = (
            os.path.join(metrics_directory, "output-greedy.jsonl")
            if utils["output_file"] == '???'
            else utils["output_file"]
        )
        save_metrics_file = (
            os.path.join(metrics_directory, "metrics-greedy.jsonl")
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
            if range_random_mode
            else utils['random_seed']
        )
        random_seed_end = (
            utils['random_seed']
            if range_random_mode
            else utils['random_seed'] + 1
        )
        for random_seed in range(random_seed_start, random_seed_end):
            output_file = (
                output_file
                if not range_random_mode
                else os.path.join(
                    metrics_directory, f"output-rs{random_seed}.jsonl"
                )
            )
            save_metrics_file = (
                save_metrics_file
                if not range_random_mode
                else os.path.join(
                    metrics_directory, f"metrics-rs{random_seed}.jsonl"
                )
            )
            generate_solution_command = generate_solution_template.format(
                output_file=output_file,
                sandbox_host=config['server']['host'],
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
                **config['server'],
            )
            if config['data_file']:
                generate_solution_command += f"++data_file={utils['data_file']}"

            evaluate_results_command = evaluate_results_template.format(
                prediction_jsonl_files=output_file,
                sandbox_host=config['server']['host'],
                ssh_server=config['server']['ssh_server'],
                ssh_key_path=config['server']['ssh_key_path'],
            )

            compute_metrics_command = compute_metrics_template.format(
                prediction_jsonl_files=output_file,
                save_metrics=save_metrics_file,
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
    logging.info("change_mode")
    return get_query_params_layout(run_mode), None


@app.callback(
    [
        Output("query_input_children", "children", allow_duplicate=True),
    ],
    [
        Input("query_search_button", "n_clicks"),
    ],
    [
        State("query_search_input", "value"),
        State("dataset", "value"),
        State("split_name", "value"),
    ],
    prevent_initial_call=True,
)
def prompt_search(
    n_clicks: int, view_mode: bool, index: int, dataset: str, split_name: str
) -> Tuple[Union[List[str], NoUpdate]]:
    logging.info("prompt_search")
    return [
        get_query_input_children_layout(index, dataset, split_name, view_mode)
    ]


@app.callback(
    Output("results_content", "children", allow_duplicate=True),
    [
        Input("preview_button", "n_clicks"),
    ],
    [
        State("run_mode_options", "value"),
        State("utils_group", "children"),
        State({"type": "query_input", "id": ALL}, "value"),
        State({"type": "query_input", "id": ALL}, "id"),
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
    logging.info("preview")
    config = ConfigHolder.get_config()

    utils = get_values_from_input_group(utils)

    question_id = query_params_ids.index(
        {"type": "query_input", "id": "question"}
    )

    if run_mode == WHOLE_DATASET_MODE:
        question = "***your question***"
    else:
        question = query_params[question_id]

    prompt_config = {
        key: value
        for key, value in utils.items()
        if key in config['prompt'].keys()
    }

    prompt = (
        get_prompt(
            PromptConfig(**prompt_config),
            input_dict={
                'question': question,
            },
            context=utils['context_templates'],
            examples=examples.get(
                utils['examples_type'] if utils['examples_type'] else "", []
            ),
        )
        if run_mode != COMPLETE_MODE
        else str(question)
    )
    return html.Div(
        [
            dbc.Checklist(
                id={
                    "type": "view_mode",
                    "id": "preview",
                },
                options=[
                    {
                        "label": "view mode",
                        "value": 1,
                    }
                ],
                switch=True,
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
    logging.info("change_preview_mode")
    return (
        get_single_prompt_output_layout(text)
        if view_mode and len(view_mode)
        else text
    )
