from typing import Dict, List

from dash import html, dcc
import dash_bootstrap_components as dbc

from layouts.table_layouts import (
    get_filter_layout,
    get_selector_layout,
    get_sorting_layout,
)
from layouts.base_layouts import get_switch_layout
from settings.constants import DELETE, GENERAL_STATS
from utils.common import get_available_models


def get_models_options_layout() -> dbc.Accordion:
    runs_storage = get_available_models()
    items = [
        dbc.AccordionItem(
            dbc.Accordion(
                [
                    get_utils_layout(values["utils"]),
                    get_few_shots_layout(values["examples"]),
                ],
                start_collapsed=True,
                always_open=True,
            ),
            title=model,
        )
        for model, values in runs_storage.items()
    ]
    models_options = dbc.Accordion(
        items,
        start_collapsed=True,
        always_open=True,
    )
    return dbc.Accordion(
        dbc.AccordionItem(
            models_options,
            title="Models options",
        ),
        start_collapsed=True,
        always_open=True,
    )


def get_utils_layout(utils: Dict) -> dbc.AccordionItem:
    input_groups = [
        dbc.InputGroup(
            [
                html.Pre(f"{name}: ", className="mr-2"),
                html.Pre(
                    (
                        value
                        if value == "" or str(value).strip() != ""
                        else repr(value)[1:-1]
                    ),
                    className="mr-2",
                ),
            ],
            className="mb-3",
        )
        for name, value in utils.items()
    ]
    return dbc.AccordionItem(
        html.Div(input_groups),
        title="Utils",
    )


def get_few_shots_layout(examples: List[Dict]) -> dbc.AccordionItem:
    example_layout = lambda example: [
        html.Div(
            [
                dcc.Markdown(f'**{name}**'),
                html.Pre(value),
            ]
        )
        for name, value in example.items()
    ]
    examples_layout = [
        dbc.Accordion(
            dbc.AccordionItem(
                example_layout(example),
                title=f"example {id}",
            ),
            start_collapsed=True,
            always_open=True,
        )
        for id, example in enumerate(examples)
    ]
    return dbc.AccordionItem(
        html.Div(examples_layout),
        title="Few shots",
    )


def get_compare_test_layout() -> html.Div:
    return html.Div(
        [
            get_models_options_layout(),
            dbc.InputGroup(
                [
                    get_sorting_layout(),
                    get_filter_layout(),
                    get_add_stats_layout(),
                    dbc.Button(
                        "Save dataset",
                        id="save_dataset",
                    ),
                    dbc.Button(
                        "+",
                        id="add_model",
                        outline=True,
                        color="primary",
                        className="me-1",
                    ),
                    get_selector_layout(
                        get_available_models().keys(),
                        'base_model_answers_selector',
                    ),
                ]
            ),
            html.Pre(id="filtering_container"),
            html.Pre(id="sorting_container"),
            html.Div(  # TODO spinner
                children=[],
                id="compare_models_rows",
            ),
        ],
    )


def get_stats_text(general_stats: bool = False, delete: bool = False):
    if delete:
        return "write name of the statistic you want to delete"
    else:
        if general_stats:
            return (
                "The last line should be a dictionary with new custom stats.\n"
                "The keys will be the stat names, and the values should be\n"
                "functions of arrays with data from all files, where first dimension\n"
                "is a question number and second is a file number (both sorted and filtered).\n"
                "For example\n"
                "def my_func(datas):\n"
                "    correct_responses = 0\n"
                "    for question_data in datas:\n"
                "        for file_data in question_data:\n"
                "            correct_responses += file_data['is_correct']\n"
                "    return correct_responses\n"
                "{'correct_responses': my_func}"
            )
        else:
            return (
                "The last line should be a dictionary with new custom stats.\n"
                "The keys will be the stat names, and the values should be\n"
                "functions of arrays with data from all files.\n"
                "Avoid using the same name as the existing stats or json fields.\n"
                "For example:\n"
                "def my_func(datas):\n"
                "    errors_set = set()\n"
                "    for data in datas:\n"
                "        errors_set.add(data.get('error_message'))\n"
                "    return len(errors_set)\n"
                "{'my_custom_stats': my_func}"
            )


def get_add_stats_layout() -> html.Div:
    modal_header = dbc.ModalHeader(
        [
            dbc.ModalTitle("set up your stats"),
            get_switch_layout(
                id="stats_modes",
                labels=["general stats", "delete mode"],
                values=[GENERAL_STATS, DELETE],
                additional_params={"inline": True},
            ),
        ],
        close_button=True,
    )
    modal_body = dbc.ModalBody(
        html.Div(
            [
                html.Pre(get_stats_text(), id="stats_text"),
                dbc.Textarea(id="new_stats_input"),
            ]
        )
    )
    modal_footer = dbc.ModalFooter(
        dbc.Button(
            "Apply",
            id="apply_new_stats",
            className="ms-auto",
            n_clicks=0,
        )
    )
    return html.Div(
        [
            dbc.Button("Stats", id="set_new_stats_button"),
            dbc.Modal(
                [
                    modal_header,
                    modal_body,
                    modal_footer,
                ],
                size="lg",
                id="new_stats",
                centered=True,
                is_open=False,
            ),
        ],
        style={'display': 'inline-block'},
    )
