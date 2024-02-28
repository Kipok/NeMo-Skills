from typing import Dict, List

from dash import html
import dash_bootstrap_components as dbc

from layouts.table_layouts import (
    get_available_models,
    get_filter_layout,
    get_selector_layout,
    get_sorting_layout,
)
from visualization.settings.constants import DELETE, GENERAL_STATS


def get_models_options_layout() -> dbc.Accordion:
    runs_storage = get_available_models()

    return dbc.Accordion(
        dbc.AccordionItem(
            dbc.Accordion(
                [
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
                ],
                start_collapsed=True,
                always_open=True,
            ),
            title="Models options",
        ),
        start_collapsed=True,
        always_open=True,
    )


def get_utils_layout(utils: Dict) -> dbc.AccordionItem:
    return dbc.AccordionItem(
        html.Div(
            [
                dbc.InputGroup(
                    [
                        html.Pre(f"{name}: ", className="mr-2"),
                        html.Pre(value, className="mr-2"),
                    ],
                    className="mb-3",
                )
                for name, value in utils.items()
            ]
        ),
        title="Utils",
    )


def get_few_shots_layout(examples: List[Dict]) -> dbc.AccordionItem:
    return dbc.AccordionItem(
        html.Div(
            [
                dbc.Accordion(
                    dbc.AccordionItem(
                        [
                            html.Div(
                                [
                                    html.P(
                                        name,
                                        className="font-weight-bold",
                                    ),
                                    html.Pre(value),
                                ]
                            )
                            for name, value in example.items()
                        ],
                        title=f"example {id}",
                    ),
                    start_collapsed=True,
                    always_open=True,
                )
                for id, example in enumerate(examples)
            ]
        ),
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
            dbc.Container(id="filtering_container"),
            dbc.Container(id="sorting_container"),
            # dcc.Loading( # TODO spinner
            # children=
            html.Div(
                children=[],
                id="compare_models_rows",
            ),
            #     type="circle",
            #     style={'margin-top': '50px'},
            # ),
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
    return html.Div(
        [
            dbc.Button("Stats", id="set_new_stats_button"),
            dbc.Modal(
                [
                    dbc.ModalHeader(
                        [
                            dbc.ModalTitle("set up your stats"),
                            dbc.Checklist(
                                id="stats_modes",
                                options=[
                                    {
                                        "label": "general stats",
                                        "value": GENERAL_STATS,
                                    },
                                    {
                                        "label": "delete mode",
                                        "value": DELETE,
                                    },
                                ],
                                switch=True,
                                inline=True,
                            ),
                        ],
                        close_button=True,
                    ),
                    dbc.ModalBody(
                        html.Div(
                            [
                                html.Pre(get_stats_text(), id="stats_text"),
                                dbc.Textarea(id="new_stats_input"),
                            ]
                        )
                    ),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Apply",
                            id="apply_new_stats",
                            className="ms-auto",
                            n_clicks=0,
                        )
                    ),
                ],
                size="lg",
                id="new_stats",
                centered=True,
                is_open=False,
            ),
        ],
        style={'display': 'inline-block'},
    )
