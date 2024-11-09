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

import os
from typing import Dict, List

import dash_bootstrap_components as dbc
from dash import dcc, html
from flask import current_app
from layouts.base_layouts import get_selector_layout, get_switch_layout
from layouts.table_layouts import get_change_label_layout, get_filter_layout, get_sorting_layout
from settings.constants import CHOOSE_MODEL, CUSTOM, DELETE, GENERAL_STATS, INLINE_STATS
from utils.common import get_available_models, get_custom_stats, get_general_custom_stats, get_stats_raw
from settings.ui_config import UIConfig


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
            style=UIConfig.STYLES["card"]
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
            title="Generation parameters",
            style=UIConfig.STYLES["card"]
        ),
        start_collapsed=True,
        always_open=True,
    )

def get_utils_layout(utils: Dict) -> dbc.AccordionItem:
    input_groups = [
        dbc.InputGroup(
            [
                html.Pre(
                    f"{name}: ",
                    className="mr-2",
                    style={"color": UIConfig.THEME["text_secondary"]}
                ),
                html.Pre(
                    (value if value == "" or str(value).strip() != "" else repr(value)[1:-1]),
                    className="mr-2",
                    style={
                        "overflow-x": "scroll",
                        "color": UIConfig.THEME["text"],
                        "background-color": UIConfig.THEME["surface"],
                        "padding": "0.5rem",
                        "border-radius": "4px"
                    },
                ),
            ],
            className="mb-3",
        )
        for name, value in utils.items()
    ]
    return dbc.AccordionItem(
        html.Div(input_groups),
        title="Utils",
        style=UIConfig.STYLES["card"]
    )

def get_few_shots_layout(examples: List[Dict]) -> dbc.AccordionItem:
    example_layout = lambda example: [
        html.Div(
            [
                dcc.Markdown(
                    f'**{name}**',
                    style={"color": UIConfig.THEME["text"]}
                ),
                html.Pre(
                    value,
                    style={
                        "background-color": UIConfig.THEME["surface"],
                        "padding": "1rem",
                        "border-radius": "6px",
                        "color": UIConfig.THEME["text"],
                        "border": f"1px solid {UIConfig.THEME['border']}"
                    }
                ),
            ]
        )
        for name, value in example.items()
    ]
    examples_layout = [
        dbc.Accordion(
            dbc.AccordionItem(
                example_layout(example),
                title=f"example {id}",
                style=UIConfig.STYLES["card"]
            ),
            start_collapsed=True,
            always_open=True,
        )
        for id, example in enumerate(examples)
    ]
    return dbc.AccordionItem(
        html.Div(examples_layout),
        title="Few shots",
        style=UIConfig.STYLES["card"]
    )

def get_button_style(primary=False):
    return {
        "background-color": UIConfig.THEME["primary"] if primary else "transparent",
        "color": "white" if primary else UIConfig.THEME["text"],
        "border": f"1px solid {UIConfig.THEME['primary']}" if not primary else "none",
        "border-radius": "6px",
        "padding": "0.5rem 1rem",
        "font-size": "0.875rem",
        "transition": "all 0.2s ease-in-out",
        "hover": {
            "background-color": UIConfig.THEME["primary_hover"] if primary else UIConfig.THEME["surface"],
            "transform": "translateY(-1px)"
        }
    }

def get_modal_style():
    return {
        "background-color": UIConfig.THEME["surface"],
        "color": UIConfig.THEME["text"],
        "border-radius": "8px",
        "border": f"1px solid {UIConfig.THEME['border']}",
    }

# Update other modal and button components
def get_update_dataset_modal_layout() -> html.Div:
    text = (
        "Write an expression to modify the data\n\n"
        "For example: {**data, 'generation': data['generation'].strip()}\n\n"
        "The function has to return a new dict"
    )
    # header = dbc.ModalHeader(
    #     dbc.ModalTitle("Update Dataset", style={"color": UIConfig.THEME["text"]}),
    #     close_button=True,
    #     style={"border-bottom": f"1px solid {UIConfig.THEME['border']}"}
    # )
    # body = dbc.ModalBody(
    #     html.Div(
    #         [
    #             html.Pre(
    #                 text,
    #                 style={"color": UIConfig.THEME["text_secondary"]}
    #             ),
    #             dbc.Textarea(
    #                 id="update_dataset_input",
    #                 style={
    #                     "background-color": UIConfig.THEME["background"],
    #                     "color": UIConfig.THEME["text"],
    #                     "border": f"1px solid {UIConfig.THEME['border']}"
    #                 }
    #             ),
    #         ],
    #     )
    # )
    # footer = dbc.ModalFooter(
    #     dbc.Button(
    #         "Apply",
    #         id="apply_update_dataset_button",
    #         className="ms-auto",
    #         n_clicks=0,
    #         style=get_button_style(primary=True)
    #     )
    # )
    # return html.Div(
    #     [
    #         dbc.Button(
    #             "Update dataset",
    #             id="update_dataset_button",
    #             style=get_button_style(),
    #             className='button-class',
    #         ),
    #         dbc.Modal(
    #             [header, body, footer],
    #             size="lg",
    #             id="update_dataset_modal",
    #             centered=True,
    #             is_open=False,
    #             style=get_modal_style()
    #         ),
    #     ],
    #     style={'display': 'inline-block'},
    # )
    # def get_update_dataset_modal(text: str) -> html.Div:
    header = dbc.ModalHeader(
        dbc.ModalTitle(
            "Update Dataset", 
            style={"color": UIConfig.THEME["text"]}
        ),
        close_button=True,
        style={
            "border-bottom": f"1px solid {UIConfig.THEME['border']}",
            "background-color": UIConfig.THEME["surface"]
        }
    )

    body = dbc.ModalBody(
        html.Div(
            [
                html.Pre(
                    text,
                    style={
                        "color": UIConfig.THEME["text_secondary"],
                        "background-color": UIConfig.THEME["background"],
                        "padding": "1rem",
                        "border-radius": "6px",
                        "border": f"1px solid {UIConfig.THEME['border']}",
                        "font-family": "monospace",
                        "margin-bottom": "1rem",
                        "white-space": "pre-wrap",
                    }
                ),
                dbc.Textarea(
                    id="update_dataset_input",
                    style={
                        "background-color": UIConfig.THEME["background"],
                        "color": UIConfig.THEME["text"],
                        "border": f"1px solid {UIConfig.THEME['border']}",
                        "border-radius": "6px",
                        "padding": "0.75rem",
                        "font-family": "monospace",
                        "min-height": "150px",
                        "resize": "vertical",
                        "box-shadow": f"0 1px 2px {UIConfig.THEME['background']}"
                    }
                ),
            ],
        )
    )

    footer = dbc.ModalFooter(
        dbc.Button(
            "Apply",
            id="apply_update_dataset_button",
            className="ms-auto",
            n_clicks=0,
            style=UIConfig.STYLES["button_primary"]
        ),
        style={
            "background-color": UIConfig.THEME["surface"],
            "border-top": f"1px solid {UIConfig.THEME['border']}"
        }
    )

    return html.Div(
        [
            dbc.Button(
                "Update dataset",
                id="update_dataset_button",
                # style=UIConfig.STYLES["button_secondary"],
                className='me-1',
            ),
            dbc.Modal(
                [header, body, footer],
                size="lg",
                id="update_dataset_modal",
                centered=True,
                is_open=False,
                style={
                    **UIConfig.STYLES["modal"],
                    "background-color": UIConfig.THEME["surface"],
                    "max-width": "800px",
                    "box-shadow": "0 8px 24px rgba(0,0,0,0.2)"
                }
            ),
        ],
        style={'display': 'inline-block'},
    )

def get_save_dataset_layout() -> html.Div:
    return html.Div(
        [
            dbc.Button(
                "Save dataset", 
                id="save_dataset", 
                # style=UIConfig.STYLES["button_secondary"],
                className='me-1'
            ),
            dbc.Modal(
                [
                    dbc.ModalBody(
                        [
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText(
                                        'save_path',
                                        style={
                                            "background-color": UIConfig.THEME["surface"],
                                            "color": UIConfig.THEME["text_secondary"],
                                            "border": f"1px solid {UIConfig.THEME['border']}"
                                        }
                                    ),
                                    dbc.Input(
                                        value=os.path.join(
                                            current_app.config['nemo_inspector']['inspector_params'][
                                                'save_generations_path'
                                            ],
                                            'default_name',
                                        ),
                                        id='save_path',
                                        type='text',
                                        style=UIConfig.STYLES["input"]
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.Container(
                                id="error_message",
                                style={"color": UIConfig.THEME["error"]}
                            ),
                        ]
                    ),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Save",
                            id="save_dataset_button",
                            className="ms-auto",
                            n_clicks=0,
                            style=UIConfig.STYLES["button_primary"]
                        )
                    ),
                ],
                id="save_dataset_modal",
                is_open=False,
                style={
                    'text-align': 'center',
                    "margin-top": "10px",
                    "margin-bottom": "10px",
                    **UIConfig.STYLES["modal"]
                },
            ),
        ],
    )

def get_compare_test_layout() -> html.Div:
    return html.Div(
        [
            # get_models_options_layout(),
            dbc.InputGroup(
                [
                    get_sorting_layout(),
                    get_filter_layout(),
                    get_add_stats_layout(),
                    get_change_label_layout(apply_for_all_files=False),
                    get_update_dataset_modal_layout(),
                    get_save_dataset_layout(),
                    dbc.Button(
                        "+",
                        id="add_model",
                        # style={
                        #     **UIConfig.STYLES["button_secondary"],
                        #     'margin-left': '1px'
                        # },
                        className="me-1",
                    ),
                    get_selector_layout(
                        get_available_models().keys(),
                        'base_model_answers_selector',
                        value=CHOOSE_MODEL,
                    ),
                ],
                style={"gap": "0.5rem"}
            ),
            html.Pre(
                id="filtering_container",
                style={
                    "color": UIConfig.THEME["text_secondary"],
                    "background-color": UIConfig.THEME["surface"],
                    "padding": "1rem",
                    "border-radius": "6px",
                    "margin-top": "1rem"
                }
            ),
            html.Pre(
                id="sorting_container",
                style={
                    "color": UIConfig.THEME["text_secondary"],
                    "background-color": UIConfig.THEME["surface"],
                    "padding": "1rem",
                    "border-radius": "6px"
                }
            ),
            dcc.Loading(
                children=dbc.Container(
                    id="loading_container",
                    style={'display': 'none'},
                    children=""
                ),
                type='circle',
                style={'margin-top': '50px'},
                color=UIConfig.THEME["primary"]
            ),
            html.Div(
                children=[],
                id="compare_models_rows",
            ),
        ],
        style={"color": UIConfig.THEME["text"]}
    )


def get_stats_text(general_stats: bool = False, delete: bool = False):
    if delete:
        return "Choose the name of the statistic you want to delete"
    else:
        if general_stats:
            return (
                "Creating General Custom Statistics:\n\n"
                "To introduce new general custom statistics:\n"
                "1. Create a dictionary where keys are the names of your custom stats.\n"
                "2. Assign functions as values. These functions should accept arrays where first dimension\n"
                "is a question index and second is a file number (both sorted and filtered).\n\n"
                "Example:\n\n"
                "Define a custom function to integrate into your stats:\n\n"
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
                "Creating Custom Statistics:\n\n"
                "To introduce new custom statistics:\n"
                "1. Create a dictionary where keys are the names of your custom stats.\n"
                "2. Assign functions as values. These functions should accept arrays containing data\n"
                "from all relevant files.\n\n"
                "Note: Do not use names that already exist in the current stats or JSON fields\n"
                "to avoid conflicts.\n\n"
                "Example:\n\n"
                "Define a custom function to integrate into your stats:\n\n"
                "def unique_error_counter(datas):\n"
                "    unique_errors = set()\n"
                "    for data in datas:\n"
                "        unique_errors.add(data.get('error_message'))\n"
                "    return len(unique_errors)\n\n"
                "{'unique_error_count': unique_error_counter}"
            )
        

def get_stats_input(modes: List[str] = []) -> List:
    body = []
    if DELETE in modes:
        delete_options = list(
            get_general_custom_stats().keys() if GENERAL_STATS in modes else get_custom_stats().keys()
        )
        body += [
            get_selector_layout(
                delete_options,
                "stats_input",
                delete_options[0] if delete_options else "",
            )
        ]
    else:
        mode = GENERAL_STATS if GENERAL_STATS in modes else INLINE_STATS
        extractor_options = list(get_stats_raw()[mode].keys())
        body += [
            get_selector_layout(extractor_options, "stats_extractor", CUSTOM),
            dbc.Textarea(
                id="stats_input",
                style={
                    **UIConfig.STYLES["input"],
                    "min-height": "200px",
                    "font-family": "monospace"
                }
            )
        ]
    return [
        html.Pre(
            get_stats_text(GENERAL_STATS in modes, DELETE in modes),
            id="stats_text",
            style={
                "color": UIConfig.THEME["text"],
                "background-color": UIConfig.THEME["surface"],
                "padding": "1rem",
                "border-radius": "6px",
                "font-family": "monospace"
            }
        ),
    ] + body

def get_add_stats_layout() -> html.Div:
    modal_header = dbc.ModalHeader(
        [
            dbc.ModalTitle(
                "Set Up Your Stats",
                style={"color": UIConfig.THEME["text"]}
            ),
            get_switch_layout(
                id="stats_modes",
                labels=["general stats", "delete mode"],
                values=[GENERAL_STATS, DELETE],
                additional_params={
                    "inline": True,
                    "style": {
                        "margin-left": "10px",
                        "color": UIConfig.THEME["text"]
                    }
                },
            ),
        ],
        close_button=True,
        style={"border-bottom": f"1px solid {UIConfig.THEME['border']}"}
    )
    modal_body = dbc.ModalBody(
        html.Div(
            get_stats_input(),
            id="stats_input_container",
        )
    )
    modal_footer = dbc.ModalFooter(
        dbc.Button(
            "Apply",
            id="apply_new_stats",
            className="ms-auto",
            n_clicks=0,
            style=UIConfig.STYLES["button_primary"]
        )
    )
    return html.Div(
        [
            dbc.Button(
                "Stats",
                id="set_new_stats_button",
                color="primary",
                className="me-1",
                # style=UIConfig.STYLES["button_secondary"],
                # className='button-class',
            ),
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
                style=UIConfig.STYLES["modal"]
            ),
        ],
        style={'display': 'inline-block'},
    )