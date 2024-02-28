from gc import disable
from typing import Dict, List, Tuple, Union

from dash import html, dcc
import dash_bootstrap_components as dbc


from settings.constants import (
    COMPLETE_MODE,
    ANSWER_FIELD,
    ONE_TEST_MODE,
    PARAMS_FOR_WHOLE_DATASET_ONLY,
    QUESTION_FIELD,
    WHOLE_DATASET_MODE,
)

from nemo_skills.inference.prompt.utils import (
    context_templates,
)
from utils.common import (
    examples,
    get_estimated_height,
    get_test_data,
)

from settings.config import ConfigHolder
from layouts.table_layouts import get_single_prompt_output_layout


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


def get_utils_input_layout(mode: bool = True) -> dbc.AccordionItem:
    return [
        dbc.AccordionItem(
            (
                (
                    [
                        dbc.Checklist(
                            id="range_random_seed_mode",
                            options=[
                                {
                                    "label": "use random seed range",
                                    "value": 1,
                                    "disabled": mode != WHOLE_DATASET_MODE,
                                }
                            ],
                            switch=True,
                        )
                    ]
                )
                + [
                    html.Div(
                        (
                            [
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText(name),
                                        dbc.Input(
                                            value=(
                                                value
                                                if value == ""
                                                or str(value).strip() != ""
                                                else repr(value)[1:-1]
                                            ),
                                            id=name,
                                            **validation_parameters(
                                                name, value
                                            ),
                                            debounce=True,
                                        ),
                                    ],
                                    className="mb-3",
                                )
                                for name, value in list(
                                    ConfigHolder.get_config()[
                                        "inference"
                                    ].items()
                                )
                                + (
                                    list(ConfigHolder.get_config().items())
                                    if mode != COMPLETE_MODE
                                    else []
                                )
                                if not isinstance(value, dict)
                                and (
                                    mode == WHOLE_DATASET_MODE
                                    or name not in PARAMS_FOR_WHOLE_DATASET_ONLY
                                )
                            ]
                            + (
                                [
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText(param_name),
                                            dbc.Textarea(
                                                value=(
                                                    value
                                                    if value == ""
                                                    or str(value).strip() != ""
                                                    else repr(value)[1:-1]
                                                ),
                                                id=param_name,
                                                style={
                                                    'width': '100%',
                                                    **(
                                                        {
                                                            'height': get_estimated_height(
                                                                value
                                                            )
                                                        }
                                                        # if value != "" and str(value).strip() != ""
                                                        # else {}
                                                    ),
                                                },  # TODO: add js to update height
                                            ),
                                        ],
                                        className="mb-3",
                                    )
                                    for param_name, value in ConfigHolder.get_config()[
                                        "prompt"
                                    ].items()
                                ]
                                if mode != COMPLETE_MODE
                                else []
                            )
                            + (
                                [
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText(
                                                "context_templates"
                                            ),
                                            dbc.Textarea(
                                                value=context_templates[
                                                    ConfigHolder.get_config()[
                                                        "prompt"
                                                    ]["context_type"]
                                                ],
                                                id="context_templates",
                                                style={
                                                    'width': '100%',
                                                    'height': get_estimated_height(
                                                        context_templates[
                                                            ConfigHolder.get_config()[
                                                                "prompt"
                                                            ][
                                                                "context_type"
                                                            ]
                                                        ],
                                                    ),
                                                },  # TODO: add js to update height
                                            ),
                                        ],
                                        className="mb-3",
                                    )
                                ]
                                if mode != COMPLETE_MODE
                                else []
                            )
                        ),
                        id="utils_group",
                    ),
                ]
            ),
            title="Utils",
        )
    ]


def get_few_shots_div_layout(examples_type: str):
    return html.Div(
        [
            html.Div(
                [
                    dbc.Pagination(
                        id="few_shots_pagination",
                        max_value=(
                            len(
                                examples.get(
                                    examples_type,
                                    [],
                                )
                            )
                        ),
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
                    dbc.Checklist(
                        id={
                            "type": "view_mode",
                            "id": "few_shots_input",
                        },
                        options=[
                            {
                                "label": "view mode",
                                "value": 1,
                            }
                        ],
                        switch=True,
                    ),
                ]
            ),
            dbc.Container(id="few_shots_pagination_content"),
        ],
        id="few_shots_div",
    )


def get_few_shots_input_layout(
    examples_type=None, mode=ONE_TEST_MODE
) -> dbc.AccordionItem:
    if not examples_type:
        examples_type = ConfigHolder.get_config()["prompt"]["examples_type"]
    return (
        [
            dbc.AccordionItem(
                get_few_shots_div_layout(examples_type),
                title="Few shots",
                id="few_shots_group",
            )
        ]
        if mode != COMPLETE_MODE
        else []
    )


def get_few_shots_by_id_layout(
    page: int, examples_type: str, view_mode: bool
) -> Tuple[html.Div]:
    return (
        (
            html.Div(
                [
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText(key),
                            get_text_area_layout(key, value, view_mode),
                        ],
                        className="mb-3",
                    )
                    for key, value in (
                        examples.get(
                            examples_type,
                            [{}],
                        )[page - 1].items()
                    )
                ],
            ),
        )
        if page
        and len(
            examples.get(
                examples_type,
                [{}],
            )
        )
        > page - 1
        else html.Div()
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
            "type": "query_input",
            "id": key,
        },
        style={
            'width': '100%',
            'height': get_estimated_height(value),
            'border': "1px solid #dee2e6",
        },  # TODO: add js to update height
    )


def get_query_input_children_layout(
    index: int, dataset: str, split_name: str, view_mode: bool = False
) -> List[dbc.InputGroup]:
    return [
        dbc.InputGroup(
            [
                dbc.InputGroupText(key),
                get_text_area_layout(key, value, view_mode),
            ],
            className="mb-3",
        )
        for key, value in get_test_data(
            index,
            dataset,
            split_name,
        )[0].items()
    ]


def get_query_input_layout(mode=ONE_TEST_MODE) -> dbc.AccordionItem:
    return (
        [
            dbc.AccordionItem(
                html.Div(
                    (
                        [
                            dbc.Checklist(
                                id={
                                    "type": "view_mode",
                                    "id": "query_input",
                                },
                                options=[
                                    {
                                        "label": "view mode",
                                        "value": 1,
                                    }
                                ],
                                switch=True,
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("Index of test"),
                                    dbc.Input(
                                        value=1,
                                        id="query_search_input",
                                        type="number",
                                        size="sm",
                                        disabled=mode != ONE_TEST_MODE,
                                    ),
                                    dbc.Button(
                                        "Search",
                                        id="query_search_button",
                                        outline=True,
                                        size="sm",
                                        color="primary",
                                        className="me-1",
                                        disabled=mode != ONE_TEST_MODE,
                                    ),
                                ],
                                className="mb-3",
                            ),
                        ]
                        if mode == ONE_TEST_MODE
                        else []
                    )
                    + [
                        html.Div(
                            (
                                get_query_input_children_layout(
                                    0,
                                    ConfigHolder.get_config()['dataset'],
                                    ConfigHolder.get_config()['split_name'],
                                )
                                if mode == ONE_TEST_MODE
                                else [
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText(key),
                                            get_text_area_layout(key, ""),
                                        ],
                                        className="mb-3",
                                    )
                                    for key in [
                                        QUESTION_FIELD,
                                        ANSWER_FIELD,
                                    ]
                                ]
                            ),
                            id="query_input_children",
                        ),
                    ]
                ),
                title="Input",
                id="query_input_content",
            )
        ]
        if mode != WHOLE_DATASET_MODE
        else []
    )


def get_query_params_layout(
    mode: str = ONE_TEST_MODE,
) -> List[dbc.AccordionItem]:
    return (
        get_utils_input_layout(mode)
        + get_few_shots_input_layout(mode=mode)
        + get_query_input_layout(mode)
    )


def get_run_mode_layout() -> html.Div:
    return html.Div(
        [
            dbc.RadioItems(
                id="run_mode_options",
                className="btn-group",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-primary",
                labelCheckedClassName="active",
                options=[
                    {"label": "Complete", "value": COMPLETE_MODE},
                    {"label": "Run one test", "value": ONE_TEST_MODE},
                    {"label": "Run whole dataset", "value": WHOLE_DATASET_MODE},
                ],
                value=ONE_TEST_MODE,
            ),
        ],
        className="radio-group",
    )


def get_run_test_layout() -> html.Div:
    return html.Div(
        [
            get_run_mode_layout(),
            dbc.Accordion(
                get_query_params_layout(),
                start_collapsed=True,
                always_open=True,
                id="prompt_params_input",
            ),
            dbc.Button(
                "preview",
                id="preview_button",
                outline=True,
                color="primary",
                className="me-1 mb-2",
            ),
            dbc.Button(
                "run",
                id="run_button",
                outline=True,
                color="primary",
                className="me-1 mb-2",
            ),
            dcc.Loading(
                children=(dbc.Container(id="results_content")),
                type='circle',
                style={'margin-top': '50px'},
            ),
        ]
    )
