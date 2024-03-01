from typing import List, Tuple

import dash_bootstrap_components as dbc
from dash import dcc, html
from layouts.base_layouts import get_text_area_layout
from settings.constants import COMPLETE_MODE, ONE_TEST_MODE, WHOLE_DATASET_MODE
from utils.common import examples
from utils.strategies.strategy_maker import RunPromptStrategyMaker


def get_few_shots_by_id_layout(page: int, examples_type: str, view_mode: bool) -> Tuple[html.Div]:
    examples_list = examples.get(
        examples_type,
        [{}],
    )
    if not page or len(examples_list) <= page - 1:
        return html.Div()
    return (
        html.Div(
            [
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(key),
                        get_text_area_layout(key, value, view_mode),
                    ],
                    className="mb-3",
                )
                for key, value in (examples_list[page - 1].items())
            ],
        ),
    )


def get_query_params_layout(
    mode: str = ONE_TEST_MODE,
) -> List[dbc.AccordionItem]:
    strategy = RunPromptStrategyMaker(mode).get_strategy()
    return (
        strategy.get_utils_input_layout() + strategy.get_few_shots_input_layout() + strategy.get_query_input_layout()
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
