from typing import Dict, List

from settings.constants import (
    ONE_TEST_MODE,
    PARAMS_FOR_WHOLE_DATASET_ONLY,
)
import dash_bootstrap_components as dbc
from dash import html

from utils.common import (
    get_test_data,
)
from visualization.utils.strategies.base_strategy import ModeStrategies


class OneTestModeStrategy(ModeStrategies):
    mode = ONE_TEST_MODE

    def __init__(self):
        super().__init__()

    def get_utils_input_layout(self) -> List[dbc.AccordionItem]:
        inference_condition = (
            lambda name, value: not isinstance(value, dict)
            and name not in PARAMS_FOR_WHOLE_DATASET_ONLY
        )
        return super().get_utils_input_layout(
            inference_condition,
            lambda name, value: True,
            True,
            list(self.config.items()),
        )

    def get_query_input_layout(self) -> List[dbc.AccordionItem]:
        return super().get_query_input_layout(
            get_test_data(
                0,
                self.config['dataset'],
                self.config['split_name'],
            )[0].items()
        )

    def run(self, utils: Dict, params: Dict) -> html.Div:
        params['prompts'] = [self.get_prompt(utils, params['question'])]
        return super().run(utils, params)
