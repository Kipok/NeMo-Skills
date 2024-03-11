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

from typing import Dict, List

import dash_bootstrap_components as dbc
from dash import html
from settings.constants import ONE_TEST_MODE, PARAMS_FOR_WHOLE_DATASET_ONLY
from utils.common import get_test_data

from visualization.utils.strategies.base_strategy import ModeStrategies


class OneTestModeStrategy(ModeStrategies):
    mode = ONE_TEST_MODE

    def __init__(self):
        super().__init__()

    def get_utils_input_layout(self) -> List[dbc.AccordionItem]:
        inference_condition = (
            lambda name, value: isinstance(value, (str, int, float))
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
            )[0].items()
        )

    def run(self, utils: Dict, params: Dict) -> html.Div:
        params['prompts'] = [self.get_prompt(utils, params['question'])]
        return super().run(utils, params)
