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

from visualization.settings.constants import ANSWER_FIELD, ONE_SAMPLE_MODE, QUESTION_FIELD
from visualization.utils.common import get_utils_from_config
from visualization.utils.strategies.base_strategy import ModeStrategies


class ChatModeStrategy(ModeStrategies):
    mode = ONE_SAMPLE_MODE

    def __init__(self):
        super().__init__()

    def get_utils_input_layout(self) -> List[dbc.AccordionItem]:
        return super().get_utils_input_layout(
            lambda key, value: key in self.config['inference'].keys(), True
        )

    def get_few_shots_input_layout(self) -> List[dbc.AccordionItem]:
        return []

    def get_query_input_layout(self, dataset) -> List[dbc.AccordionItem]:
        return super().get_query_input_layout(
            {
                QUESTION_FIELD: "",
                ANSWER_FIELD: "",
            },
            False,
        )

    def run(self, utils: Dict, params: Dict):
        utils_updated = {
            **utils,
            **{
                key: ""
                for key, value in get_utils_from_config(self.config).items()
                if isinstance(value, str)
            },
        }
        params['prompts'] = [self.get_prompt(utils, params)]
        return super().run(utils_updated, params)

    def get_prompt(self, utils: Dict, params: Dict) -> str:
        utils_updated = {
            **utils,
            **{
                key: ""
                for key, value in get_utils_from_config(self.config).items()
                if isinstance(value, str)
            },
        }
        utils_updated['user'] = '{question}'
        utils_updated['prompt_template'] = '{user}'
        return super().get_prompt(
            utils_updated,
            params,
        )
