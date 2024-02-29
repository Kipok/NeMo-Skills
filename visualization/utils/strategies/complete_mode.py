from typing import Dict, List

from settings.constants import (
    ANSWER_FIELD,
    ONE_TEST_MODE,
    QUESTION_FIELD,
)
import dash_bootstrap_components as dbc

from visualization.utils.strategies.base_strategy import ModeStrategies


class CompleteModeStrategy(ModeStrategies):
    mode = ONE_TEST_MODE

    def __init__(self):
        super().__init__()

    def get_utils_input_layout(self) -> List[dbc.AccordionItem]:
        inference_condition = lambda name, value: not isinstance(value, dict)

        return super().get_utils_input_layout(
            inference_condition, lambda name, value: False, True
        )

    def get_few_shots_input_layout(self) -> List[dbc.AccordionItem]:
        return []

    def get_query_input_layout(self) -> List[dbc.AccordionItem]:
        return super().get_query_input_layout(
            [
                (QUESTION_FIELD, ""),
                (ANSWER_FIELD, ""),
            ],
            False,
        )

    def run(self, utils: Dict, params: Dict):
        utils['delimiter'] = "\n\n\n\n\n\n"
        params['prompts'] = [self.get_prompt(utils, params['question'])]
        return super().run(utils, params)

    def get_prompt(self, utils: Dict, question: str) -> str:
        return str(question)
