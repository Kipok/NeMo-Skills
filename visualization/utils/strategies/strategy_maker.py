from typing import Optional

from settings.constants import COMPLETE_MODE, ONE_TEST_MODE, WHOLE_DATASET_MODE
from utils.strategies.complete_mode import CompleteModeStrategy
from utils.strategies.one_test_mode import OneTestModeStrategy
from utils.strategies.whole_dataset_mode import WholeDatasetModeStrategy

from visualization.utils.strategies.base_strategy import ModeStrategies


class RunPromptStrategyMaker:
    strategies = {
        ONE_TEST_MODE: OneTestModeStrategy,
        WHOLE_DATASET_MODE: WholeDatasetModeStrategy,
        COMPLETE_MODE: CompleteModeStrategy,
    }

    def __init__(self, mode: Optional[str] = None):
        self.mode = mode

    def get_strategy(self) -> ModeStrategies:
        return self.strategies.get(self.mode, ModeStrategies)()
