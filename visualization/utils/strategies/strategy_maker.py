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
