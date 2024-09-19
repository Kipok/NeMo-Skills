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
from nemo_skills.evaluation.metrics import MathMetrics

# settings that define how evaluation should be done by default (all can be changed from cmdline)
PROMPT_CONFIG = 'generic/multichoice'
DATASET_GROUP = 'multichoice'
METRICS_CLASS = MathMetrics  # TODO: update this?
DEFAULT_EVAL_ARGS = (
    "++eval_type=math "
    "++eval_config.extract_from_boxed=False "
    "'++eval_config.extract_regex=\"The best answer is (.+)$\"'"
)
DEFAULT_GENERATION_ARGS = ""
