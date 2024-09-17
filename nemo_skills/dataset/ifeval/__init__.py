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
from nemo_skills.evaluation.graders import if_grader
from nemo_skills.evaluation.metrics import IFMetrics

# settings that define how evaluation should be done by default (all can be changed from cmdline)
PROMPT_CONFIG = 'generic/default'
DATASET_GROUP = 'chat'
METRICS_CLASS = IFMetrics
GRADER_CLASS = if_grader
DEFAULT_EVAL_ARGS = "++eval_type=ifeval"
DEFAULT_GENERATION_ARGS = "++generation_key=response"
