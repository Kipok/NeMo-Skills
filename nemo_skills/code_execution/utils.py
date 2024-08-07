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

import re
from typing import Dict, Tuple

CODE_SEPARATORS = (
    '<|python_tag|>',
    '<|eom_id|>',
)  # used to execute code within these tags
CODE_OUTPUT_SEPARATORS = (
    '<|start_header_id|>ipython<|end_header_id|>',
    # we assume that assistant always has more to say after executing the code!
    '<|eot_id|><|start_header_id|>assistant<|end_header_id|>',
)  # used to extract the code output


def format_code_output(execution_dict: Dict[str, str]):
    """Formatting code output to be displayed as an llm expects it.

    Adapted from https://github.com/meta-llama/llama-agentic-system/blob/ce0c90271bdfd4205cd4f086cad9005b48797333/llama_agentic_system/tools/builtin.py#L310

    """
    pieces = [execution_dict["process_status"]]
    for out_type in ["stdout", "stderr"]:
        res_out = execution_dict[out_type]
        if res_out != "":
            pieces.extend([f"[{out_type}]", res_out, f"[/{out_type}]"])

    return "\n".join(pieces)


def _extract_between_separators(generation, separators: Tuple[str, str], extract_all=False):
    """Extracting all text between last occurrence of separators[0] and [1].

    If extract_all is True, returning a list with all occurrences of text between separators.
    """
    if extract_all:
        pattern = f'{separators[0]}(.*?){separators[1]}'
        return re.findall(pattern, generation, re.DOTALL)
    return generation.split(separators[0])[-1].split(separators[1])[0]


def extract_code_to_execute(generation, extract_all=False):
    return _extract_between_separators(generation, CODE_SEPARATORS, extract_all)


def extract_code_output(generation, extract_all=False):
    return _extract_between_separators(generation, CODE_OUTPUT_SEPARATORS, extract_all)
