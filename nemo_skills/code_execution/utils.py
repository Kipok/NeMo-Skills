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

from nemo_skills.code_execution.sandbox import Sandbox

CODE_SEPARATORS = ('<llm-code>', '</llm-code>')  # used to execute code within these tags
CODE_OUTPUT_SEPARATORS = ('<llm-code-output>', '</llm-code-output>')  # used to extract the code output


def extract_code_to_execute(output):
    """Extracting all text between last occurrence of code_separators[0] and [1]"""
    return output.split(CODE_SEPARATORS[0])[-1].split(CODE_SEPARATORS[1])[0]


def extract_error_message(output):
    """Parsing output for any error messages and returning if found."""
    if CODE_OUTPUT_SEPARATORS[0] not in output:
        return Sandbox.NOT_EXECUTED
    code_output = output.split(CODE_OUTPUT_SEPARATORS[0])[-1].split(CODE_OUTPUT_SEPARATORS[1])[0].strip()
    for prefix in Sandbox.ERROR_PREFIXES:
        if code_output.startswith(prefix):
            return code_output
    return ""
