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

import os

import pytest

from nemo_skills.code_execution import extract_code_output, extract_code_to_execute
from nemo_skills.code_execution.sandbox import Sandbox, get_sandbox
from nemo_skills.inference.prompt.utils import Prompt, get_prompt_config


def test_rephrasing_prompt():
    prompt = Prompt(
        config=get_prompt_config('rephrasing'),
        input_dict={'question': '2 + 2 = ?'},
        example_dicts=[
            {'question': '1 + 1 = ?', 'rephrased_question': "3 + 3 = ?"},
            {'question': '5 + 5 = ?', 'rephrased_question': "7 + 7 = ?"},
        ],
    )
    expected_prompt = """You are an AI assistant that excels at rephrasing questions. Follow the given examples.

Question:
1 + 1 = ?

Rephrase the above question:
3 + 3 = ?





Question:
5 + 5 = ?

Rephrase the above question:
7 + 7 = ?





Question:
2 + 2 = ?

Rephrase the above question:
"""
    assert str(prompt) == expected_prompt


def test_augmentation_prompt():
    prompt = Prompt(
        config=get_prompt_config('augmentation'),
        input_dict={'question': '2 + 2 = ?'},
        example_dicts=[
            {'question': '1 + 1 = ?', 'augmented_question': "3 + 3 = ?"},
            {'question': '5 + 5 = ?', 'augmented_question': "7 + 7 = ?"},
        ],
    )
    expected_prompt = """You are an AI assistant that excels at creating similar questions. Follow the given examples.

Question:
1 + 1 = ?

Write another question similar to this one:
3 + 3 = ?





Question:
5 + 5 = ?

Write another question similar to this one:
7 + 7 = ?





Question:
2 + 2 = ?

Write another question similar to this one:
"""
    assert str(prompt) == expected_prompt


def test_llama3_instruct_prompt():
    prompt = Prompt(
        config=get_prompt_config('llama3_instruct'),
        input_dict={'question': '2 + 2 = ?'},
        example_dicts=[
            {'question': '1 + 1 = ?', 'generated_solution': "That's easy: 2!"},
            {'question': '5 + 5 = ?', 'generated_solution': "That's easy: 10!"},
        ],
    )
    expected_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Meta AI, a sophisticated and energetic AI Assistant. You excel at solving mathematical problems.

You will look at the examples provided by the user and try to follow the solution format as much as possible.<|eot_id|><|start_header_id|>user<|end_header_id|>

Here are some examples of questions and solutions followed by a new question that you need to solve.
Make sure to put the answer (and only answer) inside \\boxed{}.

Question:
1 + 1 = ?

My solution:
That's easy: 2!





Question:
5 + 5 = ?

My solution:
That's easy: 10!





Question:
2 + 2 = ?

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    assert str(prompt) == expected_prompt


def test_llama3_base_prompt():
    prompt = Prompt(
        config=get_prompt_config('llama3_base'),
        input_dict={'question': '2 + 2 = ?'},
        example_dicts=[
            {'question': '1 + 1 = ?', 'generated_solution': "That's easy: 2!"},
            {'question': '5 + 5 = ?', 'generated_solution': "That's easy: 10!"},
        ],
    )
    expected_prompt = """<|begin_of_text|>Here are some examples of questions and solutions followed by a new question that you need to solve.
Make sure to put the answer (and only answer) inside \\boxed{}.

Question:
1 + 1 = ?

My solution:
That's easy: 2!





Question:
5 + 5 = ?

My solution:
That's easy: 10!





Question:
2 + 2 = ?

My solution:
"""
    assert str(prompt) == expected_prompt
