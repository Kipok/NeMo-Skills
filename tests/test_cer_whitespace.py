# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

from nemo_skills.evaluation.evaluator.cer_whitespace import normalize_cer_whitespace


@pytest.mark.parametrize(
    "text,expected",
    [
        ("今 天 use AI", "今天use AI"),
        ("hello world", "hello world"),
        ("hello\t\n世 界", "hello世界"),
        ("〇 𰀀 test words", "〇𰀀test words"),
        (" नमस्ते दुनिया ", " नमस्ते दुनिया "),
        ("", ""),
    ],
)
def test_han_adjacent_keeps_latin_and_indic_boundaries(text, expected):
    assert normalize_cer_whitespace(text, "han_adjacent") == expected


def test_other_modes_and_invalid_mode():
    assert normalize_cer_whitespace("今 天 a b", "keep") == "今 天 a b"
    assert normalize_cer_whitespace("今 天 a b", "all") == "今天ab"
    with pytest.raises(ValueError):
        normalize_cer_whitespace("", "unknown")


def test_evaluate_cer_han_adjacent_standard_normalization():
    from nemo_skills.evaluation.evaluator.audio import evaluate_cer

    result = evaluate_cer("今 天 use AI", "今天 use AI", normalization_mode="standard", whitespace_mode="han_adjacent")
    assert result["cer"] == 0
    assert evaluate_cer("hello world", "helloworld", whitespace_mode="han_adjacent")["cer"] > 0
    assert evaluate_cer("", "你好", whitespace_mode="han_adjacent")["cer"] is None
    with pytest.raises(ValueError):
        evaluate_cer("你好", "你好", normalize_compound=True, whitespace_mode="han_adjacent")
