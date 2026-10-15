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

from nemo_skills.evaluation.evaluator.audio import MultilingualTextNormalizer


@pytest.mark.parametrize(
    "lang,digits,spoken",
    [
        ("es", "22", "veintidós"),
        ("pt", "3", "três"),
        ("it", "23", "ventitré"),
        ("de", "12", "zwölf"),
        ("fr", "80", "quatre-vingts"),
    ],
)
@pytest.mark.parametrize("remove_diacritics", [True, False])
def test_number_expansion_matches_cleaned_spoken_words(lang, digits, spoken, remove_diacritics):
    normalizer = MultilingualTextNormalizer(remove_diacritics=remove_diacritics)
    expanded = normalizer(digits, lang=lang)
    assert expanded == normalizer(spoken, lang=lang)
    assert normalizer(expanded, lang=lang) == expanded


def test_no_language_and_unsupported_language_retain_digits():
    normalizer = MultilingualTextNormalizer()
    assert normalizer("22") == "22"
    assert normalizer("22", lang="unsupported") == "22"
    assert normalizer("10 000", lang="en") == "ten thousand"
