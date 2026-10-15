# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""CER whitespace policies that preserve embedded Latin word boundaries."""

import re
from typing import Literal

CERWhitespaceMode = Literal["keep", "all", "han_adjacent"]

_HAN_INTERVALS = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0x2CEB0, 0x2EBEF),
    (0x2EBF0, 0x2EE5F),
    (0x2F800, 0x2FA1F),
    (0x30000, 0x3134F),
    (0x31350, 0x323AF),
)


def _is_han(character: str) -> bool:
    codepoint = ord(character)
    return codepoint == 0x3007 or any(start <= codepoint <= end for start, end in _HAN_INTERVALS)


def normalize_cer_whitespace(text: str, mode: CERWhitespaceMode) -> str:
    """Remove whitespace globally or next to Han, keeping other word boundaries."""
    if mode not in ("keep", "all", "han_adjacent"):
        raise ValueError(f"Unknown CER whitespace mode: {mode!r}")
    if mode == "keep":
        return text
    if mode == "all":
        return re.sub(r"\s+", "", text)

    def replace(match: re.Match[str]) -> str:
        left = text[match.start() - 1] if match.start() else ""
        right = text[match.end()] if match.end() < len(text) else ""
        adjacent = (left and _is_han(left)) or (right and _is_han(right))
        return "" if adjacent else match.group()

    return re.sub(r"\s+", replace, text)
