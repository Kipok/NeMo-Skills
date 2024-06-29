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

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/reasoning-machines/gsm-hard/raw/main/gsmhardv2.jsonl"

# Data Format
#
# Required:
#   - question (problem statement)
#
# Optional:
#   - expected_answer (expected answer)
#   - reference_solution (text-based solution)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--add_rounding_instructions", type=bool, default=True)
    args = parser.parse_args()

    split_name = "test"
    data_folder = Path(__file__).absolute().parent
    data_folder.mkdir(exist_ok=True)
    original_file = str(data_folder / f"original_{split_name}.jsonl")
    output_file = str(data_folder / f"{split_name}.jsonl")

    if not os.path.exists(original_file):
        urllib.request.urlretrieve(URL, original_file)

    with open(original_file, "r") as fin, open(output_file, "wt", encoding="utf-8") as fout:
        for line in fin:
            original_entry = json.loads(line)
            new_entry = dict(
                question=original_entry["input"],
                expected_answer=original_entry["target"],
            )
            # converting to int if able to for cleaner text representation
            if int(new_entry["expected_answer"]) == new_entry["expected_answer"]:
                new_entry["expected_answer"] = int(new_entry["expected_answer"])

            fout.write(json.dumps(new_entry) + "\n")

    if args.add_rounding_instructions:
        data_folder = Path(__file__).absolute().parent
        file = str(data_folder / f"{split_name}.jsonl")
        output_file = str(data_folder / f"{split_name}_rounded.jsonl")
        subprocess.run(
            f"{sys.executable} {data_folder.parent}/add_rounding_instructions.py --path {file} --save_path {output_file}",
            shell=True,
            check=True,
        )
