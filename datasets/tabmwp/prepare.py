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
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/lupantech/PromptPG/main/data/tabmwp/problems_test1k.json"

# Data Format
#
# Required:
#   - question (problem statement)
#
# Optional:
#   - expected_answer (expected answer)
#   - reference_solution (text-based solution)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Reads TabMWP data and converts it to a format readable by the llm-structured-data scripts."
    )
    parser.add_argument(
        "--data",
        default=f"{Path(__file__).absolute().parent / 'original' / 'test.jsonl'}",
        help="Path to JSON file with TabMWP data. Will automatically download if not found.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"{Path(__file__).absolute().parent / 'test.jsonl'}",
        help="Path to where the output file will be written.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        os.makedirs(os.path.dirname(args.data), exist_ok=True)
        os.system(f"wget {URL} -O {args.data}")

    original_file = args.data
    output_file = args.output

    question_template = "Read the following table and then answer the question that follows.\n{table}\n\n{question}"

    if not os.path.exists(original_file):
        urllib.request.urlretrieve(URL, original_file)

    with open(original_file, "r") as fin:
        data = json.load(fin)

    with open(output_file, "wt", encoding="utf-8") as fout:
        for original_entry in data.values():
            question = question_template.format(
                table_title=original_entry["table_title"],
                table=original_entry["table"],
                question=original_entry["question"],
            )

            new_entry = dict(
                question=question,
                expected_answer=original_entry["answer"],
                reference_solution=original_entry["solution"],
                table_title=original_entry["table_title"],
                table=original_entry["table_for_pd"],
                grade=original_entry["grade"],
            )

            if original_entry["ques_type"] == "multi_choice":
                new_entry["question"] += "\nAnswer options:\n{}".format(", ".join(original_entry["choices"]))

            # converting to int if able to for cleaner text representation
            if original_entry["ans_type"] == "integer_number":
                new_entry["expected_answer"] = int(new_entry["expected_answer"].replace(",", ""))

            fout.write(json.dumps(new_entry) + "\n")
