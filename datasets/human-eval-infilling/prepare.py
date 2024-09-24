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
import gzip
import json
import os
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/openai/human-eval-infilling/master/data/HumanEval-{split_name}.jsonl.gz"


def read_jsonl_file_from_gz(gz_file_path):
    # data format
    # {
    #     "task_id": "RandomSpanInfillingLight/HumanEval/0/1",
    #     "entry_point": "has_close_elements",
    #     "prompt": "\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):",
    #     "suffix": "\n                if distance < threshold:\n                    return True\n\n    return False\n\n",
    #     "canonical_solution": "\n            if idx != idx2:\n                distance = abs(elem - elem2)",
    #     "test": "\n\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n"
    # }

    data = []

    with gzip.open(gz_file_path, 'rt', encoding='utf-8') as gz_file:
        for line in gz_file:
            data.append(json.loads(line.strip()))

    return data


def save_data(split_name):
    data_folder = Path(__file__).absolute().parent
    data_file = str(data_folder / f"{split_name}.jsonl.gz")
    data_folder.mkdir(exist_ok=True)
    output_file = str(data_folder / f"{split_name}.jsonl")

    if not os.path.exists(data_file):
        urllib.request.urlretrieve(URL.format(
            split_name=split_name), data_file)

    problems = read_jsonl_file_from_gz(data_file)

    with open(output_file, "wt", encoding="utf-8") as fout:
        for problem in problems:
            # somehow models like tabs more than spaces
            problem['question'] = problem['prompt'].replace('    ', '\t')
            problem['suffix'] = problem['suffix'].replace('    ', '\t')
            fout.write(json.dumps(problem) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_name",
        default="RandomSpanInfilling",
        choices=("MultiLineInfilling", "SingleLineInfilling",
                 "RandomSpanInfilling", "RandomSpanInfillingLight"),
    )
    args = parser.parse_args()
    save_data(args.split_name)
