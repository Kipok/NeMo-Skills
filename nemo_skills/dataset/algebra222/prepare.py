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

import csv
import json
import os
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/joyheyueya/declarative-math-word-problem/main/algebra222.csv"


if __name__ == "__main__":
    data_folder = Path(__file__).absolute().parent
    original_file = str(data_folder / f"original_test.csv")
    data_folder.mkdir(exist_ok=True)
    output_file = str(data_folder / f"test.jsonl")

    if not os.path.exists(original_file):
        urllib.request.urlretrieve(URL, original_file)

    with open(original_file, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = [{h: x for (h, x) in zip(headers, row)} for row in reader]

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in data:
            sample = dict(
                problem=entry["question"],
                expected_answer=float(entry["final_answer"]),
            )
            if int(sample["expected_answer"]) == sample["expected_answer"]:
                sample["expected_answer"] = int(sample["expected_answer"])
            fout.write(json.dumps(sample) + "\n")
