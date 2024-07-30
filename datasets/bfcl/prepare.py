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

import json
import time
import urllib.request
from os import path
from pathlib import Path

from utils import augment_prompt_by_language, language_specific_pre_processing, process_api_in_file

URL_PREFIX = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/data/"


AST_TEST_FILE_MAPPING = {
    "simple": "gorilla_openfunctions_v1_test_simple.json",
    "relevance": "gorilla_openfunctions_v1_test_relevance.json",
    "parallel_function": "gorilla_openfunctions_v1_test_parallel_function.json",
    "multiple_function": "gorilla_openfunctions_v1_test_multiple_function.json",
    "parallel_multiple_function": "gorilla_openfunctions_v1_test_parallel_multiple_function.json",
    "java": "gorilla_openfunctions_v1_test_java.json",
    "javascript": "gorilla_openfunctions_v1_test_javascript.json",
}

EXEC_TEST_FILE_MAPPING = {
    "executable_simple": "gorilla_openfunctions_v1_test_executable_simple.json",
    "executable_parallel_function": "gorilla_openfunctions_v1_test_executable_parallel_function.json",
    "executable_multiple_function": "gorilla_openfunctions_v1_test_executable_multiple_function.json",
    "executable_parallel_multiple_function": "gorilla_openfunctions_v1_test_executable_parallel_multiple_function.json",
    "rest": "gorilla_openfunctions_v1_test_rest.json",
}


if __name__ == "__main__":
    root_folder = Path(__file__).absolute().parent
    output_file = root_folder / f"test.jsonl"

    data_folder = root_folder / "data"
    data_folder.mkdir(exist_ok=True)

    with open(output_file, "w") as writer:
        # First process the AST test
        print("Preparing AST tests:")

        for test_category, test_file in AST_TEST_FILE_MAPPING.items():
            print(f"- Downloading {test_file}")
            local_path = path.join(data_folder, test_file)
            url_path = path.join(URL_PREFIX, test_file)

            urllib.request.urlretrieve(url_path, local_path)

            # Download the possible_answer folder
            if test_category != "relevance":
                answer_file_path = path.join(data_folder, "answer_" + test_file)
                url_path = path.join(path.join(URL_PREFIX, "possible_answer"), test_file)
                urllib.request.urlretrieve(url_path, answer_file_path)
                answers = [json.loads(line)["ground_truth"] for line in open(answer_file_path).readlines()]

            with open(local_path) as reader:
                for idx, line in enumerate(reader):
                    instance = json.loads(line.strip())
                    instance["function"] = language_specific_pre_processing(instance["function"], test_category)
                    instance["question"] = augment_prompt_by_languge(instance["question"], test_category)
                    instance["test_category"] = test_category
                    if test_category != "relevance":
                        instance["expected_answer"] = answers[idx]
                    writer.write(json.dumps(instance) + "\n")

            # Github can issue a 104 Error
            time.sleep(0.2)

        print("\nPreparing Execution tests:")
        for test_category, test_file in EXEC_TEST_FILE_MAPPING.items():
            print(f"- Downloading {test_file}")
            local_path = path.join(data_folder, test_file)
            url_path = path.join(URL_PREFIX, test_file)
            urllib.request.urlretrieve(url_path, local_path)

            process_api_in_file(local_path)

            with open(local_path) as reader:
                for idx, line in enumerate(reader):
                    instance = json.loads(line.strip())
                    instance["function"] = language_specific_pre_processing(instance["function"], test_category)
                    instance["question"] = augment_prompt_by_languge(instance["question"], test_category)
                    instance["test_category"] = test_category
                    writer.write(json.dumps(instance) + "\n")
