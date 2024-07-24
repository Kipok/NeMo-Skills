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
import urllib.request
import json
from os import path
from pathlib import Path


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

API_KEYS_REQUIRED = [
    "RAPID_API_KEY",  # Get key from - https://rapidapi.com/hub
    "EXCHANGERATE_API_KEY",  # Get key from - https://www.exchangerate-api.com
    "OMDB_API_KEY",  # Get key from - http://www.omdbapi.com/apikey.aspx
    "GEOCODE_API_KEY",  # Get key from - https://geocode.maps.co/
]

KEY_PREFIX = "YOUR-"


# Function adapted from - https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/apply_function_credential_config.py
def replace_placeholders(data, replacement_dict):
    """
    Recursively replace placeholders in a nested dictionary or list using string.replace.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                replace_placeholders(value, replacement_dict)
            elif isinstance(value, str):
                for placeholder, actual_value in replacement_dict.items():
                    if placeholder in value:  # Check if placeholder is in the string
                        data[key] = value.replace(placeholder, actual_value)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, (dict, list)):
                replace_placeholders(item, replacement_dict)
            elif isinstance(item, str):
                for placeholder, actual_value in replacement_dict.items():
                    if placeholder in item:  # Check if placeholder is in the string
                        data[idx] = item.replace(placeholder, actual_value)
    return data


# Function adapted from - https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/apply_function_credential_config.py
def process_file(file_path, replacement_dict):
    modified_data = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            try:
                data = json.loads(line)
                data = replace_placeholders(data, replacement_dict)  # Replace placeholders
                modified_data.append(json.dumps(data))
            except json.JSONDecodeError:
                print("Invalid JSON line skipped.")
                continue
            
    with open(file_path, "w") as f:
        for i, modified_line in enumerate(modified_data):
            f.write(modified_line)
            if i < len(lines) - 1:
                f.write("\n")

    print(f"API key placeholders have been replaced for {file_path}")
    

if __name__ == "__main__":
    data_folder = Path(__file__).absolute().parent / "data"
    data_folder.mkdir(exist_ok=True)
    
    answer_folder = data_folder / "possible_answer"
    answer_folder.mkdir(exist_ok=True)

    # First process the AST test
    print("Preparing AST tests:")
    for test, test_file in AST_TEST_FILE_MAPPING.items():
        print(f"- Downloading {test_file}")
        local_path = path.join(data_folder, test_file)
        url_path = path.join(URL_PREFIX, test_file)
        urllib.request.urlretrieve(url_path, local_path)

        # Download the possible_answer folder
        if test != "relevance":
            local_path = path.join(answer_folder, test_file)
            url_path = path.join(path.join(URL_PREFIX, "possible_answer"), test_file)
            urllib.request.urlretrieve(url_path, local_path)

    print("\nPreparing Execution tests:")
    for test, test_file in EXEC_TEST_FILE_MAPPING.items():
        print(f"- Downloading {test_file}")
        local_path = path.join(data_folder, test_file)
        url_path = path.join(URL_PREFIX, test_file)
        urllib.request.urlretrieve(url_path, local_path)

    # Update the REST test with APIs
    print("\nPreparing REST API test:")
    rest_test_file = path.join(data_folder, EXEC_TEST_FILE_MAPPING["rest"])
    try:
        api_key_dict = {
            (KEY_PREFIX + api_key.replace("_", "-")): os.environ[api_key] for api_key in API_KEYS_REQUIRED 
        }
        # print(api_key_dict)
    except KeyError:
         raise SystemExit(f"Missing APIs, check environment variable - {API_KEYS_REQUIRED}")

    process_file(rest_test_file, api_key_dict)

