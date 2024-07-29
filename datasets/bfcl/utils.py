"""Preprocessing utils adapted from the gorilla library"""

import json
import os

API_KEYS_REQUIRED = [
    "RAPID_API_KEY",  # Get key from - https://rapidapi.com/hub
    "EXCHANGERATE_API_KEY",  # Get key from - https://www.exchangerate-api.com
    "OMDB_API_KEY",  # Get key from - http://www.omdbapi.com/apikey.aspx
    "GEOCODE_API_KEY",  # Get key from - https://geocode.maps.co/
]

KEY_PREFIX = "YOUR-"


# Function adapted from - https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/apply_function_credential_config.py
def _replace_placeholders(data, replacement_dict):
    """
    Recursively replace placeholders in a nested dictionary or list using string.replace.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                _replace_placeholders(value, replacement_dict)
            elif isinstance(value, str):
                for placeholder, actual_value in replacement_dict.items():
                    if placeholder in value:  # Check if placeholder is in the string
                        data[key] = value.replace(placeholder, actual_value)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, (dict, list)):
                _replace_placeholders(item, replacement_dict)
            elif isinstance(item, str):
                for placeholder, actual_value in replacement_dict.items():
                    if placeholder in item:  # Check if placeholder is in the string
                        data[idx] = item.replace(placeholder, actual_value)
    return data


# Function adapted from - https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/apply_function_credential_config.py
def process_api_in_file(file_path):
    api_key_dict = {}
    try:
        api_key_dict = {(KEY_PREFIX + api_key.replace("_", "-")): os.environ[api_key] for api_key in API_KEYS_REQUIRED}
        # print(api_key_dict)
    except KeyError:
        print(f"\033[91mWarning: Missing APIs, check environment variable - {API_KEYS_REQUIRED}\033[0m")

    modified_data = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            try:
                data = json.loads(line)
                data = _replace_placeholders(data, api_key_dict)  # Replace placeholders
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


# Adapted from here - https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/model_handler/utils.py
def augment_prompt_by_languge(prompt, test_category):
    if test_category == "java":
        prompt = prompt + "\nNote that the provided function is in Java 8 SDK syntax."
    elif test_category == "javascript":
        prompt = prompt + "\nNote that the provided function is in JavaScript."
    else:
        prompt = prompt + "\nNote that the provided function is in Python."
    return prompt


# Adapted from here - https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/model_handler/utils.py
def language_specific_pre_processing(function, test_category):
    if type(function) is dict:
        function = [function]
    if len(function) == 0:
        return function
    for item in function:
        properties = item["parameters"]["properties"]
        if test_category == "java":
            for key, value in properties.items():
                if value["type"] == "Any" or value["type"] == "any":
                    properties[key]["description"] += "This parameter can be of any type of Java object."
                    properties[key]["description"] += "This is Java" + value["type"] + " in string representation."
        elif test_category == "javascript":
            for key, value in properties.items():
                if value["type"] == "Any" or value["type"] == "any":
                    properties[key]["description"] += "This parameter can be of any type of Javascript object."
                else:
                    if "description" not in properties[key]:
                        properties[key]["description"] = ""
                    properties[key]["description"] += (
                        "This is Javascript " + value["type"] + " in string representation."
                    )
        return function
