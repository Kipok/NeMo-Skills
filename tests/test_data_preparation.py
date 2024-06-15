import hashlib
import os
import subprocess

import requests


def download_data(split):
    # URL of the data file
    directory = "open-math-instruct-1"
    if not os.path.exists(directory):
        os.makedirs(directory)

    url = f"https://huggingface.co/datasets/nvidia/OpenMathInstruct-1/resolve/main/correct_solutions/{split}.jsonl?download=true"
    file_path = f"{directory}/{split}.jsonl"

    # Download the data using requests
    response = requests.get(url)
    response.raise_for_status()  # Ensure that the request was successful
    with open(file_path, 'wb') as f:
        f.write(response.content)

    return file_path


def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def test_code_files():
    output_file = "tests/data/processed_output.jsonl"
    subprocess.run(
        [
            "python",
            "nemo_skills/finetuning/prepare_sft_data.py",
            "prediction_jsonl_files='tests/data/output-rs*.test'",
            f"output_path={output_file}",
            "filters.drop_multi_boxed=true",
            "filters.drop_broken_code=true",
            "filters.trim_solutions=true",
            "filters.drop_incorrect_arithmetic=false",
            "filters.split_arithmetic=false",
            "filters.code_text_filter=null",
            "num_output_samples=null",
            "downsampling_method=null",
            "do_shuffle=false",
        ],
        check=True,
    )

    expected_md5 = "779c70a2d84d96997336bcd47b3e99f9"
    output_md5 = compute_md5(output_file)

    assert (
        expected_md5 == output_md5
    ), "MD5 hashes do not match, something is wrong with nemo_skills/finetuning/prepare_sft_data.py"


def test_openmathinstruct():
    download_data("validation")
    output_file = "open-math-instruct-1/validation-sft.jsonl"

    subprocess.run(
        [
            "python",
            "nemo_skills/finetuning/prepare_sft_data.py",
            "preprocessed_dataset_files='open-math-instruct-1/validation.jsonl'",
            f"output_path={output_file}",
            "filters.drop_multi_boxed=true",
            "filters.drop_broken_code=true",
            "filters.trim_solutions=true",
            "filters.drop_incorrect_arithmetic=false",
            "filters.split_arithmetic=false",
            "filters.code_text_filter=any_code",
            "num_output_samples=2048",
            "downsampling_method=fair",
            "do_shuffle=true",
        ],
        check=True,
    )

    expected_md5 = "c105c1c3369e5cee569dcba74e7d4d61"
    output_md5 = compute_md5(output_file)

    assert (
        expected_md5 == output_md5
    ), "MD5 hashes do not match, something is wrong with nemo_skills/finetuning/prepare_sft_data.py"
