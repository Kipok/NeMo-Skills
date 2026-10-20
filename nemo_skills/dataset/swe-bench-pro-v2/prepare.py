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

import argparse
from pathlib import Path

import datasets

from nemo_skills.dataset.utils import get_swe_sanitization_command

DATASET_CONFIGS = ("default", "hard")

# Convert language codes to the same format as swe-bench-multilingual.
LANGUAGE_MAP = {
    "js": "javascript",
    "ts": "typescript",
    "go": "go",
    "python": "python",
}

# These repositories use Alpine-based task images and require an Alpine-based
# NeMo-Skills container when the task image is run through Apptainer.
ALPINE_REPOS = {
    "flipt-io/flipt",
    "future-architect/vuls",
    "gravitational/teleport",
    "navidrome/navidrome",
    "protonmail/webclients",
}


def format_container(container_formatter: str, docker_image: str) -> str:
    if container_formatter.startswith("docker://"):
        image_name = docker_image
    else:
        # dump_images.py uses this normalization when naming local SIF files.
        image_name = docker_image.replace("/", "_").replace(":", "_")
    return container_formatter.format(docker_image=image_name)


def prepare_config(
    dataset_name: str,
    dataset_config: str,
    split: str,
    container_formatter: str,
) -> None:
    dataset = datasets.load_dataset(path=dataset_name, name=dataset_config, split=split)

    required_columns = {"docker_image", "hard", "version"}
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Dataset config {dataset_config!r} is not SWE-bench Pro V2; missing columns: {sorted(missing_columns)}"
        )

    dataset = dataset.map(
        lambda row: {
            "language": LANGUAGE_MAP[row["repo_language"]],
            "problem_statement": (
                f"{row['problem_statement']}\n\n"
                f"Requirements:\n{row['requirements']}\n\n"
                f"New interfaces introduced:\n{row['interface']}"
            ),
        },
        remove_columns=["repo_language", "interface", "requirements"],
    )

    dataset = dataset.add_column(
        "container_formatter",
        [format_container(container_formatter, row["docker_image"]) for row in dataset],
    )
    dataset = dataset.add_column("container_id", list(range(len(dataset))))
    dataset = dataset.add_column("dataset_name", [dataset_name] * len(dataset))
    dataset = dataset.add_column("split", [split] * len(dataset))
    dataset = dataset.add_column("container_repo_dir", ["/app"] * len(dataset))
    dataset = dataset.add_column("pre_commands", [get_swe_sanitization_command()] * len(dataset))

    output_dir = Path(__file__).parent
    alpine_dataset = dataset.filter(lambda row: row["repo"].lower() in ALPINE_REPOS)
    alpine_dataset.to_json(output_dir / f"{dataset_config}.alpine.jsonl", orient="records", lines=True)
    ubuntu_dataset = dataset.filter(lambda row: row["repo"].lower() not in ALPINE_REPOS)
    ubuntu_dataset.to_json(output_dir / f"{dataset_config}.ubuntu.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--container_formatter",
        type=str,
        default="docker://{docker_image}",
        help="Container formatter string. For local SIF images, use a path such as "
        "'/swe-bench-images/{docker_image}.sif'.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ScaleAI/SWE-bench_Pro",
        help="Dataset name to load",
    )
    parser.add_argument(
        "--dataset_config",
        nargs="+",
        choices=DATASET_CONFIGS,
        default=DATASET_CONFIGS,
        help="Dataset configurations to prepare. Both V2 and HARD-51 are prepared by default.",
    )
    parser.add_argument("--split", type=str, default="test", help="SWE-bench dataset split to use")
    args = parser.parse_args()

    if "{docker_image}" not in args.container_formatter:
        parser.error("container_formatter must contain {docker_image}")

    for config in args.dataset_config:
        prepare_config(
            dataset_name=args.dataset_name,
            dataset_config=config,
            split=args.split,
            container_formatter=args.container_formatter,
        )
