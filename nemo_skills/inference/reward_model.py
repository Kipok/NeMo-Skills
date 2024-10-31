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

import importlib
import json
import logging
import sys
from copy import deepcopy
from dataclasses import asdict, field
from pathlib import Path

import hydra
from tqdm import tqdm

from nemo_skills.code_execution.sandbox import get_sandbox, sandbox_params
from nemo_skills.inference.server.code_execution_model import (
    ErrorRecoveryConfig,
    server_params,
)
from nemo_skills.inference.server.reward_model import (
    get_reward_model,
)
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_fields_docstring, get_help_message, nested_dataclass, setup_logging

LOG = logging.getLogger(__file__)


@nested_dataclass(kw_only=True)
class RewardModelConfig:
    """LLM reward model parameters."""

    output_file: str | None = None  # Where to save the generations
    # Inference server configuration {server_params} {error_recovery_params}
    server: dict = field(default_factory=dict)
    # Sandbox configuration {sandbox_params}
    sandbox: dict = field(default_factory=dict)
    # Prompt configuration - path to yaml files
    prompt_template: str | None = None  # not required for OpenAI server
    prompt_config: str | None = None  # we will fetch it from dataset dir if not provided
    examples_type: str | None = None  # to be able to customize few-shot examples

    # Can specify one of the existing datasets.
    dataset: str | None = None
    split: str | None = None  # Can be train, validation, test or train_full (train + validation)
    input_file: str | None = None  # Can directly specify an input file, if using a custom dataset
    input_dir: str | None = None # Can specify an input direct
    output_dir: str | None = None
    random_seed: str | None = None

    batch_size: int = 128
    max_samples: int = -1  # If > 0, will stop after generating this many samples. Useful for debugging
    skip_filled: bool = False  # If True, will skip the generations that are already in the output file

    # if > 0, will skip this many samples from the beginning of the data file.
    # Useful if need to run multiple slurm jobs on the same data file
    offset: int = 0

    generation_key: str = "generation"

    # can add this flag to just print the first prompt instead of running generation
    # useful to double check that your data can be loaded and prompt has what you expect
    dry_run: bool = False

    def __post_init__(self):
        if self.input_dir is None and self.input_file is None:
            if self.dataset is None or self.split is None:
                raise ValueError("Either `input_file`, `input_directory` or `dataset` and `split` should be provided")
            self.input_file = Path(__file__).parents[1] / "dataset" / self.dataset / f"{self.split}.jsonl"
            if self.output_file is None:
                raise ValueError("Output file should be provided if using `dataset` and `split`")
        elif self.input_file is None and self.input_dir is not None:
            self.check_dataset_and_split_none(against='`input_directory`')
            seed = f'rs{self.random_seed}' if self.random_seed is not None else 'greedy'
            self.input_file = Path(self.input_dir) / f"output-{seed}.jsonl"
            self.output_file = Path(self.output_dir) / f"output-{seed}.jsonl"
        elif self.input_file is not None and self.input_dir is None:
            self.check_dataset_and_split_none(against='`input_file`')
            if self.output_file is None:
                raise ValueError("Output file should be provided if providing `input_file`")
        else:
            raise ValueError("`input_file` and `input_directory` cannot be provided at the same time")


        if self.dataset is None and self.prompt_config is None:
            raise ValueError("If `dataset` is not provided, `prompt_config` is required")

        if self.server["server_type"] != "openai" and self.prompt_template is None:
            raise ValueError("Prompt template is required for non-OpenAI servers")

        if self.server["server_type"] == "openai" and self.prompt_template is not None:
            raise ValueError("Prompt template is not supported for OpenAI server")

    def check_dataset_and_split_none(self, against):
        if self.dataset is not None or self.split is not None:
            raise ValueError(f"Either {against} or `dataset` and `split` should be provided, but not both")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_reward_model_config", node=RewardModelConfig)


@hydra.main(version_base=None, config_name='base_reward_model_config')
def generate(cfg: RewardModelConfig):
    cfg = RewardModelConfig(_init_nested=True, **cfg)

    LOG.info("Config used: %s", cfg)
    llm = get_reward_model(**cfg.server)

    # making sure output dir exists
    Path(cfg.output_file).absolute().parent.mkdir(parents=True, exist_ok=True)

    # we currently assume the dataset is small enough to be loaded into memory
    data = []
    with open(cfg.input_file, "rt", encoding="utf-8") as fin:
        for line in fin:
            data.append(json.loads(line))

    # skipping based on the offset first
    data = data[cfg.offset :]

    starting_idx = 0
    if cfg.skip_filled:
        try:
            with open(cfg.output_file, "rt", encoding="utf-8") as fin:
                starting_idx = len(fin.readlines())
        except FileNotFoundError:
            LOG.warning(f"File `{cfg.output_file}` not found, starting from scratch")

    # additionally, skipping whatever is pre-filled, assuming offset didn't change
    data = data[starting_idx:]
    if cfg.prompt_config is None:
        # fetching from the default for corresponding dataset
        dataset_module = importlib.import_module(f"nemo_skills.dataset.{cfg.dataset}")
        cfg.prompt_config = dataset_module.PROMPT_CONFIG

    prompt = get_prompt(cfg.prompt_config, cfg.prompt_template, examples_type=cfg.examples_type)
    LOG.info("Prompt used: %s", prompt)

    # need to account for anything that's prefilled
    if 0 <= cfg.max_samples <= starting_idx:
        cfg.max_samples = 0

    if starting_idx < cfg.max_samples:
        cfg.max_samples -= starting_idx

    if cfg.max_samples < 0 or cfg.max_samples > len(data):
        cfg.max_samples = len(data)

    if len(data) == 0:  # we might not have any examples if skip_filled=True
        return

    LOG.info("Example prompt:\nData dictionary: %s\nPrompt: %s", data[0], prompt.fill(data[0]))

    if cfg.dry_run:
        return

    # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
    with open(cfg.output_file, "at" if cfg.skip_filled else "wt", encoding="utf-8", buffering=1) as fout:
        data_points = []
        for idx, data_point in tqdm(enumerate(data), initial=starting_idx, total=len(data) + starting_idx):
            if idx >= cfg.max_samples:
                break
            data_point.pop(cfg.generation_key, None)
            data_points.append(data_point)

            if len(data_points) == cfg.batch_size or idx == cfg.max_samples - 1:
                outputs = llm.score(
                    prompts=[prompt.fill(dp) for dp in data_points],
                )

                for output, original_data_point in zip(outputs, data_points):
                    # to make it easier to follow up with evaluation and limit accidental errors, we are adding
                    # all of the ground-truth data to the output file alongside the generated solutions
                    output[cfg.generation_key] = output.pop("generation")
                    output.update(original_data_point)

                    fout.write(json.dumps(output) + "\n")
                data_points = []


error_recovery_params = '\n' + get_fields_docstring(
    ErrorRecoveryConfig,
    prefix='server.error_recovery.',
    level=2,
)


HELP_MESSAGE = get_help_message(
    RewardModelConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
    error_recovery_params=error_recovery_params,
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()
