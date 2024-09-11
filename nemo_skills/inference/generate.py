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
from dataclasses import asdict, field
from pathlib import Path

import hydra
from tqdm import tqdm

from nemo_skills.code_execution.sandbox import get_sandbox, sandbox_params
from nemo_skills.inference.server.code_execution_model import (
    ErrorRecoveryConfig,
    get_code_execution_model,
    get_model,
    server_params,
)
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_fields_docstring, get_help_message, nested_dataclass, setup_logging

LOG = logging.getLogger(__file__)


@nested_dataclass(kw_only=True)
class InferenceConfig:
    temperature: float = 0.0  # Temperature of 0 means greedy decoding
    top_k: int = 0
    top_p: float = 0.95
    random_seed: int = 0
    tokens_to_generate: int = 2048
    repetition_penalty: float = 1.0


@nested_dataclass(kw_only=True)
class GenerateSolutionsConfig:
    """LLM generation parameters."""

    output_file: str  # Where to save the generations
    # Inference server configuration {server_params} {error_recovery_params}
    server: dict = field(default_factory=dict)
    # Sandbox configuration {sandbox_params}
    sandbox: dict = field(default_factory=dict)
    # Prompt configuration - path to yaml files
    prompt_template: str | None = None  # not required for OpenAI server
    prompt_config: str | None = None  # we will fetch it from dataset folder if not provided
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters

    # Can specify one of the existing datasets.
    dataset: str | None = None
    split_name: str | None = None  # Can be train, validation, test or train_full (train + validation)
    data_file: str | None = None  # Can directly specify a data file, if using a custom dataset

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

    # set to True if code execution needs to be supported
    code_execution: bool = False

    def __post_init__(self):
        """Building data_file from dataset/split_name if not provided directly."""
        if self.data_file is not None:
            if self.dataset is not None or self.split_name is not None:
                raise ValueError("Either `data_file` or `dataset` and `split_name` should be provided, but not both")
        else:
            if self.dataset is None or self.split_name is None:
                raise ValueError("Either `data_file` or `dataset` and `split_name` should be provided")
            self.data_file = Path(__file__).parents[1] / "dataset" / self.dataset / f"{self.split_name}.jsonl"

        if self.dataset is None and self.prompt_config is None:
            raise ValueError("If `dataset` is not provided, `prompt_config` is required")

        if self.server.server_type != "openai" and self.prompt_template is None:
            raise ValueError("Prompt template is required for non-OpenAI servers")

        if self.server.server_type == "openai" and self.prompt_template is not None:
            raise ValueError("Prompt template is not supported for OpenAI server")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_generation_config", node=GenerateSolutionsConfig)


@hydra.main(version_base=None, config_name='base_generation_config')
def generate(cfg: GenerateSolutionsConfig):
    cfg = GenerateSolutionsConfig(_init_nested=True, **cfg)

    LOG.info("Config used: %s", cfg)
    if cfg.code_execution:
        sandbox = get_sandbox(**cfg.sandbox) if cfg.sandbox is not None else None
        llm = get_code_execution_model(**cfg.server, sandbox=sandbox)
    else:
        llm = get_model(**cfg.server)

    # making sure output folder exists
    Path(cfg.output_file).absolute().parent.mkdir(parents=True, exist_ok=True)

    # we currently assume the dataset is small enough to be loaded into memory
    data = []
    with open(cfg.data_file, "rt", encoding="utf-8") as fin:
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

    prompt = get_prompt(cfg.prompt_config, cfg.prompt_template)
    LOG.info("Prompt used: %s", prompt)

    if cfg.max_samples < 0:
        cfg.max_samples = len(data)

    if cfg.prompt_template:
        LOG.info("Example prompt:\nData dictionary: %s\nPrompt string: %s", data[0], prompt.build_string(data[0]))
    else:
        LOG.info("Example prompt:\nData dictionary: %s\nPrompt messages: %s", data[0], prompt.build_messages(data[0]))
    if cfg.dry_run:
        return

    # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
    with open(cfg.output_file, "at" if cfg.skip_filled else "wt", encoding="utf-8", buffering=1) as fout:
        data_points = []
        for idx, data_point in tqdm(enumerate(data), initial=starting_idx, total=len(data) + starting_idx):
            if idx >= cfg.max_samples:
                break

            data_points.append(data_point)

            if len(data_points) == cfg.batch_size or idx == cfg.max_samples - 1:
                if cfg.prompt_template:
                    prompts = [prompt.build_string(dp) for dp in data_points]
                    stop_phrases = list(prompt.config.template.stop_phrases)
                else:
                    prompts = [prompt.build_messages(dp) for dp in data_points]
                    stop_phrases = None

                outputs = llm.generate(prompts=prompts, stop_phrases=stop_phrases, **asdict(cfg.inference))

                for output, original_data_point in zip(outputs, data_points):
                    # to make it easier to follow up with evaluation and limit accidental errors, we are adding
                    # all of the ground-truth data to the output file alongside the generated solutions
                    if cfg.generation_key != "generation":
                        output[cfg.generation_key] = output.pop("generation")
                        original_data_point.pop(cfg.generation_key, None)
                    else:
                        original_data_point.pop('generation', None)
                    output.update(original_data_point)

                    fout.write(json.dumps(output) + "\n")
                data_points = []


error_recovery_params = '\n' + get_fields_docstring(
    ErrorRecoveryConfig,
    prefix='server.error_recovery.',
    level=2,
)


HELP_MESSAGE = get_help_message(
    GenerateSolutionsConfig,
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
