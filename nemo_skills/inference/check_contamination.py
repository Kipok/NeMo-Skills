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
import logging
import sys
from dataclasses import asdict, field
from pathlib import Path
from typing import Any

import hydra
from tqdm import tqdm

from nemo_skills.code_execution.sandbox import get_sandbox, sandbox_params
from nemo_skills.inference.generate import InferenceConfig
from nemo_skills.inference.server.code_execution_model import get_model, server_params
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_help_message, nested_dataclass, setup_logging, unroll_files

LOG = logging.getLogger(__file__)


@nested_dataclass(kw_only=True)
class CheckContaminationConfig:
    """Top-level parameters for the script"""

    input_file: str  # an output of the retrieve_similar.py script
    output_file: str  # where to save the generations
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)
    # Prompt configuration - path to yaml files
    prompt_template: str | None = None  # not required for OpenAI server
    prompt_config: str = "judge/check-contamination"
    examples_type: str | None = None  # to be able to customize few-shot examples
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters

    batch_size: int = 128
    generation_key: str = "contaminated"

    skip_filled: bool = False  # skip already filled generations

    # TODO: support this
    # ask both with question / candidate and candidate / question and fill True if any is True
    # ask_both_ways: bool = False

    # can add this flag to just print the first prompt instead of running generation
    # useful to double check that your data can be loaded and prompt has what you expect
    dry_run: bool = False

    def __post_init__(self):
        if self.server.server_type != "openai" and self.prompt_template is None:
            raise ValueError("Prompt template is required for non-OpenAI servers")

        if self.server.server_type == "openai" and self.prompt_template is not None:
            raise ValueError("Prompt template is not supported for OpenAI server")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_check_contamination_config", node=CheckContaminationConfig)


def prefill_result(question: str, candidate: str) -> str | None:
    """Will automatically fill result if there is an exact match."""
    # TODO: update with ngram match?
    if question.strip() == candidate.strip():
        return "True"
    return None


# TODO: try to unify common parts in 3 similar generation scripts
@hydra.main(version_base=None, config_name='base_check_contamination_config', config_path='.')
def check_contamination(cfg: CheckContaminationConfig):
    cfg = CheckContaminationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    llm = get_model(**cfg.server)
    prompt = get_prompt(cfg.prompt_config, cfg.prompt_template, examples_type=cfg.examples_type)
    LOG.info("Prompt used: %s", prompt)

    # assuming everything fits in memory for simplicity
    with open(cfg.input_file, 'rt', encoding='utf-8') as fin:
        data = [json.loads(line) for line in fin]

    if cfg.prompt_template:
        LOG.info("Example prompt:\nData dictionary: %s\nPrompt string: %s", data[0], prompt.build_string(data[0]))
    else:
        LOG.info("Example prompt:\nData dictionary: %s\nPrompt messages: %s", data[0], prompt.build_messages(data[0]))
    if cfg.dry_run:
        return

    data_points = []
    prefilled_results = []
    prefilled_indices = []

    starting_idx = 0
    if cfg.skip_filled:
        try:
            with open(cfg.output_file, "rt", encoding="utf-8") as fin:
                starting_idx = len(fin.readlines())
        except FileNotFoundError:
            LOG.warning(f"File `{cfg.output_file}` not found, starting from scratch")
    data = data[starting_idx:]

    with open(cfg.output_file, "at" if cfg.skip_filled else "wt", encoding="utf-8", buffering=1) as fout:
        for idx, data_point in enumerate(tqdm(data, initial=starting_idx, total=len(data) + starting_idx)):
            data_point.pop(cfg.generation_key, None)
            result = prefill_result(data_point)
            if result is None:
                data_points.append(data_point)
            else:
                prefilled_results.append({cfg.generation_key: result, **data_point})
                prefilled_indices.append(idx)

            if len(data_points) == cfg.batch_size or idx == len(data) - 1:
                if cfg.prompt_template:
                    prompts = [prompt.build_string(dp) for dp in data_points]
                    stop_phrases = list(prompt.config.template.stop_phrases)
                else:
                    prompts = [prompt.build_messages(dp) for dp in data_points]
                    stop_phrases = None

                outputs = llm.generate(prompts=prompts, stop_phrases=stop_phrases, **asdict(cfg.inference))

                prefilled_idx = 0
                generated_idx = 0
                for cur_idx in range(idx - len(data_points) - len(prefilled_results) + 1, idx + 1):
                    if cur_idx in prefilled_indices:
                        fout.write(json.dumps(prefilled_results[prefilled_idx]) + "\n")
                        prefilled_idx += 1
                    else:
                        output = outputs[generated_idx]
                        output[cfg.generation_key] = output.pop("generation")
                        output.update(data_points[generated_idx])
                        fout.write(json.dumps(output) + "\n")
                        generated_idx += 1

                data_points.clear()
                prefilled_results.clear()
                prefilled_indices.clear()


HELP_MESSAGE = get_help_message(
    CheckContaminationConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        check_contamination()