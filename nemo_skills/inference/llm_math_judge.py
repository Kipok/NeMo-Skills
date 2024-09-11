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

import hydra
from tqdm import tqdm

from nemo_skills.code_execution.sandbox import get_sandbox, sandbox_params
from nemo_skills.inference.generate import GenerateSolutionsConfig
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
class LlmMathJudgeConfig(GenerateSolutionsConfig):
    """Top-level parameters for the script"""

    prompt_config: str = "judge/math"
    generation_key: str = "judgement"


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_llm_math_judge_config", node=LlmMathJudgeConfig)


def prefill_judgement(data_point: dict) -> str | None:
    """Will automatically fill judgement if there is an exact match or the answer is None."""
    if data_point['predicted_answer'] is None:
        return "No answer was provided.\nJudgement: No"

    if str(data_point['predicted_answer']).strip() == str(data_point['expected_answer']).strip():
        return "The two answers are identical.\nJudgement: Yes"

    return None


@hydra.main(version_base=None, config_name='base_llm_math_judge_config', config_path='.')
def llm_math_judge(cfg: GenerateSolutionsConfig):
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
        judgements = []
        prefilled_indices = []
        for idx, data_point in tqdm(enumerate(data), initial=starting_idx, total=len(data) + starting_idx):
            if idx >= cfg.max_samples:
                break

            data_point.pop('judgement', None)
            judgement = prefill_judgement(data_point)
            if judgement is None:
                data_points.append(data_point)
            else:
                judgements.append({'judgement': judgement})
                judgements[-1].update(data_point)
                prefilled_indices.append(idx)

            if len(data_points) == cfg.batch_size or idx == cfg.max_samples - 1:
                if cfg.prompt_template:  # using strings as input if template is provided
                    outputs = llm.generate(
                        prompts=[prompt.build_string(data_point) for data_point in data_points],
                        stop_phrases=list(prompt.config.template.stop_phrases),
                        **asdict(cfg.inference),
                    )
                else:
                    outputs = llm.generate(
                        prompts=[prompt.build_messages(data_point) for data_point in data_points],
                        **asdict(cfg.inference),
                    )

                prefilled_idx = 0
                generated_idx = 0
                for cur_idx in range(idx - len(data_points) - len(judgements) + 1, idx + 1):
                    if cur_idx in prefilled_indices:
                        fout.write(json.dumps(judgements[prefilled_idx]) + "\n")
                        prefilled_idx += 1
                    else:
                        output, original_data_point = outputs[generated_idx], data_points[generated_idx]
                        output['judgement'] = output.pop("generation")
                        output.update(original_data_point)
                        fout.write(json.dumps(output) + "\n")
                        generated_idx += 1
                data_points = []
                judgements = []
                prefilled_indices = []


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
        llm_math_judge()
