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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

# guarding import to allow prompt_types to be used in scripts without extra packages required
try:
    from omegaconf import MISSING
except ImportError:
    MISSING = ''

from nemo_skills.inference.prompt.few_shot_examples import examples_map

# listing all available configs here
prompt_types = [cfg.stem for cfg in Path(__file__).parent.glob("*.yaml")]

# listing all dataset folders available - note this will not be available
# if using from installed package but you need to have data files available anyway
datasets = [
    d.name for d in (Path(__file__).parents[3] / 'datasets').glob("*") if d.is_dir() and d.name != "__pycache__"
]


@dataclass
class PromptConfig:
    delimiter: str = MISSING
    prefix: str = MISSING
    template: str = MISSING
    context_type: str = "empty"
    examples_type: Optional[str] = None
    num_few_shots: int = 5


def get_prompt_config(prompt_type: str) -> PromptConfig:
    # reading prompt format from the yaml file, if not running through hydra
    with open(Path(__file__).parent / f"{prompt_type}.yaml", "rt", encoding="utf-8") as fin:
        return PromptConfig(**yaml.safe_load(fin))


# this does not come from the config as it's coupled with few shot examples structure
# and so most often will require changes to the code anyway
context_templates = {
    "empty": "",
    "reference_solution": "Reference solution (do not copy it):\n{reference_solution}\n\n",
    "masked_solution": "Reference solution:\n{reference_masked_solution}\n\n",
}


def get_prompt(
    prompt_config: PromptConfig,
    input_dict: dict,
    examples: Optional[list[dict]] = None,
    context: Optional[str] = None,
):
    """Will build few-shot prompt from the provided pieces of information."""
    if prompt_config.num_few_shots != 0:
        examples = (
            examples[: prompt_config.num_few_shots]
            if examples is not None
            else examples_map[prompt_config.examples_type][
                : prompt_config.num_few_shots
            ]
        )
    else:
        examples = []
    context = (
        context
        if context is not None
        else context_templates[prompt_config.context_type]
    )

    filled_examples = []
    for example_dict in examples:
        filled_examples.append(prompt_config.template.format(context=context.format(**example_dict), **example_dict))
    filled_examples.append(
        prompt_config.template.format(context=context.format(**input_dict), **input_dict, generated_solution="")
    )

    return prompt_config.prefix + prompt_config.delimiter.join(filled_examples)
