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
from typing import Any, Dict, List, Optional

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
    example_template: str = MISSING
    prompt_template: str = MISSING
    prefix: Optional[str] = None
    system: Optional[str] = None
    context_type: str = "empty"
    examples_type: Optional[str] = None
    num_few_shots: int = 5


@dataclass
class Prompt:
    config: PromptConfig
    input_dict: Dict[str, Any]
    example_dicts: List[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize example_dicts if not provided."""
        if self.example_dicts is None:
            self.example_dicts = examples_map.get(self.config.examples_type, [])[: self.config.num_few_shots]

    def build_context(self, example_dict: Dict[str, Any]) -> str:
        """Builds the context string based on the example dictionary."""
        template = context_templates.get(self.config.context_type, "")
        return template.format(**example_dict)

    def build_filled_example(self, example_dict: Dict[str, Any]) -> str:
        """Builds a filled example string based on the example dictionary."""
        context = self.build_context(example_dict)
        return self.config.example_template.format(context=context, **example_dict)

    def build_examples(self) -> str:
        """Builds all examples string concatenated by delimiter."""
        filled_examples = [self.build_filled_example(example) for example in self.example_dicts]
        filled_examples.append(self.build_filled_example({**self.input_dict, 'generated_solution': ''}))
        return self.config.delimiter.join(filled_examples)

    def build_structured_examples(self) -> List[Dict[str, str]]:
        """Builds a structured representation of the prompt."""
        structured_prompt = [{"role": "system", "content": self.config.system or self.config.prefix}]
        for example in self.example_dicts:
            structured_prompt.append({"role": "user", "content": example['question']})
            if context := self.build_context(example):
                structured_prompt.append({"role": "user", "content": context})
            structured_prompt.append({"role": "assistant", "content": example['generated_solution']})
        structured_prompt.append({"role": "user", "content": self.input_dict['question']})
        if context := self.build_context(self.input_dict):
            structured_prompt.append({"role": "user", "content": context})
        return structured_prompt

    def __str__(self) -> str:
        """Returns the complete prompt string representation."""
        prompt = self.config.prompt_template.format(
            system=self.config.system,
            prefix=self.config.prefix,
            examples=self.build_examples(),
        )
        return prompt


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


def get_prompt(prompt_config: PromptConfig, input_dict: Dict[str, Any]) -> Prompt:
    """Builds and returns a string representation of a few-shot prompt from the provided configuration and input dictionary."""
    prompt = Prompt(prompt_config, input_dict)
    return prompt
