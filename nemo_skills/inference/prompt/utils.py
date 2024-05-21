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
from dataclasses import asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import MISSING, OmegaConf

from nemo_skills.inference.prompt.few_shot_examples import examples_map
from nemo_skills.utils import nested_dataclass

# listing all available configs here
prompt_types = [cfg.stem for cfg in Path(__file__).parent.glob("*.yaml")]

# listing all dataset folders available - note this will not be available
# if using from installed package but you need to have data files available anyway
datasets = [
    d.name for d in (Path(__file__).parents[3] / 'datasets').glob("*") if d.is_dir() and d.name != "__pycache__"
]


class BM25Retriever:
    def __init__(self, data_path: str, field: str):
        from rank_bm25 import BM25Okapi

        with open(data_path, "rt", encoding="utf-8") as fin:
            self.entries = [json.loads(x) for x in fin]

        corpus = [entry[field] for entry in self.entries]
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 1):
        tokenized_query = query.split(" ")
        return self.bm25.get_top_n(tokenized_query, self.entries, n=top_k)


@nested_dataclass
class FewShotExamples:
    template: str = MISSING
    num_few_shots: int = 5

    examples_type: Optional[str] = None
    example_dicts: Optional[List[Dict[str, Any]]] = None

    retrieval_field: Optional[str] = None  # e.g. question, reference_solution, etc.
    retrieval_file: Optional[str] = None  # needs to be provided if retrieval_field is not None
    retriever: Optional[Any] = None

    def __post_init__(self):
        """Error checks + building example_dicts and retriever if needed."""
        if self.examples_type is not None:  # building example_dicts
            if self.example_dicts is not None:
                raise ValueError(
                    "You specified both examples_type and example_dicts. "
                    "This is redundant, so please remove one to avoid accidental errors."
                )
            self.example_dicts = examples_map[self.examples_type][: self.num_few_shots]

        if self.example_dicts is not None and self.num_few_shots > len(self.example_dicts):
            raise ValueError(
                f"There are not enough few shot examples in {self.examples_type}. "
                f"Max number is {len(self.example_dicts)}"
            )

        if self.retriever is not None:
            return

        if self.retrieval_field is not None:  # building retriever
            if self.retrieval_file is None:
                raise ValueError("retrieval_file must be provided if retrieval_field is not None")

            if self.retriever is not None:
                raise ValueError(
                    "You specified both retrieval field/file and retriever. "
                    "This is redundant, so please remove one to avoid accidental errors."
                )

            self.retriever = BM25Retriever(self.retrieval_file, field=self.retrieval_field)
        else:
            if self.retrieval_file is not None:
                raise ValueError("retrieval_field must be provided if retrieval_file is not None")

        if self.example_dicts is None and self.retriever is None and self.num_few_shots > 0:
            raise ValueError("You need to construct either example_dicts or retriever if num_few_shots > 0")

        if self.example_dicts is not None and self.retriever is not None:
            raise ValueError("example_dicts and retriever cannot be used together")

    def get_examples(self, input_dict):
        if self.num_few_shots == 0:
            return []

        if self.example_dicts:
            return self.example_dicts

        example_dicts = self.retriever.retrieve(
            query=input_dict[self.retrieval_field],
            top_k=self.num_few_shots + 1,
        )
        reference = input_dict[self.retrieval_field]
        # filtering exact match if it's there
        if example_dicts[0][self.retrieval_field] == reference:
            example_dicts = example_dicts[1:]
        else:  # removing the last one to match desired number of examples
            example_dicts = example_dicts[:-1]
        # if still has a match, let's error out for now
        for example_dict in example_dicts:
            if example_dict[self.retrieval_field] == reference:
                raise ValueError("Exact match found in retrieved examples")

        # let's reverse the order to show the most relevant last
        return example_dicts[::-1]


@nested_dataclass
class PromptConfig:
    few_shot_examples: FewShotExamples = field(default_factory=FewShotExamples)
    prompt_template: str = MISSING
    user: str = MISSING
    system: str = MISSING
    context_type: str = "empty"
    stop_phrases: List[str] = field(default_factory=list)


@nested_dataclass
class Prompt:
    config: PromptConfig
    input_dict: Dict[str, Any]
    context_template: Optional[str] = None
    generated_solution: str = ""

    def __post_init__(self):
        """Initialize context_template if not provided."""
        if self.context_template is None:
            self.context_template = context_templates.get(self.config.context_type, "")

    def build_context(self, example_dict: Dict[str, Any]) -> str:
        """Builds the context string based on the example dictionary."""
        context = self.context_template.format(**example_dict)
        return context

    def build_filled_example(self, example_dict: Dict[str, Any]) -> str:
        """Builds a filled example string based on the example dictionary."""
        context = self.build_context(example_dict)
        return self.config.few_shot_examples.template.format(context=context, **example_dict)

    def build_examples(self) -> str:
        """Builds all examples string concatenated by delimiter."""
        example_dicts = self.config.few_shot_examples.get_examples(self.input_dict)

        filled_examples = [self.build_filled_example(example) for example in example_dicts]
        examples = "".join(filled_examples)
        context = self.build_context(self.input_dict)
        user = self.config.user.format(examples=examples, context=context, **self.input_dict)
        return user

    def build_chat_prompt(self) -> List[Dict[str, str]]:
        """Builds a structured representation of the prompt."""
        structured_prompt = [{"role": "system", "content": self.config.system}] if self.config.system else []
        structured_prompt.append({"role": "user", "content": self.build_examples()})
        if self.generated_solution:
            structured_prompt.append({"role": "assistant", "content": self.generated_solution})
        return structured_prompt

    def __str__(self) -> str:
        """Returns the complete prompt string representation."""
        prompt = self.config.prompt_template.format(
            system=self.config.system,
            user=self.build_examples(),
            generated_solution=self.generated_solution,
        )
        return prompt


def get_prompt_config(prompt_type: str) -> PromptConfig:
    # reading prompt format from the yaml file, if not running through hydra
    config_path = Path(__file__).parent / f"{prompt_type}.yaml"
    with open(config_path, "rt", encoding="utf-8") as fin:
        # Load the YAML file using OmegaConf
        loaded_config = OmegaConf.load(fin)

        # Create a default PromptConfig and convert it to a DictConfig
        default_config = OmegaConf.create(asdict(PromptConfig()))

        # Merge the default config with the loaded config
        merged_config = OmegaConf.merge(default_config, loaded_config)

        prompt_config = OmegaConf.structured(PromptConfig(_init_nested=True, **merged_config))
        return prompt_config


# this does not come from the config as it's coupled with few shot examples structure
# and so most often will require changes to the code anyway
context_templates = {
    "empty": "",
    "reference_solution": "Reference solution (do not copy it):\n{reference_solution}\n\n",
    "masked_solution": "Reference solution:\n{reference_masked_solution}\n\n",
}
