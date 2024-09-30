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
import random
from dataclasses import asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from nemo_skills.prompt.few_shot_examples import examples_map
from nemo_skills.utils import nested_dataclass

LOG = logging.getLogger(__file__)


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


@nested_dataclass(kw_only=True)
class FewShotExamplesConfig:
    prefix: str = ""
    template: str = ""
    suffix: str = ""

    examples_type: Optional[str] = None

    retrieval_field: Optional[str] = None  # e.g. question, reference_solution, etc.
    retrieval_file: Optional[str] = None  # needs to be provided if retrieval_field is not None
    retrieved_entries: int = 10  # need to set higher than few_shots to filter out exact matches
    retrieved_few_shots: int = 5
    randomize_retrieved_entries: bool = False
    max_retrieved_chars: int = 100000000  # no limit by default
    max_retrieved_chars_field: str = "reference_solution"
    retriever: Optional[Any] = None

    def __post_init__(self):
        """Error checks + building example_dicts and retriever if needed."""
        if self.examples_type is not None and self.retriever is not None:
            raise ValueError("examples_type and retriever cannot be used together")

        if self.retriever is not None:
            return

        if self.retrieval_field is not None:  # building retriever
            if self.retrieval_file is None:
                raise ValueError("retrieval_file must be provided if retrieval_field is not None")
            self.retriever = BM25Retriever(self.retrieval_file, field=self.retrieval_field)
        else:
            if self.retrieval_file is not None:
                raise ValueError("retrieval_field must be provided if retrieval_file is not None")


@nested_dataclass(kw_only=True)
class PromptTemplate:
    text_begin: str
    system_begin: str
    system_end: str
    user_begin: str
    user_end: str
    assistant_begin: str
    assistant_end: str

    # TODO: should stop phrases not be here?
    stop_phrases: List[str]

    # used to execute code within these tags
    code_begin: str = '<llm-code>'
    code_end: str = '</llm-code>'
    # used to extract the code output
    code_output_begin: str = '<llm-code-output>'
    code_output_end: str = '</llm-code-output>'


@nested_dataclass(kw_only=True)
class PromptConfig:
    user: str
    system: str = ""
    template: PromptTemplate = None
    few_shot_examples: FewShotExamplesConfig = field(default_factory=FewShotExamplesConfig)


class Prompt:
    # TODO: multiturn
    FORMAT = "{text_begin}{system_begin}{system}{system_end}{user_begin}{user}{user_end}{assistant_begin}{generation}"

    def __init__(self, config):
        # rebuilding prompt config to make sure post init is called again in
        # case some parameters were manually changed after the config was created
        self.config = PromptConfig(_init_nested=True, **asdict(config))

    def build_filled_example(self, example_dict: Dict[str, Any]) -> str:
        """Builds a filled example string based on the example dictionary."""

        # replacing code/code-output separators in the examples if present
        example_dict = example_dict.copy()
        if 'solution' in example_dict:
            example_dict["solution"] = example_dict["solution"].replace(
                "{code_begin}", self.config.template.code_begin
            )
            example_dict["solution"] = example_dict["solution"].replace("{code_end}", self.config.template.code_end)
            example_dict["solution"] = example_dict["solution"].replace(
                "{code_output_begin}", self.config.template.code_output_begin
            )
            example_dict["solution"] = example_dict["solution"].replace(
                "{code_output_end}", self.config.template.code_output_end
            )

        return self.config.few_shot_examples.template.format(**example_dict, **asdict(self.config.template))

    def build_examples_dict(self, input_dict):
        if self.config.few_shot_examples.examples_type:
            return examples_map[self.config.few_shot_examples.examples_type]

        if self.config.few_shot_examples.retriever is None:
            return []

        example_dicts = self.config.few_shot_examples.retriever.retrieve(
            query=input_dict[self.config.few_shot_examples.retrieval_field],
            top_k=self.config.few_shot_examples.retrieved_entries,
        )
        reference = input_dict[self.config.few_shot_examples.retrieval_field]
        # filtering exact match if it's there
        while example_dicts and example_dicts[0][self.config.few_shot_examples.retrieval_field] == reference:
            example_dicts = example_dicts[1:]

        # removing too long solutions
        example_dicts = [
            example_dict
            for example_dict in example_dicts
            if len(example_dict[self.config.few_shot_examples.max_retrieved_chars_field])
            < self.config.few_shot_examples.max_retrieved_chars
        ]

        if len(example_dicts) < self.config.few_shot_examples.retrieved_few_shots:
            LOG.warning(
                'Too little examples (%d) found for the query "%s"',
                len(example_dicts),
                input_dict[self.config.few_shot_examples.retrieval_field],
            )

        # let's reverse the order to show the most relevant last
        examples = example_dicts[: self.config.few_shot_examples.retrieved_few_shots][::-1]
        if self.config.few_shot_examples.randomize_retrieved_entries:
            random.shuffle(examples)

        return examples

    def build_user_message(self, input_dict: Dict[str, str]) -> str:
        """Builds all examples string concatenated by delimiter."""
        example_dicts = self.build_examples_dict(input_dict)

        filled_examples = "".join([self.build_filled_example(example) for example in example_dicts])
        if not filled_examples:
            examples = ""
        else:
            examples = f"{self.config.few_shot_examples.prefix}{filled_examples}{self.config.few_shot_examples.suffix}"
        user = self.config.user.format(examples=examples, **input_dict)
        return user

    def fill(
        self, input_dict: Dict[str, str], include_generation: bool = False, message_as_prompt: bool = False
    ) -> str | List[dict]:
        """
        Fills the prompt with the input_dict.
        Operates in two modes:
        - If `config.template` is set, it will use the template to fill the prompt, returning a string.
        - If `config.template` is not set, it will assume chat format and return a list of dictionaries.

        Args:
            input_dict: The input dictionary to fill the prompt with.
            include_generation: Whether to include the generation in the prompt.
            message_as_prompt: Whether to return the prompt as a single string or as a list of dictionaries.
                If True, the prompt will be returned as a single string. If False, the prompt will be returned as a list
                of dictionaries. Defaults to False.

        Returns:
            The filled prompt - either a string or a list of dictionaries.
        """
        if include_generation:
            generation = input_dict.get("generation", "")
        else:
            generation = ""

        if self.config.template:
            prompt = self.FORMAT.format(
                system=self.config.system,
                user=self.build_user_message(input_dict),
                generation=generation,
                **asdict(self.config.template),
            )

            return prompt

        else:
            messages = [
                {"role": "system", "content": self.config.system},
                {"role": "user", "content": self.build_user_message(input_dict)},
            ]
            if generation and include_generation:
                messages.append({"role": "assistant", "content": generation})

            if message_as_prompt:
                # return the prompt as a single string
                # if the first message (system) is empty, remove it
                messages = [m["content"] for m in messages]

                if messages[0] == "":
                    messages = messages[1]
                else:
                    # join the messages with newlines
                    messages = "\n\n".join(messages)

            return messages

    @property
    def stop_phrases(self):
        """Returns the stop phrases from the template if it exists, otherwise None."""
        if self.config.template:
            return list(self.config.template.stop_phrases)

        return None

    def __str__(self):
        return str(self.config)


def load_config(config: str, config_dir: str | None = None) -> dict:
    """
    Reads the prompt config/template from the yaml file.

    Args:
        config (str): The location of the prompt config file.
            Can be the full path to a yaml file (if ends with .yaml) or one of the available configs.
            If configs starts with nemo_skills we will look relative to the repo root.
            If not, we will look relative to the config_dir parameter
        config_dir (str): The dir to look for the config file.

    Returns:
        The loaded dictionary.
    """
    if config_dir is None:
        config_dir = str(Path(__file__).parent.absolute() / 'config')

    if config.endswith(".yaml"):
        config_path = Path(config).absolute()
    elif config.startswith("nemo_skills"):
        config_path = Path(__file__).parents[2].absolute() / f"{config}.yaml"
    else:
        config_path = Path(config_dir) / f"{config}.yaml"

    with open(config_path, "rt", encoding="utf-8") as fin:
        return yaml.safe_load(fin)


def get_prompt(
    prompt_config: str,
    prompt_template: str | None = None,
    examples_type: str | None = None,
    config_dir: str | None = None,
    template_dir: str | None = None,
) -> Prompt:
    if template_dir is None:
        template_dir = Path(__file__).parent.absolute() / 'template'
    config = load_config(prompt_config, config_dir)
    if prompt_template is not None:
        template = load_config(prompt_template, template_dir)
        prompt = Prompt(PromptConfig(**config, template=PromptTemplate(**template)))
    else:
        prompt = Prompt(PromptConfig(**config))
    if examples_type is not None:
        prompt.config.few_shot_examples.examples_type = examples_type
    return prompt
