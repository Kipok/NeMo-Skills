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


import abc
import json
import logging
import os
import re
import time

import requests

from nemo_skills.inference.prompt.utils import Prompt

LOG = logging.getLogger(__name__)


def remove_stop_tokens(text: str, stop_phrases: list[str]) -> str:
    """Removes everything after the last stop token."""
    if not stop_phrases:
        return text
    return re.split("|".join([sp.replace('|', '\\|') for sp in stop_phrases]), text, maxsplit=1)[0]


class BaseModel(abc.ABC):
    def __init__(
        self,
        host: str = '127.0.0.1',
        port: str = '5000',
        ssh_server: str | None = None,
        ssh_key_path: str | None = None,
    ):
        self.server_host = host
        self.server_port = port
        self.ssh_server = os.getenv("NEMO_SKILLS_SSH_SERVER", ssh_server)
        self.ssh_key_path = os.getenv("NEMO_SKILLS_SSH_KEY_PATH", ssh_key_path)

        if self.ssh_server and self.ssh_key_path:
            import sshtunnel_requests

            self.requests_lib = sshtunnel_requests.from_url(f"ssh://{self.ssh_server}:22", self.ssh_key_path)
        else:
            self.requests_lib = requests

    def _prepare_request(self, request: dict):
        """Just a small utility to pre-process some of the parameters of request."""
        # temperature of 0 means greedy, but it's not always supported by the server
        # so setting explicit greedy parameters instead
        if request["temperature"] == 0:
            request["temperature"] = 1.0
            request["top_k"] = 1
            request["top_p"] = 1.0

    @abc.abstractmethod
    def generate(
        self,
        prompt: Prompt,
        input_dicts: list[dict],
        tokens_to_generate: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        random_seed: int,
        stop_phrases: list[str],
    ) -> list[dict]:
        pass

    def _send_request(self, request: dict):
        outputs = self.requests_lib.put(
            url="http://{}:{}/generate".format(self.server_host, self.server_port),
            data=json.dumps(request),
            headers={"Content-Type": "application/json"},
        ).json()

        return outputs


class TensorRTLLMModel(BaseModel):
    def generate(
        self,
        prompt: Prompt,
        input_dicts: list[dict],
        tokens_to_generate: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        random_seed: int,
        stop_phrases: list[str],
    ) -> list[dict]:
        string_prompts = [prompt.build_string(input_dict) for input_dict in input_dicts]
        request = {
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "random_seed": random_seed,
            "repetition_penalty": repetition_penalty,
            "stop_words_list": stop_phrases,
        }
        self._prepare_request(request)

        generation_ids = []

        for prompt in string_prompts:
            request["prompt"] = prompt
            generation_ids.append(
                self.requests_lib.put(
                    url="http://{}:{}/start_generation".format(self.server_host, self.server_port),
                    data=json.dumps(request),
                    headers={"Content-Type": "application/json"},
                ).json()
            )

        outputs = [None] * len(generation_ids)
        finished_count = 0
        while finished_count < len(generation_ids):
            time.sleep(0.1)
            for pos, generation_id in enumerate(generation_ids):
                if outputs[pos] is not None:
                    continue
                result = self.requests_lib.put(
                    url="http://{}:{}/get_result".format(self.server_host, self.server_port),
                    data=json.dumps({'generation_id': generation_id}),
                    headers={"Content-Type": "application/json"},
                ).json()
                if result is not None:
                    finished_count += 1
                    outputs[pos] = {'generation': result}

        return outputs


class NemoModel(BaseModel):
    def generate(
        self,
        prompt: Prompt,
        input_dicts: list[dict],
        tokens_to_generate: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        random_seed: int,
        stop_phrases: list[str],
    ) -> list[dict]:
        string_prompts = [prompt.build_string(input_dict) for input_dict in input_dicts]
        request = {
            "sentences": string_prompts,
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "random_seed": random_seed,
            "repetition_penalty": repetition_penalty,
            "end_strings": ["<|endoftext|>"] + stop_phrases,
        }
        self._prepare_request(request)
        generations = self.requests_lib.put(
            url="http://{}:{}/generate".format(self.server_host, self.server_port),
            data=json.dumps(request),
            headers={"Content-Type": "application/json"},
        ).json()
        # we need to remove the original prompt as nemo always returns it
        outputs = [None] * len(generations['sentences'])
        for idx, generation in enumerate(generations['sentences']):
            outputs[idx] = {'generation': generation[len(string_prompts[idx]) :]}
        return outputs


class OpenAIModel(BaseModel):
    def __init__(
        self,
        model,
        api_key=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        from openai import OpenAI

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", api_key)

        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: Prompt,
        input_dicts: list[dict],
        tokens_to_generate: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        random_seed: int,
        stop_phrases: list[str],
        top_k: int = 0,
    ) -> list[dict]:
        if top_k != 0:
            raise ValueError("`top_k` is not supported by OpenAI, please set it to default value `0`.")

        outputs = []
        for input_dict in input_dicts:
            response = self._send_request(
                prompt=prompt,
                input_dict=input_dict,
                tokens_to_generate=tokens_to_generate,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                random_seed=random_seed,
                stop_phrases=stop_phrases,
            )
            outputs.append({'generation': response})
        return outputs

    def _send_request(
        self,
        prompt: Prompt,
        input_dict: dict,
        tokens_to_generate: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        random_seed: int,
        stop_phrases: list[str],
    ):
        messages = prompt.build_structured(input_dict)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=tokens_to_generate,
            presence_penalty=repetition_penalty,
            seed=random_seed,
            stop=stop_phrases,
            messages=messages,
        ).choices[0]
        content = response.message.content

        return content


models = {
    'tensorrt_llm': TensorRTLLMModel,
    'nemo': NemoModel,
    'openai': OpenAIModel,
}


def get_model(server_type, **kwargs):
    """A helper function to make it easier to set server through cmd."""
    model_class = models[server_type.lower()]
    return model_class(**kwargs)
