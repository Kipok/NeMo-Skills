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
from concurrent.futures import ThreadPoolExecutor
from typing import List

import requests

from nemo_skills.code_execution import (
    CODE_OUTPUT_SEPARATORS,
    CODE_SEPARATORS,
    extract_code_to_execute,
    extract_error_message,
)
from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.code_execution.sandbox import Sandbox
from nemo_skills.utils import python_doc_to_cmd_help

LOG = logging.getLogger(__name__)

# TODO: we should make this more efficient by asynchronously submitting eval
#       requests and getting new prompts in a queue to make sure we can always
#       pack full batches, instead of only resubmitting a small set of prompts
#       that require additional code executions


def remove_stop_tokens(text: str, stop_phrases: List[str]) -> str:
    return re.split("|".join(stop_phrases), text, maxsplit=1)[0]


class BaseModel(abc.ABC):
    """Base model class for handling requests to the inference server.

    Args:
        host: Optional[str] = '127.0.0.1' - Host of the inference server.
        port: Optional[str] = '5000' - Port of the inference server.
        sandbox: Optional[Sandbox] = None - Sandbox for executing code.
            Only required if handle_code_execution is True.
        ssh_server: Optional[str] = None - SSH server for tunneling requests.
            Useful if server is running on slurm cluster to which there is an ssh access
            Can also be specified through SSH_SERVER env var..
        ssh_key_path: Optional[str] = None - Path to the ssh key for tunneling.
            Can also be specified through SSH_KEY_PATH env var.
        max_code_output_characters: Optional[int] = 1000 - Maximum number of characters for code execution output.
        code_execution_timeout: Optional[float] = 10.0 - Timeout for code execution in seconds.
        max_code_executions: Optional[int] = 3 - Maximum number of code executions per generation.
        stop_on_code_error: Optional[bool] = True - Whether to stop generation if code execution fails.
        handle_code_execution: Optional[bool] = True - Whether to handle code execution in this class
            or make a single call to the server. If set to False, the server needs to have special logic
            for communicating with the sandbox.
    """

    def __init__(
        self,
        host='127.0.0.1',
        port='5000',
        sandbox=None,
        ssh_server=None,
        ssh_key_path=None,
        max_code_output_characters=1000,
        code_execution_timeout=10.0,
        max_code_executions=3,
        stop_on_code_error=True,
        handle_code_execution=True,
    ):
        self.server_host = host
        self.server_port = port
        self.ssh_server = os.getenv("SSH_SERVER", ssh_server)
        self.ssh_key_path = os.getenv("SSH_KEY_PATH", ssh_key_path)
        self.max_code_output_characters = max_code_output_characters
        self.code_execution_timeout = code_execution_timeout
        self.max_code_executions = max_code_executions
        self.handle_code_execution = handle_code_execution
        self.stop_on_code_error = stop_on_code_error
        if self.handle_code_execution and sandbox is None:
            raise ValueError("Sandbox is required for handling code execution")
        if not self.handle_code_execution and not stop_on_code_error:
            # TODO: same warning for other ignored parameters
            LOG.warning("When code execution is not handled here, stop_on_code_error is ignored.")
        self.sandbox = sandbox

    @abc.abstractmethod
    def _single_call(
        self,
        prompts,
        tokens_to_generate,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        random_seed,
        stop_phrases: List[str],
    ):
        pass

    def __call__(
        self,
        prompts,
        tokens_to_generate,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        random_seed,
        stop_phrases: List[str],
    ):
        # temperature of 0 means greedy, but it's not always supported by the server
        # so setting explicit greedy parameters instead
        if temperature == 0:
            temperature = 1.0
            top_k = 1
            top_p = 1.0

        if self.handle_code_execution:
            full_stop_phrases = stop_phrases + [CODE_SEPARATORS[-1]]
        else:
            full_stop_phrases = stop_phrases
        # prompts are added later
        request = {
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "random_seed": random_seed,
            "repetition_penalty": repetition_penalty,
            "stop_phrases": full_stop_phrases,
        }

        # if code execution is handled by the inference framework, we only need to make a single call
        # and then apply postprocessing to extract errors from the output
        if not self.handle_code_execution:
            request["prompts"] = prompts
            outputs = self._single_call(**request)
            outputs = [
                {
                    'generated_solution': remove_stop_tokens(output, stop_phrases),
                    'predicted_answer': extract_answer(output),
                    'error_message': extract_error_message(output),
                }
                for output in outputs
            ]
            return outputs

        # making requests to LLM and iterating on prompts that produce code tokens
        # after executing code and getting result back into the prompt
        new_outputs = [
            {
                'full_prompt': prompts[idx],
                'result': None,
                'error_message': Sandbox.NOT_EXECUTED,
                'session_id': None,
            }
            for idx in range(len(prompts))
        ]
        remaining_ids = list(range(len(new_outputs)))
        num_executions = 0
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            while len(remaining_ids) > 0:
                num_executions += 1
                request["prompts"] = [new_outputs[idx]['full_prompt'] for idx in remaining_ids]

                outputs = self._single_call(**request)
                new_ids = []
                # checking if any of the outputs need code execution and submitting requests in parallel
                futures = [None] * len(prompts)
                for idx, output in zip(remaining_ids, outputs):
                    if output.endswith(CODE_SEPARATORS[-1]):
                        futures[idx] = executor.submit(
                            self.sandbox.execute_code,
                            generated_code=extract_code_to_execute(output),
                            timeout=self.code_execution_timeout,
                            max_output_characters=self.max_code_output_characters,
                            session_id=new_outputs[idx]['session_id'],
                        )
                for idx, output in zip(remaining_ids, outputs):
                    new_outputs[idx]['full_prompt'] += output
                    if output.endswith(CODE_SEPARATORS[-1]):
                        result, new_outputs[idx]['session_id'] = futures[idx].result()
                        # for now if there is any error or no output, we stop generation
                        # might revise in the future to allow LLM to recover
                        if result['error_message']:
                            new_outputs[idx]['error_message'] = result['error_message']
                            if self.stop_on_code_error:
                                continue
                        else:
                            new_outputs[idx]['error_message'] = ''

                        # adding code output to the prompt
                        code_output = (
                            f'\n{CODE_OUTPUT_SEPARATORS[0]}\n{result["result"]}\n{CODE_OUTPUT_SEPARATORS[1]}\n'
                        )
                        new_outputs[idx]['full_prompt'] += code_output
                        # setting a limit on max code executions to speed things up
                        # (sometimes keeps repeating the same sequence forever)
                        if num_executions >= self.max_code_executions:
                            new_outputs[idx]['error_message'] = "Max code executions reached"
                        else:
                            new_ids.append(idx)
                remaining_ids = new_ids

        # removing original prompt and stop tokens from the end of the generated text
        outputs = []
        for original_prompt, output in zip(prompts, new_outputs):
            if output['session_id'] is not None:
                self.sandbox.clear_session(output['session_id'])
            generated_solution = remove_stop_tokens(output['full_prompt'][len(original_prompt) :], stop_phrases)
            outputs.append(
                {
                    'generated_solution': generated_solution,
                    'predicted_answer': extract_answer(generated_solution),
                    'error_message': output['error_message'],
                }
            )
        return outputs

    def _send_request(self, request):
        if self.ssh_server and self.ssh_key_path:
            import sshtunnel_requests

            sshtunnel_request = sshtunnel_requests.from_url(f"ssh://{self.ssh_server}:22", self.ssh_key_path)
            outputs = sshtunnel_request.put(
                url="http://{}:{}/generate".format(self.server_host, self.server_port),
                data=json.dumps(request),
                headers={"Content-Type": "application/json"},
            ).json()
        else:
            outputs = requests.put(
                url="http://{}:{}/generate".format(self.server_host, self.server_port),
                data=json.dumps(request),
                headers={"Content-Type": "application/json"},
            ).json()

        return outputs


class TensorRTLLMModel(BaseModel):
    def _single_call(
        self,
        prompts,
        tokens_to_generate,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        random_seed,
        stop_phrases: List[str],
    ):
        request = {
            "prompts": prompts,
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "random_seed": random_seed,
            "repetition_penalty": repetition_penalty,
            "stop_words_list": stop_phrases,
        }
        return self._send_request(request)


class NemoModel(BaseModel):
    def __init__(
        self,
        **kwargs,
    ):
        # nemo inference handles code execution directly
        kwargs['handle_code_execution'] = False
        super().__init__(**kwargs)

    def _single_call(
        self,
        prompts,
        tokens_to_generate,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        random_seed,
        stop_phrases: List[str],
    ):
        request = {
            "sentences": prompts,
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "random_seed": random_seed,
            "repetition_penalty": repetition_penalty,
            "end_strings": ["<|endoftext|>"] + stop_phrases,
        }
        outputs = self._send_request(request)
        outputs = outputs['sentences']
        # always returns full prompt, so we need to remove the original prompt
        for idx, output in enumerate(outputs):
            outputs[idx] = output[len(prompts[idx]) :]
        return outputs


models = {
    'tensorrt_llm': TensorRTLLMModel,
    'nemo': NemoModel,
}


def get_model(server_type, **kwargs):
    """A helper function to make it easier to set server through cmd."""
    model_class = models[server_type.lower()]
    return model_class(**kwargs)


def server_params():
    """Returns server documentation (to include in cmd help)."""
    prefix = f'\n        server_type: str = MISSING - Choices: {list(models.keys())}'
    return python_doc_to_cmd_help(BaseModel, docs_prefix=prefix, arg_prefix="server.")
