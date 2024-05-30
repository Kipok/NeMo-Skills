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

# adapted from https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/examples/run.py


import json
import logging
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import numpy as np
import tensorrt_llm
import torch
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from mpi4py import MPI
from tensorrt_llm.runtime import ModelRunnerCpp
from transformers import AutoTokenizer, T5Tokenizer


# keeping it here to make this file self-contained. This is duplicated from model.py
def remove_stop_tokens(text: str, stop_phrases: List[str]) -> str:
    """Removes everything after the last stop token."""
    return re.split("|".join([sp.replace('|', '\\|') for sp in stop_phrases]), text, maxsplit=1)[0]


class CustomSentencePieceTokenizer(T5Tokenizer):
    """
    Adapted from https://github.com/NVIDIA/Megatron-LM/blob/db3a3f79d1cda60ea4b3db0ceffcf20c5760e11d/examples/inference/trtllm_text_generation.py
    """

    def __init__(self, model):
        super().__init__(model, extra_ids=0, bos_token="<s>", pad_token="<pad>")

    def encode(self, text, add_special_tokens: bool = True, **kwargs):
        return torch.Tensor(self.sp_model.encode_as_ids(text))

    def batch_encode_plus(self, batch_text_or_text_pairs, add_special_tokens: bool = True, **kwargs):
        return {'input_ids': self.sp_model.encode_as_ids(batch_text_or_text_pairs)}

    def batch_decode(self, sequences, skip_special_tokens: bool = False, **kwargs):
        if isinstance(sequences, np.ndarray) or torch.is_tensor(sequences):
            sequences = sequences.tolist()
        return self.sp_model.decode(sequences)

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs):
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        return self.sp_model.decode([token_ids])[0]


class TrtServerGenerate(Resource):
    def __init__(self, model):
        self.model = model
        self.comm = MPI.COMM_WORLD

    def generate(
        self,
        prompts,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        random_seed,
        stop_words_list,
    ):
        output = self.model.forward(
            prompts,
            max_output_token=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            random_seed=random_seed,
            stop_words_list=stop_words_list,
        )
        return output

    def put(self):
        logging.info("request IP: " + str(request.remote_addr))
        logging.info(json.dumps(request.get_json()))

        input_request = request.get_json()

        tokens_to_generate = input_request.get("tokens_to_generate", 64)
        temperature = input_request.get("temperature", 1.0)
        top_k = input_request.get("top_k", 0)
        top_p = input_request.get("top_p", 1.0)
        repetition_penalty = input_request.get("repetition_penalty", 1.2)
        stop_words_list = input_request.get("stop_words_list")
        random_seed = input_request.get("random_seed", 0)
        prompts = input_request["prompts"]

        data = dict(
            prompts=prompts,
            max_new_tokens=tokens_to_generate,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            random_seed=random_seed,
            stop_words_list=stop_words_list,
        )
        self.comm.Barrier()
        data = self.comm.bcast(data, root=0)

        out = self.generate(**data)
        return jsonify(out)


def parse_input(input_texts: str, tokenizer):
    batch_input_ids = [
        tokenizer.encode(
            input_text,
            add_special_tokens=False,
        )
        for input_text in input_texts
    ]
    batch_input_ids = [torch.tensor(x, dtype=torch.int32).unsqueeze(0) for x in batch_input_ids]
    input_lengths = [x.size(1) for x in batch_input_ids]

    return batch_input_ids, input_lengths


def get_output_single(output_ids, input_length, max_output_len, tokenizer, eos_token):
    output_begin = input_length
    output_end = input_length + max_output_len
    outputs = output_ids[output_begin:output_end]
    eos_ids = (outputs == eos_token).nonzero(as_tuple=True)[-1]
    if len(eos_ids) > 0:
        outputs = outputs[: eos_ids[0]]
    outputs = outputs.tolist()
    outputs = [elem if elem < tokenizer.vocab_size else tokenizer.vocab_size - 1 for elem in outputs]
    outputs = [elem if 0 <= elem else 0 for elem in outputs]
    return tokenizer.decode(outputs)


def get_output(output_ids, input_lengths, max_output_len, tokenizer, eos_token):
    num_beams = output_ids.size(1)
    assert num_beams == 1
    output_texts = []
    for idx, input_len in enumerate(input_lengths):
        output_texts.append(get_output_single(output_ids[idx, 0], input_len, max_output_len, tokenizer, eos_token))
    return output_texts


def load_tokenizer(tokenizer_dir: str, model_name: str):
    if model_name == 'gpt-next':
        tokenizer = CustomSentencePieceTokenizer(str(Path(tokenizer_dir) / 'tokenizer.model'))
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            tokenizer_type=model_name,
            legacy=False,
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id


def read_model_name(config):
    name = config['pretrained_config']['architecture'].lower()
    name_map = {
        'MistralForCausalLM'.lower(): 'mistral',
        'LlamaForCausalLM'.lower(): 'llama',
        'MixtralForCausalLM'.lower(): 'mixtral',
        'GPTForCausalLM'.lower(): 'gpt-next',
    }
    return name_map[name]


class TensorRTLLM:
    def __init__(self, model_path: str):
        with open(Path(model_path) / "config.json", 'r') as f:
            config = json.load(f)
        self.tokenizer, self.pad_id, self.end_id = load_tokenizer(
            tokenizer_dir=model_path, model_name=read_model_name(config)
        )
        self.runner = ModelRunnerCpp.from_dir(
            engine_dir=model_path,
            rank=tensorrt_llm.mpi_rank(),
            max_beam_width=config['build_config']['max_beam_width'],
            max_input_len=config['build_config']['max_input_len'],
            max_output_len=config['build_config']['max_output_len'],
            max_batch_size=config['build_config']['max_batch_size'],
        )

    def get_output(
        self,
        # input_text,
        batch_input_ids,
        input_lengths,
        max_output_token,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
        random_seed,
        stop_words_list,
    ):
        # TODO: return dictionary with a proper error reporting
        sampling_config = SamplingConfig(
            end_id=self.end_id,
            pad_id=self.pad_id,
            max_new_tokens=max_output_token,
            temperature=temperature,
        )
        # sampling_config.temperature = [temperature]
        # # sampling_config.top_k = [top_k]
        # # sampling_config.top_p = [top_p]
        # sampling_config.repetition_penalty = [repetition_penalty]
        # print(random_seed)
        # sampling_config.random_seed = [random_seed]
        # stop words in trtllm are supported on the token-level only and this representation is not unique
        # so instead of passing in all tokenizations (is that even possible?) of each phrase, we will
        # instead stream outputs and detokenize them to check for stop words

        try:
            output_generator = self.executor.generate(
                prompt=batch_input_ids[0],
                streaming=True,
                sampling_config=sampling_config,
            )
            print(output_generator)
            # checking the last 20 tokens for stop words
            num_tokens_to_check = 20
            matching_stop_word = None
            for idx, output in enumerate(output_generator, 1):
                # checking every half of the required tokens to have overlapping checks
                if idx < num_tokens_to_check - 1 or idx % (num_tokens_to_check // 2) != 0:
                    continue
                seq_length = output['sequence_lengths']
                generation_suffix = output['output_ids'][0, 0, seq_length[0] - num_tokens_to_check : seq_length[0]]
                output_string = get_output_single(
                    generation_suffix, 0, num_tokens_to_check, self.tokenizer, self.end_id
                )
                for stop_word in stop_words_list:
                    if stop_word in output_string:
                        matching_stop_word = stop_word
                        break

                if matching_stop_word is not None:
                    break

            output = get_output(output['output_ids'], input_lengths, seq_length[0], self.tokenizer, self.end_id)[0]
            if matching_stop_word is not None:
                output = remove_stop_tokens(output, stop_words_list)
                # adding it back, since we only need to remove what's *after* the stop phrase
                output += matching_stop_word
        except RuntimeError as e:
            logging.error("RuntimeError: %s", e)
            output = f"RuntimeError: {e}"

        return output

    @torch.no_grad()
    def forward(
        self,
        input_texts,
        max_output_token,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
        random_seed,
        stop_words_list,
    ):
        # TODO: remove batch dimension since it's not needed anymore?
        outputs = []
        for input_text in input_texts:
            # hashing based on all parameters so that we do not execute
            # the same requests and can identify futures later
            batch_input_ids, input_lengths = parse_input([input_text], self.tokenizer)
            outputs.append(
                self.get_output(
                    batch_input_ids,
                    input_lengths,
                    max_output_token,
                    top_k,
                    top_p,
                    temperature,
                    repetition_penalty,
                    random_seed,
                    stop_words_list,
                )
            )

        return outputs


class WrapperServer:
    def __init__(self, model_path: str):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        self.model = TensorRTLLM(model_path=model_path)

        if self.rank == 0:
            self.app = Flask(__file__, static_url_path="")
            api = Api(self.app)
            api.add_resource(TrtServerGenerate, "/generate", resource_class_args=[self.model])

    def run(self, url, port=5000):
        if self.rank == 0:
            self.app.run(url, threaded=True, port=port, debug=False)
        else:
            self.worker_loop()

    def worker_loop(self):
        server = TrtServerGenerate(self.model)
        while True:
            self.comm.Barrier()
            data = None
            data = self.comm.bcast(data, root=0)
            server.generate(**data)


if __name__ == "__main__":
    # TODO: can we reuse normal logger here?
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    server = WrapperServer(model_path=args.model_path)
    server.run(args.host, args.port)
