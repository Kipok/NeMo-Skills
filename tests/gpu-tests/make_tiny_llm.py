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

# adapted from https://huggingface.co/stas/tiny-random-llama-2/blob/main/make_tiny_model.py

import argparse

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description="Create a tiny model for testing.")
parser.add_argument("--model_type", type=str, required=True, choices=("qwen", "llama", "mistral_emb"))
args = parser.parse_args()

if args.model_type == 'qwen':
    model_name = "Qwen/Qwen2.5-Math-7B"
    output_dir = "/tmp/nemo-skills-tests/qwen/tiny-model-hf"
    hidden_dim = 56
    head_dim = 2
elif args.model_type == 'mistral_emb':
    model_name = "intfloat/e5-mistral-7b-instruct"
    output_dir = "/tmp/nemo-skills-tests/mistral_emb/tiny-model-hf"
    hidden_dim = 128
    head_dim = 64
else:
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    output_dir = "/tmp/nemo-skills-tests/llama/tiny-model-hf"
    hidden_dim = 64
    head_dim = 2

config = AutoConfig.from_pretrained(model_name)
config.update(
    dict(
        hidden_size=hidden_dim,
        head_dim=head_dim,
        intermediate_size=hidden_dim,
        num_hidden_layers=2,
        max_position_embeddings=256,
    )
)
print("new config", config)

if args.model_type == 'mistral_emb':
    tiny_model = AutoModel.from_config(config)
else:
    # create a tiny random model
    tiny_model = AutoModelForCausalLM.from_config(config)

print(f"num of params {tiny_model.num_parameters()}")

# shrink it more and save
tiny_model.bfloat16()  # half-size
tiny_model.save_pretrained(output_dir)

hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_tokenizer.save_pretrained(output_dir)
