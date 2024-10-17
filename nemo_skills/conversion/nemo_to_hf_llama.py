# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

# largely copied from https://github.com/NVIDIA/NeMo/blob/main/scripts/nlp_language_modeling/convert_nemo_llama_to_hf.py
# NOTE: Only works with Llama-like models


import os
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path

import torch
import yaml
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils import logging
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--in-path",
        required=True,
        help="Path to .nemo file (can be unpacked)",
    )
    parser.add_argument("--out-path", type=str, default=None, required=True, help="Where to create HF model dir")
    parser.add_argument(
        "--tmp-out-path",
        type=str,
        required=False,
        help="Will save temporary HF weights there (can be used to split large model conversion into 2 parts).",
    )
    parser.add_argument(
        '--hf-model-name', required=True, help="Name of HF model we are converting to (e.g. mistralai/Mistral-7B-v0.1)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Precision of output weights."
        "Defaults to precision of the input nemo weights (model.cfg.trainer.precision)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Load model in cpu only. Useful if the model cannot fit in GPU memory, "
        "but this option makes the conversion script significantly slower.",
    )
    parser.add_argument("--max-shard-size", default="5GB", help="Maximum shard size for the output HF model")
    args = parser.parse_args()
    return args


def create_hf_config(hf_model_name, nemo_config):
    hf_config = AutoConfig.from_pretrained(hf_model_name)
    # SWA; nemo_config.window_size is list [left-bound, right-bound]
    if 'mistral' in hf_model_name.lower():
        hf_config.sliding_window = nemo_config.window_size[0] if 'window_size' in nemo_config else 4096
    hf_config.max_position_embeddings = nemo_config.encoder_seq_length
    hf_config.num_hidden_layers = nemo_config.num_layers
    hf_config.hidden_size = nemo_config.hidden_size
    hf_config.intermediate_size = nemo_config.ffn_hidden_size
    hf_config.num_attention_heads = nemo_config.num_attention_heads
    hf_config.max_position_embeddings = nemo_config.max_position_embeddings
    hf_config.initializer_range = nemo_config.init_method_std
    hf_config.rms_norm_eps = nemo_config.layernorm_epsilon
    hf_config.num_key_value_heads = nemo_config.num_query_groups
    if 'mistral' in hf_model_name.lower():
        if nemo_config.activation == 'fast-swiglu':
            hf_config.activation = 'silu'
        else:
            logging.warning(f"Got unknown activation function {nemo_config.activation}")

    hf_config.rope_theta = nemo_config.get('rotary_base', 10000.0)
    return hf_config


def convert(
    input_nemo_path, output_hf_path, hf_model_name, max_shard_size, tmp_out_path=None, precision=None, cpu_only=False
) -> None:
    """
    Convert NeMo weights to HF weights
    """
    input_nemo_path = os.path.abspath(input_nemo_path)
    output_hf_path = os.path.abspath(output_hf_path)
    dummy_trainer = Trainer(devices=1, accelerator='cpu', strategy=NLPDDPStrategy())
    model_config = OmegaConf.load(str(Path(input_nemo_path) / "model_config.yaml"))
    model_config.tensor_model_parallel_size = 1
    model_config.pipeline_model_parallel_size = 1
    if cpu_only:
        map_location = torch.device('cpu')
        model_config.use_cpu_initialization = True
    else:
        map_location = None

    if precision is None:
        precision = model.cfg.precision
    if precision in [32, "32"]:
        dtype = torch.float32
    elif precision in [16, "16", "16-mixed"]:
        dtype = torch.float16
    elif precision in ["bf16", "bf16-mixed"]:
        dtype = torch.bfloat16
    else:
        logging.warning(f"Precision string {precision} is not recognized, falling back to fp32")
        dtype = torch.float32  # fallback

    checkpoint_in_memory = False
    if tmp_out_path is None or not Path(tmp_out_path).exists():
        checkpoint_in_memory = True

        if cpu_only:
            logging.info("******** Loading model on CPU. This will take a significant amount of time.")
        model = MegatronGPTModel.restore_from(
            input_nemo_path, trainer=dummy_trainer, override_config_path=model_config, map_location=map_location
        )

        param_to_weights = lambda param: param.to(dtype)
        checkpoint = OrderedDict()

        hidden_size = model.cfg.hidden_size
        head_num = model.cfg.num_attention_heads
        num_layers = model.cfg.num_layers
        ffn_hidden_size = model.cfg.ffn_hidden_size
        num_query_groups = model.cfg.get("num_query_groups", head_num)  # different num_query_groups for 70B

        head_size = hidden_size // head_num
        heads_per_group = head_num // num_query_groups
        qkv_total_dim = head_num + 2 * num_query_groups

        weight_base_name = 'model'
        if f'{weight_base_name}.embedding.word_embeddings.weight' not in model.state_dict():
            weight_base_name = 'model.module'
        # Embedding
        embed_weight = model.state_dict()[f'{weight_base_name}.embedding.word_embeddings.weight']
        embed_weights_base_name = f'model.embed_tokens.weight'
        checkpoint[embed_weights_base_name] = param_to_weights(embed_weight)

        for l in range(int(num_layers)):
            print(f"converting layer {l}")

            qkv_weights = model.state_dict()[f'{weight_base_name}.decoder.layers.{l}.self_attention.linear_qkv.weight']
            qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])

            q_slice = torch.cat(
                [
                    torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                    for i in range(num_query_groups)
                ]
            )
            k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
            v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))
            ## Example of slices
            ## 7b: num_query_groups = head_num = 32,
            ## q_slice = [0, 3, 6, 9 , ... 90, 93]
            ## k_slice = [1, 4, 7, 10, ... 91, 94]
            ## v_slice = [2, 5, 8, 11, ... 92, 95]
            ## 70b (with GQA): num_query_groups = 8, head_num = 64
            ## q_slice = [0, 1, .. 6, 7, 10, 11, .. 16, 17, 20, 21, .. 67, 70, ... 76, 77]
            ## k_slice = [8, 18, 28, ... 68, 78]
            ## v_slice = [9, 19, 29, ... 69, 79]

            q_weights_base_name = f'model.layers.{l}.self_attn.q_proj.weight'
            k_weights_base_name = f'model.layers.{l}.self_attn.k_proj.weight'
            v_weights_base_name = f'model.layers.{l}.self_attn.v_proj.weight'

            checkpoint[q_weights_base_name] = param_to_weights(qkv_weights[q_slice].reshape(-1, hidden_size))
            checkpoint[k_weights_base_name] = param_to_weights(qkv_weights[k_slice].reshape(-1, hidden_size))
            checkpoint[v_weights_base_name] = param_to_weights(qkv_weights[v_slice].reshape(-1, hidden_size))

            # attention dense
            o_weight = model.state_dict()[f'{weight_base_name}.decoder.layers.{l}.self_attention.linear_proj.weight']
            o_weight_base_name = f'model.layers.{l}.self_attn.o_proj.weight'
            checkpoint[o_weight_base_name] = param_to_weights(o_weight)

            # mlp
            mlp_weights = model.state_dict()[f'{weight_base_name}.decoder.layers.{l}.mlp.linear_fc1.weight']
            mlp_down_proj_weight = mlp_weights[:ffn_hidden_size, :]
            mlp_gate_proj_weight = mlp_weights[ffn_hidden_size:, :]

            mlp_down_proj_base_name = f'model.layers.{l}.mlp.gate_proj.weight'
            mlp_gate_proj_base_name = f'model.layers.{l}.mlp.up_proj.weight'

            checkpoint[mlp_down_proj_base_name] = param_to_weights(mlp_down_proj_weight)
            checkpoint[mlp_gate_proj_base_name] = param_to_weights(mlp_gate_proj_weight)

            mlp_up_proj_weight = model.state_dict()[f'{weight_base_name}.decoder.layers.{l}.mlp.linear_fc2.weight']
            mlp_up_proj_base_name = f'model.layers.{l}.mlp.down_proj.weight'
            checkpoint[mlp_up_proj_base_name] = param_to_weights(mlp_up_proj_weight)

            # layernorm
            input_ln_weight = model.state_dict()[
                f'{weight_base_name}.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight'
            ]
            input_ln_base_name = f'model.layers.{l}.input_layernorm.weight'
            checkpoint[input_ln_base_name] = param_to_weights(input_ln_weight)

            post_attn_ln_weight = model.state_dict()[
                f'{weight_base_name}.decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight'
            ]
            post_attn_ln_base_name = f'model.layers.{l}.post_attention_layernorm.weight'
            checkpoint[post_attn_ln_base_name] = param_to_weights(post_attn_ln_weight)

            print(f"done layer {l}")

        final_ln_weight = model.state_dict()[f'{weight_base_name}.decoder.final_layernorm.weight']
        final_ln_base_name = f'model.norm.weight'
        checkpoint[final_ln_base_name] = param_to_weights(final_ln_weight)

        output_layer_weight = model.state_dict()[f'{weight_base_name}.output_layer.weight']
        output_layer_base_name = f'lm_head.weight'
        checkpoint[output_layer_base_name] = param_to_weights(output_layer_weight)
        if tmp_out_path is not None:
            torch.save(checkpoint, tmp_out_path)

    if not checkpoint_in_memory:
        checkpoint = torch.load(tmp_out_path, map_location=map_location)

    hf_config = create_hf_config(hf_model_name, model_config)
    model = AutoModelForCausalLM.from_config(hf_config)
    model.load_state_dict(checkpoint)
    model.to(dtype)
    model.save_pretrained(output_hf_path, max_shard_size=max_shard_size)
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_tokenizer.save_pretrained(output_hf_path)
    logging.info(f'HF checkpoint saved to: {output_hf_path}')


if __name__ == '__main__':
    args = get_args()
    convert(
        args.in_path,
        args.out_path,
        args.hf_model_name,
        tmp_out_path=args.tmp_out_path,
        max_shard_size=args.max_shard_size,
        precision=args.precision,
        cpu_only=args.cpu_only,
    )
