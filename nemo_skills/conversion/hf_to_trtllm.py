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

# copied from https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llama/convert_checkpoint.py

import argparse
import copy
import functools
import json
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import safetensors
import tensorrt_llm
import torch
import torch.nn as nn
from tensorrt_llm._utils import pad_vocab_size
from tensorrt_llm.layers import MoeConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.llama.weight import (
    load_from_fp8_llama,
    load_from_gptq_llama,
    load_from_hf_checkpoint,
    load_from_meta_llama,
)
from tensorrt_llm.models.modeling_utils import PretrainedConfig
from tensorrt_llm.runtime.lora_manager import LoraConfig
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.pytorch_utils import Conv1D

from datasets import load_dataset

try:
    from transformers import MixtralForCausalLM
except ImportError:
    MixtralForCausalLM = None

try:
    from transformers import LlavaConfig, LlavaForConditionalGeneration
except ImportError:
    pass


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--meta_ckpt_dir', type=str, default=None)

    parser.add_argument('--tp_size', type=int, default=1, help='N-way tensor parallelism size')
    parser.add_argument('--pp_size', type=int, default=1, help='N-way pipeline parallelism size')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--n_positions', type=int, default=2048)
    parser.add_argument('--n_layer', type=int, default=32)
    parser.add_argument('--n_head', type=int, default=32)
    parser.add_argument('--n_kv_head', type=int, default=None)
    parser.add_argument('--n_embd', type=int, default=4096)
    parser.add_argument('--inter_size', type=int, default=11008)
    parser.add_argument('--rms_norm_eps', type=float, default=1e-06)

    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.' 'See --weight_only_precision to set the precision',
    )
    parser.add_argument(
        '--disable_weight_only_quant_plugin',
        default=False,
        action="store_true",
        help='By default, using plugin implementation for weight quantization. Enabling disable_weight_only_quant_plugin flag will use ootb implementation instead of plugin.'
        'You must also use --use_weight_only for that argument to have an impact.',
    )
    parser.add_argument(
        '--weight_only_precision',
        const='int8',
        type=str,
        nargs='?',
        default='int8',
        choices=['int8', 'int4', 'int4_gptq'],
        help='Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.',
    )
    parser.add_argument(
        "--smoothquant",
        "-sq",
        type=float,
        default=None,
        help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
        " to Smoothquant the model, and output int8 weights."
        " A good first try is 0.5. Must be in [0, 1]",
    )
    parser.add_argument(
        '--per_channel',
        action="store_true",
        default=False,
        help='By default, we use a single static scaling factor for the GEMM\'s result. '
        'per_channel instead uses a different static scaling factor for each channel. '
        'The latter is usually more accurate, but a little slower.',
    )
    parser.add_argument(
        '--per_token',
        action="store_true",
        default=False,
        help='By default, we use a single static scaling factor to scale activations in the int8 range. '
        'per_token chooses at run time, and for each token, a custom scaling factor. '
        'The latter is usually more accurate, but a little slower.',
    )
    parser.add_argument(
        '--int8_kv_cache',
        default=False,
        action="store_true",
        help='By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV',
    )
    parser.add_argument(
        '--ammo_quant_ckpt_path', type=str, default=None, help='Path of a quantized model checkpoint in .npz format'
    )

    parser.add_argument(
        '--per_group',
        default=False,
        action="store_true",
        help='By default, we use a single static scaling factor to scale weights in the int4 range. '
        'per_group chooses at run time, and for each group, a custom scaling factor. '
        'The flag is built for GPTQ/AWQ quantization.',
    )

    parser.add_argument(
        '--enable_fp8',
        default=False,
        action='store_true',
        help='Use FP8 Linear layer for Attention QKV/Dense and MLP.',
    )
    parser.add_argument(
        '--fp8_kv_cache',
        default=False,
        action="store_true",
        help='By default, we use dtype for KV cache. fp8_kv_cache chooses int8 ' 'quantization for KV',
    )
    parser.add_argument('--load_by_shard', action='store_true', help='Load a pretrained model shard-by-shard.')
    parser.add_argument('--hidden_act', type=str, default='silu')

    parser.add_argument('--rotary_base', type=float, default=10000.0)
    parser.add_argument('--rotary_scaling', nargs=2, type=str, default=None)

    parser.add_argument('--group_size', type=int, default=128, help='Group size used in GPTQ/AWQ quantization.')

    parser.add_argument("--storage-type", "-t", type=str, default="fp32", choices=["fp32", "fp16"])
    parser.add_argument(
        "--dataset-cache-dir", type=str, default=None, help="cache dir to load the hugging face dataset"
    )
    parser.add_argument("--load-model-on-cpu", action="store_true")
    parser.add_argument("--convert-model-on-cpu", action="store_true")
    parser.add_argument(
        '--use_parallel_embedding',
        action="store_true",
        default=False,
        help='By default embedding parallelism is disabled. By setting this flag, embedding parallelism is enabled',
    )
    parser.add_argument(
        '--embedding_sharding_dim',
        type=int,
        default=0,
        choices=[0, 1],
        help='By default the embedding lookup table is sharded along vocab dimension (embedding_sharding_dim=0). '
        'To shard it along hidden dimension, set embedding_sharding_dim=1'
        'Note: embedding sharing is only enabled when embedding_sharding_dim = 0',
    )
    parser.add_argument(
        '--use_embedding_sharing',
        action="store_true",
        default=False,
        help='Try to reduce the engine size by sharing the embedding lookup table between two layers.'
        'Note: the flag might not take effect when the criteria are not met.',
    )
    parser.add_argument('--use_prompt_tuning', action="store_true", default=False)
    parser.add_argument(
        '--output_dir', type=str, default='tllm_checkpoint', help='The path to save the TensorRT-LLM checkpoint'
    )
    parser.add_argument(
        '--workers', type=int, default=1, help='The number of workers for converting checkpoint in parallel'
    )
    parser.add_argument(
        '--moe_num_experts', default=0, type=int, help='Specify the number of experts to use for MOE layers'
    )
    parser.add_argument(
        '--moe_top_k',
        default=0,
        type=int,
        help='Specify the top_k value to use for MOE layers. Default to 1 if --moe_num_experts is set',
    )
    parser.add_argument(
        '--moe_tp_mode',
        default=MoeConfig.ParallelismMode.TENSOR_PARALLEL,
        type=int,
        help='Controls how to distribute experts in TP. Check layers/moe.py for accepted values',
    )
    parser.add_argument(
        '--moe_renorm_mode',
        default=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
        type=int,
        help='Controls renormalization after gate logits. Check layers/moe.py for accepted values',
    )
    parser.add_argument(
        '--enable_pos_shift', default=False, action='store_true', help='Enable position shift for streamingllm method'
    )
    parser.add_argument(
        '--dense_context_fmha',
        default=False,
        action='store_true',
        help='Enable dense fmha in context phase, otherwise sliding window attention.'
        'If dense_context_fmha=False, the sliding window size is the max attention window size.',
    )
    parser.add_argument('--hf_lora_dir', type=str, default=None)
    parser.add_argument(
        '--lora_target_modules',
        nargs='+',
        default=None,
        choices=[
            "attn_qkv",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_dense",
            "mlp_h_to_4h",
            "mlp_gate",
            "mlp_4h_to_h",
        ],
        help="Add lora in which modules. Only be activated when use_lora_plugin is enabled.",
    )
    parser.add_argument(
        '--max_lora_rank',
        type=int,
        default=64,
        help='maximum lora rank for different lora modules. '
        'It is used to compute the workspace size of lora plugin.',
    )
    args = parser.parse_args()
    return args


def generate_int8(weights, act_range, is_qkv=False, multi_query_mode=False):
    """
    This function has two purposes:
     - compute quantized weights, scaled either per-tensor or per-column
     - compute scaling factors

     Depending on the GEMM API (CUTLASS/CUBLAS) the required scaling factors differ.
     CUTLASS uses two sets of scaling factors. One for the activation X, one for the weight W.
     CUBLAS only has one (we can't do per-row scaling). So we must provide pre-multiplied scaling factor.

     Here is the list of what we need (T means per-tensor, C per-column):
       - scale_x_orig_quant puts fp activation into the quantized range (i.e. [-128, 127], for int8). Used before the GEMM. (T)
       - scale_y_quant_orig puts quantized activation into the fp range. Used if the GEMM outputs int8. (T)
       - scale_w_quant_orig puts weights from quant range to fp range (used with CUTLASS) (T, C)
       - scale_y_accum_quant puts the GEMM result (XW) from accumulation range (int32)
         to quant range (int8) (used for CUBLAS) (T, C)

     Note that we don't do anything special about row-parallel GEMM. Theoretically, we could have per-GPU scaling factors too,
     but then the model would change depending on the number of GPUs used.

     For QKV projection, the behavior is special. Even if we have a single matrix to perform QKV projection, we consider it
     as three different matrices: Q, K, and V. So per-tensor actually means one scaling factor for each Q, K and V.
     For our GEMM implementation to respect this behavior, we use per-column mode and replicate values along columns.
    """
    weights = weights.detach().cpu().numpy()

    # compute weight scaling factors for fp->int8 and int8->fp
    if is_qkv and not multi_query_mode:
        scale_w_orig_quant_t = 127.0 / act_range["w"].reshape(3, -1).max(dim=-1, keepdims=True)[0].cpu().numpy()
        scale_w_orig_quant_c = 127.0 / act_range["w"].reshape(3, -1).cpu().numpy()
    elif is_qkv and multi_query_mode:
        hidden_dim = weights.shape[0]
        local_dim = act_range["w"].shape[0]
        kv_dim = (local_dim - hidden_dim) // 2
        scale_w_q = act_range["w"][0:hidden_dim]
        scale_w_k = act_range["w"][hidden_dim : hidden_dim + kv_dim]
        scale_w_v = act_range["w"][-kv_dim:]

        scale_w_qkv_t = torch.concat(
            [
                scale_w_q.max(dim=0, keepdim=True)[0],
                scale_w_k.max(dim=0, keepdim=True)[0],
                scale_w_v.max(dim=0, keepdim=True)[0],
            ]
        )

        scale_w_orig_quant_t = 127.0 / scale_w_qkv_t.cpu().numpy()
        scale_w_orig_quant_c = 127.0 / act_range["w"].cpu().numpy()
    else:
        scale_w_orig_quant_t = 127.0 / act_range["w"].max().cpu().numpy()
        scale_w_orig_quant_c = 127.0 / act_range["w"].cpu().numpy()
    scale_w_quant_orig_t = 1.0 / scale_w_orig_quant_t
    scale_w_quant_orig_c = 1.0 / scale_w_orig_quant_c

    scale_w_orig_quant_c = scale_w_orig_quant_c.astype(np.float32)
    scale_w_orig_quant_t = scale_w_orig_quant_t.astype(np.float32)

    # compute the rest of needed scaling factors
    scale_x_orig_quant_t = np.array(127.0 / act_range["x"].max().item())
    scale_y_orig_quant_t = np.array(127.0 / act_range["y"].max().item())
    scale_y_quant_orig_t = np.array(act_range["y"].max().item() / 127.0)
    scale_y_accum_quant_t = scale_y_orig_quant_t / (scale_x_orig_quant_t * scale_w_orig_quant_t)
    scale_y_accum_quant_c = scale_y_orig_quant_t / (scale_x_orig_quant_t * scale_w_orig_quant_c)
    if is_qkv and not multi_query_mode:
        scale_y_accum_quant_t = np.broadcast_to(scale_y_accum_quant_t, scale_w_orig_quant_c.shape)
        scale_w_quant_orig_t = np.broadcast_to(scale_w_quant_orig_t, scale_w_orig_quant_c.shape)
    if is_qkv and multi_query_mode:
        scale_q_y_accum_t = np.broadcast_to(scale_y_accum_quant_t[0], scale_w_q.shape)
        scale_k_y_accum_t = np.broadcast_to(scale_y_accum_quant_t[1], scale_w_k.shape)
        scale_v_y_accum_t = np.broadcast_to(scale_y_accum_quant_t[2], scale_w_v.shape)
        scale_y_accum_quant_t = np.concatenate([scale_q_y_accum_t, scale_k_y_accum_t, scale_v_y_accum_t])
        scale_w_quant_orig_t = np.concatenate(
            [
                np.broadcast_to(scale_w_quant_orig_t[0], scale_w_q.shape),
                np.broadcast_to(scale_w_quant_orig_t[1], scale_w_k.shape),
                np.broadcast_to(scale_w_quant_orig_t[2], scale_w_v.shape),
            ]
        )

    to_i8 = lambda x: x.round().clip(-127, 127).astype(np.int8)

    if is_qkv and multi_query_mode:
        weight_int8 = to_i8(weights / scale_w_quant_orig_t)
    else:
        weight_int8 = to_i8(weights * scale_w_orig_quant_t)
    return {
        "weight.int8": weight_int8,
        "weight.int8.col": to_i8(weights * scale_w_orig_quant_c),
        "scale_x_orig_quant": scale_x_orig_quant_t.astype(np.float32),
        "scale_w_quant_orig": scale_w_quant_orig_t.astype(np.float32),
        "scale_w_quant_orig.col": scale_w_quant_orig_c.astype(np.float32),
        "scale_y_accum_quant": scale_y_accum_quant_t.astype(np.float32),
        "scale_y_accum_quant.col": scale_y_accum_quant_c.astype(np.float32),
        "scale_y_quant_orig": scale_y_quant_orig_t.astype(np.float32),
    }


@torch.no_grad()
def apply_smoothing(
    scales, gemm_weights, layernorm_weights=None, layernorm_bias=None, dtype=torch.float32, layernorm_1p=False
):
    if not isinstance(gemm_weights, list):
        gemm_weights = [gemm_weights]

    if layernorm_weights is not None:
        assert layernorm_weights.numel() == scales.numel()
        layernorm_weights.div_(scales).to(dtype)
    if layernorm_bias is not None:
        assert layernorm_bias.numel() == scales.numel()
        layernorm_bias.div_(scales).to(dtype)
    if layernorm_1p:
        layernorm_weights += (1 / scales) - 1

    for gemm in gemm_weights:
        gemm.mul_(scales.view(1, -1)).to(dtype)


@torch.no_grad()
def smooth_gemm(gemm_weights, act_scales, layernorm_weights=None, layernorm_bias=None, alpha=0.5, weight_scales=None):
    if not isinstance(gemm_weights, list):
        gemm_weights = [gemm_weights]
    orig_dtype = gemm_weights[0].dtype

    for gemm in gemm_weights:
        # gemm_weights are expected to be transposed
        assert gemm.shape[1] == act_scales.numel()

    if weight_scales is None:
        weight_scales = torch.cat([gemm.abs().max(dim=0, keepdim=True)[0] for gemm in gemm_weights], dim=0)
        weight_scales = weight_scales.max(dim=0)[0]
    weight_scales.to(float).clamp(min=1e-5)
    scales = (act_scales.to(gemm_weights[0].device).to(float).pow(alpha) / weight_scales.pow(1 - alpha)).clamp(
        min=1e-5
    )

    apply_smoothing(scales, gemm_weights, layernorm_weights, layernorm_bias, orig_dtype)

    return scales


@torch.no_grad()
def smooth_gemm_fc1_gate(
    fc1_weights, gate_weights, act_scales, layernorm_weights=None, layernorm_bias=None, alpha=0.5, weight_scales=None
):
    gemm_weights = []
    if not isinstance(fc1_weights, list):
        fc1_weights = [fc1_weights]
    if not isinstance(gate_weights, list):
        gate_weights = [gate_weights]

    for i in range(len(fc1_weights)):
        gemm_weight = torch.cat([fc1_weights[i], gate_weights[i]], dim=0)
        gemm_weights.append(gemm_weight)

    orig_dtype = gemm_weights[0].dtype

    for gemm in gemm_weights:
        # gemm_weights are expected to be transposed
        assert gemm.shape[1] == act_scales.numel()

    if weight_scales is None:
        weight_scales = torch.cat([gemm.abs().max(dim=0, keepdim=True)[0] for gemm in gemm_weights], dim=0)
        weight_scales = weight_scales.max(dim=0)[0]
    weight_scales.to(float).clamp(min=1e-5)
    scales = (act_scales.to(gemm_weights[0].device).to(float).pow(alpha) / weight_scales.pow(1 - alpha)).clamp(
        min=1e-5
    )

    apply_smoothing(scales, fc1_weights + gate_weights, layernorm_weights, layernorm_bias, orig_dtype)

    return scales


@torch.no_grad()
def smooth_llama_model(model, scales, alpha, llama_qkv_para, llama_smoother):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        if not isinstance(module, LlamaDecoderLayer):
            continue
        # qkv_proj
        layer_name_q = name + ".self_attn.q_proj"
        layer_name_k = name + ".self_attn.k_proj"
        layer_name_v = name + ".self_attn.v_proj"
        layer_name_qkv = name + ".self_attn.qkv_proj"

        weight = torch.cat(
            [module.self_attn.q_proj.weight, module.self_attn.k_proj.weight, module.self_attn.v_proj.weight], dim=0
        )

        smoother = smooth_gemm(weight, scales[layer_name_q]["x"], module.input_layernorm.weight, None, alpha)

        scales[layer_name_qkv]["x"] = scales[layer_name_q]["x"] / smoother
        scales[layer_name_qkv]["w"] = weight.abs().max(dim=1)[0]
        scales[layer_name_qkv]["y"] = torch.cat(
            [scales[layer_name_q]["y"], scales[layer_name_k]["y"], scales[layer_name_v]["y"]], dim=0
        )

        # see transpose_weights function
        llama_qkv_para[layer_name_qkv] = weight.transpose(0, 1)

        # =================================================================
        layer_name = name + ".self_attn.o_proj"
        smoother = smooth_gemm(module.self_attn.o_proj.weight, scales[layer_name]["x"], None, None, alpha)
        llama_smoother[layer_name] = smoother.float()

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.self_attn.o_proj.weight.abs().max(dim=1)[0]

        # ==================================================================
        fc1_layer_name = name + ".mlp.gate_proj"
        gate_layer_name = name + ".mlp.up_proj"

        smoother = smooth_gemm_fc1_gate(
            module.mlp.gate_proj.weight,
            module.mlp.up_proj.weight,
            scales[fc1_layer_name]["x"],
            module.post_attention_layernorm.weight,
            None,
            alpha,
        )

        scales[fc1_layer_name]["x"] = scales[fc1_layer_name]["x"] / smoother
        scales[fc1_layer_name]["w"] = module.mlp.gate_proj.weight.abs().max(dim=1)[0]

        scales[gate_layer_name]["x"] = scales[gate_layer_name]["x"] / smoother
        scales[gate_layer_name]["w"] = module.mlp.up_proj.weight.abs().max(dim=1)[0]

        # ==================================================================
        layer_name = name + ".mlp.down_proj"
        smoother = smooth_gemm(module.mlp.down_proj.weight, scales[layer_name]["x"], None, None, alpha)
        llama_smoother[layer_name] = smoother.float()
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.down_proj.weight.abs().max(dim=1)[0]


@torch.no_grad()
def capture_activation_range(model, tokenizer, dataset, num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

    tokenizer.pad_token = tokenizer.eos_token

    def stat_tensor(name, tensor, act_scales, key):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float()

        if act_scales[name][key] is None:
            act_scales[name][key] = comming_max
        else:
            act_scales[name][key] = torch.max(act_scales[name][key], comming_max)

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x, act_scales, "x")
        stat_tensor(name, y, act_scales, "y")

        if act_scales[name]["w"] is None:
            act_scales[name]["w"] = m.weight.abs().clip(1e-8, None).max(dim=1)[0]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, Conv1D):
            hooks.append(m.register_forward_hook(functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples), desc="calibrating model"):
        datapoint = dataset['train'][i : i + 1]
        line = copy.copy(datapoint['article'])
        line[0] = line[0] + ' TL;DR: '
        line[0] = line[0].strip()
        line[0] = line[0].replace(" n't", "n't")
        input_ids = tokenizer(
            line, return_tensors="pt", max_length=seq_len, padding=True, truncation=True
        ).input_ids.to(device)
        model(input_ids)
    for h in hooks:
        h.remove()
    return act_scales


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return torch.chunk(v, tp_size)[idx].contiguous()
    else:
        return torch.chunk(v, tp_size, dim=dim)[idx].contiguous()


def split_qkv_tp(v, n_head, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV matrix according to tensor parallelism
    """
    v = v.reshape(3, n_hidden, n_hidden)
    split_v = split(v, tensor_parallel, rank, dim=1)
    split_v = split_v.reshape(3 * (n_hidden // tensor_parallel), n_hidden)
    return split_v.contiguous()


def split_qkv_bias_tp(v, n_head, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV bias according to tensor parallelism
    """
    v = v.reshape(3, n_hidden)
    split_v = split(v, tensor_parallel, rank, dim=1)
    split_v = split_v.reshape(3 * (n_hidden // tensor_parallel))
    return split_v.contiguous()


def split_matrix_tp(v, tensor_parallel, rank, dim):
    return split(v, tensor_parallel, rank, dim=dim)


def get_weight(config, prefix, dtype):
    if config[prefix + '.weight'].dtype != dtype:
        config[prefix + '.weight'].data = config[prefix + '.weight'].to(dtype)
    return config[prefix + '.weight']


def get_bias(config, prefix, dtype):
    if config[prefix + '.bias'].dtype != dtype:
        config[prefix + '.bias'].data = config[prefix + '.bias'].to(dtype)
    return config[prefix + '.bias']


def get_weight_and_bias(config, prefix, dtype):
    return get_weight(config, prefix, dtype), get_bias(config, prefix, dtype)


def get_tllm_linear_weight(
    weight,
    prefix,
    bias=None,
    use_weight_only=False,
    plugin_weight_only_quant_type=torch.int8,
    dtype='float32',
    use_gemm_woq_plugin=True,
    postfix='weight',
):
    results = {}
    if use_weight_only:
        v = weight.t().contiguous()
        processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
            v.cpu(), plugin_weight_only_quant_type
        )
        if not use_gemm_woq_plugin:
            results[prefix + postfix] = v.to(dtype)
        else:
            results[prefix + postfix] = processed_torch_weights
        results[prefix + 'per_channel_scale'] = torch_weight_scales
    else:
        results[prefix + postfix] = weight.contiguous()

    if bias is not None:
        results[prefix + 'bias'] = bias

    return results


def dup_kv_weight(v, num_head, tp_size):
    assert tp_size % num_head == 0
    reps = tp_size // num_head
    head_size = v.shape[0] // num_head
    v = v.reshape(num_head, head_size, -1)[:, None, :, :].expand(num_head, reps, head_size, v.shape[1])
    return v.reshape(num_head * reps * head_size, -1).clone().detach()


def get_tllm_linear_sq_weight(
    vals,
    prefix,
    shape,
    tensor_parallel,
    is_qkv=False,
    per_token=False,
    per_channel=False,
    last_prefix=None,
    bias=None,
    smoother_value=None,
    smoother_shape=None,
    rank=0,
    cat_dim=0,
    multi_query_mode=False,
):
    results = {}

    def multi_query_split(data, local_dim, head_size, tp_size, cur_rank):
        q, k, v = np.split(data, [local_dim, local_dim + head_size], axis=-1)
        q_split = np.split(q, tp_size, axis=-1)
        k_split = np.split(k, tp_size, axis=-1)
        v_split = np.split(v, tp_size, axis=-1)
        return [np.concatenate((q_split[ii], k_split[ii], v_split[ii]), axis=-1) for ii in range(tp_size)][cur_rank]

    col_shape = shape if (is_qkv or per_channel) else [1, 1]

    if per_token:
        original_weights = vals["weight.int8.col"]

        local_dim = original_weights.shape[0]
        head_size = (original_weights.shape[1] - local_dim) // 2
        if multi_query_mode:
            cur_weights = multi_query_split(original_weights, local_dim, head_size, tensor_parallel, rank)
        else:
            cur_weights = np.split(original_weights, tensor_parallel, axis=cat_dim)[rank]
        if is_qkv:
            hidden_dim = cur_weights.shape[0]
            cur_weights = cur_weights.reshape(hidden_dim, -1)
        results[prefix + 'weight'] = torch.from_numpy(cur_weights).t().contiguous()
        if smoother_value is None:
            results[last_prefix] = torch.from_numpy(np.array([1.0], dtype=np.float32))

        if smoother_value is None:
            if multi_query_mode:
                cur_per_channel_value = multi_query_split(
                    vals["scale_w_quant_orig.col"], local_dim, head_size, tensor_parallel, rank
                )
            else:
                cur_per_channel_value = np.split(vals["scale_w_quant_orig.col"], tensor_parallel, axis=cat_dim)[rank]
        else:
            cur_per_channel_value = vals["scale_w_quant_orig.col"]
        results[prefix + 'per_channel_scale'] = torch.from_numpy(
            np.array(cur_per_channel_value, dtype=np.float32).reshape(col_shape)
        ).contiguous()
    else:
        original_weights = np.array(vals["weight.int8"])
        cur_weights = np.split(original_weights, tensor_parallel, axis=cat_dim)[rank]

        if is_qkv:
            hidden_dim = cur_weights.shape[0]
            cur_weights = cur_weights.reshape(hidden_dim, -1)
        results[prefix + 'weight'] = torch.from_numpy(cur_weights).t().contiguous()
        # 'weight'] = torch.from_numpy(cur_weights).t().contiguous()

        cur_per_channel_value = vals["scale_y_accum_quant"]

        results[prefix + 'per_channel_scale'] = torch.from_numpy(
            np.array([cur_per_channel_value], dtype=np.float32).reshape(col_shape)
        ).contiguous()

        results[last_prefix] = torch.from_numpy(np.array([vals['scale_x_orig_quant']], dtype=np.float32)).contiguous()

        results[prefix + 'act_scale'] = torch.from_numpy(
            np.array([[vals["scale_y_quant_orig"]]], dtype=np.float32)
        ).contiguous()

    if smoother_value is not None:
        cur_smoother_value = np.split(smoother_value, tensor_parallel, axis=cat_dim)[rank]
        results[prefix + 'smoother'] = cur_smoother_value.reshape(smoother_shape).contiguous().to(torch.float32)

    if bias is not None:
        results[prefix + 'bias'] = bias

    return results


def convert_hf_llama(
    hf_model,
    mapping,
    vocab_size=32000,
    dtype='float32',
    use_parallel_embedding=False,
    sharding_dim=0,
    use_weight_only=False,
    share_embedding_table=False,
    use_gemm_woq_plugin=False,
    plugin_weight_only_quant_type=torch.int8,
    use_smooth_quant=False,
    per_channel=False,
    per_token=False,
    int8_kv_cache=False,
    act_range=[],
    qkv_para=[],
    smoother=[],
    moe_config=None,
    lora_config=None,
):
    weights = {}
    tik = time.time()
    tensor_parallel = mapping.tp_size
    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, dtype)
    num_attention_heads = hf_model.config.num_attention_heads
    hidden_size = hf_model.config.hidden_size
    intermediate_size = hf_model.config.intermediate_size
    num_key_value_heads = hf_model.config.num_key_value_heads
    mha_mode = num_key_value_heads == num_attention_heads

    layers_per_pipeline_stage = hf_model.config.num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage, (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1)
    )

    if moe_config and moe_config.has_moe():
        rank_experts = list(range(moe_config.num_experts))
        if moe_config.tp_mode == moe_config.ParallelismMode.EXPERT_PARALLEL:
            rank_experts = mapping.ep_experts(moe_config.num_experts)

        for l in range(hf_model.config.num_hidden_layers):
            for suffix in ["w1", "w2", "w3"]:
                model_params[f'model.layers.{l}.block_sparse_moe.experts.{suffix}.weight'] = torch.stack(
                    list(
                        model_params[f'model.layers.{l}.block_sparse_moe.experts.{expert}.{suffix}.weight']
                        for expert in rank_experts
                    )
                )
            w3 = model_params[f'model.layers.{l}.block_sparse_moe.experts.w3.weight']
            w2 = model_params[f'model.layers.{l}.block_sparse_moe.experts.w2.weight']
            w1 = model_params[f'model.layers.{l}.block_sparse_moe.experts.w1.weight']
            if moe_config.tp_mode == moe_config.ParallelismMode.TENSOR_PARALLEL:
                w3 = split(w3, mapping.tp_size, mapping.tp_rank, dim=1)
                w2 = split(w2, mapping.tp_size, mapping.tp_rank, dim=2)
                w1 = split(w1, mapping.tp_size, mapping.tp_rank, dim=1)
            # concat w3 and w1 for gated expert
            model_params[f'model.layers.{l}.block_sparse_moe.experts.w3w1.weight'] = torch.concat([w3, w1], dim=-2)
            model_params[f'model.layers.{l}.block_sparse_moe.experts.w2.weight'] = w2

    for l in range(hf_model.config.num_hidden_layers):
        if l not in layers_range:
            continue
        prefix = f'model.layers.{l}.'
        idx = int(l) - mapping.pp_rank * layers_per_pipeline_stage
        tllm_prex = f'transformer.layers.{idx}.'
        q_weight = get_weight(model_params, prefix + 'self_attn.q_proj', dtype)
        k_weight = get_weight(model_params, prefix + 'self_attn.k_proj', dtype)
        v_weight = get_weight(model_params, prefix + 'self_attn.v_proj', dtype)

        if not mha_mode:
            head_size = hidden_size // num_attention_heads
            if num_key_value_heads < tensor_parallel:
                # duplicate the KV heads up to tensor_parallel
                k_weight = dup_kv_weight(k_weight, num_key_value_heads, tensor_parallel)
                v_weight = dup_kv_weight(v_weight, num_key_value_heads, tensor_parallel)
            assert (k_weight.shape[0] % (mapping.tp_size * head_size)) == 0
            assert (v_weight.shape[0] % (mapping.tp_size * head_size)) == 0

            wq = split(q_weight, mapping.tp_size, mapping.tp_rank)
            wk = split(k_weight, mapping.tp_size, mapping.tp_rank)
            wv = split(v_weight, mapping.tp_size, mapping.tp_rank)

            split_v = torch.concat((wq, wk, wv))

        else:
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

            split_v = split_qkv_tp(qkv_weight, num_attention_heads, hidden_size, tensor_parallel, mapping.tp_rank)
        if use_smooth_quant:
            qkv_weight = qkv_para[prefix + 'self_attn.qkv_proj']

            if not mha_mode:
                hidden_size = qkv_weight.shape[0]
                local_dim = hidden_size
                head_size = (qkv_weight.shape[-1] - local_dim) // 2
                qkv_weight = qkv_weight.reshape(hidden_size, local_dim + 2 * head_size)
            else:
                qkv_weight = qkv_weight.reshape(hidden_size, 3, hidden_size)

            int8_weights = generate_int8(
                qkv_weight,
                act_range.get(prefix + 'self_attn.qkv_proj'),
                is_qkv=True,
                multi_query_mode=bool(not mha_mode),
            )

            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + 'attention.qkv.',
                    [
                        1,
                        (
                            3 * hidden_size // tensor_parallel
                            if mha_mode
                            else hidden_size // tensor_parallel
                            + (hidden_size // num_key_value_heads) // tensor_parallel * 2
                        ),
                    ],
                    tensor_parallel,
                    is_qkv=True,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=tllm_prex + 'input_layernorm.scale_to_int',
                    smoother_value=None,
                    smoother_shape=None,
                    rank=mapping.tp_rank,
                    cat_dim=-1,
                    multi_query_mode=bool(not mha_mode),
                )
            )
        else:
            weights.update(
                get_tllm_linear_weight(
                    split_v,
                    tllm_prex + 'attention.qkv.',
                    None,
                    use_weight_only,
                    plugin_weight_only_quant_type,
                    dtype,
                    use_gemm_woq_plugin,
                )
            )

        if int8_kv_cache:
            qkv_y = torch.cat(
                [
                    act_range.get(prefix + 'self_attn.q_proj')["y"],
                    act_range.get(prefix + 'self_attn.k_proj')["y"],
                    act_range.get(prefix + 'self_attn.v_proj')["y"],
                ],
                dim=0,
            )

            int8_kv_scales = qkv_y.max() / 127.0

            kv_cache_weights = {}

            kv_cache_weights[tllm_prex + 'attention.kv_cache_scaling_factor'] = int8_kv_scales.reshape([1])

            weights.update(kv_cache_weights)

        attn_dense_weight = get_weight(model_params, prefix + 'self_attn.o_proj', dtype)
        split_v = split_matrix_tp(attn_dense_weight, tensor_parallel, mapping.tp_rank, dim=1)
        if use_smooth_quant:
            attn_dense_weight = attn_dense_weight.t()
            int8_weights = generate_int8(attn_dense_weight, act_range.get(prefix + 'self_attn.o_proj'))
            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + 'attention.dense.',
                    [1, hidden_size],
                    tensor_parallel,
                    is_qkv=False,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=tllm_prex + 'attention.quantization_scaling_factor',
                    smoother_value=smoother[(prefix + 'self_attn.o_proj')],
                    smoother_shape=[1, hidden_size // tensor_parallel],
                    rank=mapping.tp_rank,
                    cat_dim=0,
                )
            )
        else:
            weights.update(
                get_tllm_linear_weight(
                    split_v,
                    tllm_prex + 'attention.dense.',
                    None,
                    use_weight_only,
                    plugin_weight_only_quant_type,
                    dtype,
                    use_gemm_woq_plugin,
                )
            )

        if moe_config and moe_config.has_moe():
            ## block_sparse_moe.experts.w2.weight
            moe_experts_w2_weights = get_weight(model_params, prefix + 'block_sparse_moe.experts.w2', dtype)
            weights.update(
                get_tllm_linear_weight(
                    moe_experts_w2_weights,
                    tllm_prex + 'mlp.experts_weight_2',
                    None,
                    use_weight_only,
                    plugin_weight_only_quant_type,
                    dtype,
                    use_gemm_woq_plugin,
                    postfix='',
                )
            )
            ##block_sparse_moe.experts.w3w1.weight
            moe_experts_w3w1_weights = get_weight(model_params, prefix + 'block_sparse_moe.experts.w3w1', dtype)
            weights.update(
                get_tllm_linear_weight(
                    moe_experts_w3w1_weights,
                    tllm_prex + 'mlp.experts_weight_1',
                    None,
                    use_weight_only,
                    plugin_weight_only_quant_type,
                    dtype,
                    use_gemm_woq_plugin,
                    postfix='',
                )
            )

            moe_experts_gate_weights = get_weight(model_params, prefix + 'block_sparse_moe.gate', dtype)
            v = split(moe_experts_gate_weights, mapping.tp_size, mapping.tp_rank, dim=-1)

            weights.update(
                get_tllm_linear_weight(
                    v.to(torch.float32),
                    tllm_prex + 'mlp.router.',
                    None,
                    use_weight_only,
                    plugin_weight_only_quant_type,
                    dtype,
                    use_gemm_woq_plugin,
                )
            )
        else:
            mlp_gate_weight = get_weight(model_params, prefix + 'mlp.up_proj', dtype)
            split_v = split_matrix_tp(mlp_gate_weight, tensor_parallel, mapping.tp_rank, dim=0)
            if use_smooth_quant:
                mlp_gate_weight = mlp_gate_weight.t()
                int8_weights = generate_int8(mlp_gate_weight, act_range.get(prefix + 'mlp.up_proj'))

                weights.update(
                    get_tllm_linear_sq_weight(
                        int8_weights,
                        tllm_prex + 'mlp.gate.',
                        [1, intermediate_size // tensor_parallel],
                        tensor_parallel,
                        is_qkv=False,
                        per_token=per_token,
                        per_channel=per_channel,
                        last_prefix=tllm_prex + 'post_layernorm.scale_to_int',
                        smoother_value=None,
                        smoother_shape=None,
                        rank=mapping.tp_rank,
                        cat_dim=-1,
                    )
                )
            else:
                weights.update(
                    get_tllm_linear_weight(
                        split_v,
                        tllm_prex + 'mlp.gate.',
                        None,
                        use_weight_only,
                        plugin_weight_only_quant_type,
                        dtype,
                        use_gemm_woq_plugin,
                    )
                )

            mlp_fc_weight = get_weight(model_params, prefix + 'mlp.gate_proj', dtype)
            split_v = split_matrix_tp(mlp_fc_weight, tensor_parallel, mapping.tp_rank, dim=0)

            if use_smooth_quant:
                mlp_fc_weight = mlp_fc_weight.t()  # verified
                int8_weights = generate_int8(mlp_fc_weight, act_range.get(prefix + 'mlp.gate_proj'))
                weights.update(
                    get_tllm_linear_sq_weight(
                        int8_weights,
                        tllm_prex + 'mlp.fc.',
                        [1, intermediate_size // tensor_parallel],
                        tensor_parallel,
                        is_qkv=False,
                        per_token=per_token,
                        per_channel=per_channel,
                        last_prefix=tllm_prex + 'post_layernorm.scale_to_int',
                        smoother_value=None,
                        smoother_shape=None,
                        rank=mapping.tp_rank,
                        cat_dim=-1,
                    )
                )
            else:
                weights.update(
                    get_tllm_linear_weight(
                        split_v,
                        tllm_prex + 'mlp.fc.',
                        None,
                        use_weight_only,
                        plugin_weight_only_quant_type,
                        dtype,
                        use_gemm_woq_plugin,
                    )
                )

            mlp_proj_weight = get_weight(model_params, prefix + 'mlp.down_proj', dtype)
            split_v = split_matrix_tp(mlp_proj_weight, tensor_parallel, mapping.tp_rank, dim=1)

            if use_smooth_quant:
                mlp_proj_weight = mlp_proj_weight.t()
                int8_weights = generate_int8(mlp_proj_weight, act_range.get(prefix + 'mlp.down_proj'))
                weights.update(
                    get_tllm_linear_sq_weight(
                        int8_weights,
                        tllm_prex + 'mlp.proj.',
                        [1, hidden_size],
                        tensor_parallel,
                        is_qkv=False,
                        per_token=per_token,
                        per_channel=per_channel,
                        last_prefix=tllm_prex + 'mlp.quantization_scaling_factor',
                        smoother_value=smoother[prefix + 'mlp.down_proj'],
                        smoother_shape=[1, intermediate_size // tensor_parallel],
                        rank=mapping.tp_rank,
                        cat_dim=0,
                    )
                )
            else:
                weights.update(
                    get_tllm_linear_weight(
                        split_v,
                        tllm_prex + 'mlp.proj.',
                        None,
                        use_weight_only,
                        plugin_weight_only_quant_type,
                        dtype,
                        use_gemm_woq_plugin,
                    )
                )

        # Layer norms do not use tensor parallelism
        input_ln_weight = get_weight(model_params, prefix + 'input_layernorm', dtype)
        weights[tllm_prex + 'input_layernorm.weight'] = input_ln_weight

        post_ln_weight = get_weight(model_params, prefix + 'post_attention_layernorm', dtype)
        weights[tllm_prex + 'post_layernorm.weight'] = post_ln_weight

    v = get_weight(model_params, 'model.embed_tokens', dtype)
    if lora_config.is_valid and lora_config.embedding_weight is not None:
        v = lora_config.embedding_weight
    if hf_model.config.tie_word_embeddings:
        # lm_head.weight has the same weights as embedding
        if mapping.is_last_pp_rank():
            if vocab_size % mapping.tp_size != 0:
                # padding
                vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
                pad_width = vocab_size_padded - vocab_size

                v = torch.from_numpy(
                    np.pad(v.detach().cpu().numpy(), ((0, pad_width), (0, 0)), 'constant', constant_values=0)
                )
            weights['lm_head.weight'] = split(v, mapping.tp_size, mapping.tp_rank)

    if use_parallel_embedding:
        v = split_matrix_tp(v, mapping.tp_size, mapping.tp_rank, dim=sharding_dim)

    if mapping.is_first_pp_rank():
        weights['transformer.vocab_embedding.weight'] = v

    # if not use_parallel_embedding:
    #     weights['transformer.vocab_embedding.weight'] = embed_w
    # else:
    #     assert hf_model.config.vocab_size % tensor_parallel == 0
    #     weights['transformer.vocab_embedding.weight'] = split_matrix_tp(
    #         embed_w, tensor_parallel, rank

    lm_head_weights = get_weight(model_params, 'lm_head', dtype)

    if mapping.is_last_pp_rank():
        if lora_config.is_valid and lora_config.lm_head_weight is not None:
            lm_head_weights = lora_config.lm_head_weight

        if vocab_size % mapping.tp_size != 0:
            # padding
            vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
            pad_width = vocab_size_padded - vocab_size

            lm_head_weights = torch.from_numpy(
                np.pad(lm_head_weights.detach().cpu().numpy(), ((0, pad_width), (0, 0)), 'constant', constant_values=0)
            )
        weights['lm_head.weight'] = split_matrix_tp(lm_head_weights, tensor_parallel, mapping.tp_rank, dim=0)
        ln_f_w = get_weight(model_params, 'model.norm', dtype)
        weights['transformer.ln_f.weight'] = ln_f_w

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def main():
    # TODO(qijun): Currently, the convert script depends on a torch op:
    # torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix,
    # which is included in tensorrt_llm Python package. Otherwise, the convert
    # script does not need to import tensorrt_llm. Will remove it after reimplementing
    # the op with PyTorch.
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    world_size = args.tp_size * args.pp_size

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    hf_config = None
    if args.model_dir is not None:
        hf_config = LlamaConfig.from_pretrained(args.model_dir)
        if hf_config.model_type == "llava":
            # LLaVA = Vision model + Llama LLM
            # We load a llava config and use its' text config as llama config
            hf_config = LlavaConfig.from_pretrained(args.model_dir).text_config
            hf_config.model_type = "llava"  # Replace llama with llava

        args.model_type = hf_config.model_type
        args.n_head = hf_config.num_attention_heads
        args.inter_size = hf_config.intermediate_size
        args.n_layer = hf_config.num_hidden_layers
        args.n_embd = hf_config.hidden_size
        args.n_kv_head = hf_config.num_key_value_heads
        args.rms_norm_eps = hf_config.rms_norm_eps
        args.vocab_size = hf_config.vocab_size
        args.n_positions = hf_config.max_position_embeddings
        args.rotary_base = getattr(hf_config, "rope_theta", args.rotary_base)
        if hf_config.model_type == "mixtral":
            # HF LLaMA-type models are implicitly using gated activation.
            # With our MoE implementation, we must make it explicit
            args.hidden_act = "swiglu"
            args.moe_num_experts = getattr(hf_config, "num_local_experts", args.moe_num_experts)
            args.moe_top_k = getattr(hf_config, "num_experts_per_tok", args.moe_top_k)

    elif args.meta_ckpt_dir is not None:
        with open(Path(args.meta_ckpt_dir, "params.json")) as fp:
            meta_config: dict = json.load(fp)
        args.n_embd = meta_config["dim"]
        args.n_head = meta_config["n_heads"]
        args.n_layer = meta_config["n_layers"]
        args.n_kv_head = meta_config.get("n_kv_heads", args.n_head)

        if "hidden_dim" in meta_config:
            args.inter_size = meta_config["hidden_dim"]
        else:
            args.multiple_of = meta_config.get("multiple_of", 1)
            n_embd = int(4 * args.n_embd * 2 / 3)
            args.ffn_dim_multiplier = meta_config.get("ffn_dim_multiplier", 1)
            args.inter_size = args.multiple_of * (
                (int(n_embd * args.ffn_dim_multiplier) + args.multiple_of - 1) // args.multiple_of
            )
        args.rms_norm_eps = meta_config["norm_eps"]
        args.moe_num_experts = meta_config.get("moe", {}).get("num_experts", 0)
        args.moe_top_k = meta_config.get("moe", {}).get("num_experts_per_tok", 0)
    else:
        args.n_kv_head = args.n_kv_head or args.n_head

    if args.moe_num_experts and args.moe_top_k == 0:
        args.moe_top_k = 1
    args.moe_config = MoeConfig(
        args.moe_num_experts, args.moe_top_k, args.moe_tp_mode, args.moe_renorm_mode
    ).validate()

    if args.rotary_scaling is not None:
        # assert args.use_gpt_attention_plugin, "RoPE scaling is only supported through GPT attention plugin."
        rotary_scaling = {"type": args.rotary_scaling[0], "factor": float(args.rotary_scaling[1])}
        assert rotary_scaling["type"] in ["linear", "dynamic"]
        assert rotary_scaling["factor"] > 1.0
        args.rotary_scaling = rotary_scaling

    hf_modules_to_trtllm_modules = {
        "q_proj": "attn_q",
        "k_proj": "attn_k",
        "v_proj": "attn_v",
        "o_proj": "attn_dense",
        "gate_proj": "mlp_h_to_4h",
        "down_proj": "mlp_4h_to_h",
        "up_proj": "mlp_gate",
    }  # lora modules on llama

    trtllm_modules_to_hf_modules = {
        "attn_q": "q_proj",
        "attn_k": "k_proj",
        "attn_v": "v_proj",
        "attn_dense": "o_proj",
        "mlp_h_to_4h": "gate_proj",
        "mlp_4h_to_h": "down_proj",
        "mlp_gate": "up_proj",
    }

    lora_config = LoraConfig.from_hf(args.hf_lora_dir, hf_modules_to_trtllm_modules, trtllm_modules_to_hf_modules)

    if lora_config.is_valid and lora_config.vocab_size != 0:
        if args.lora_target_modules is None:
            args.lora_target_modules = lora_config.lora_target_modules

        # the lora checkpoint might finetune the embedding
        if lora_config.vocab_size != 0:
            args.vocab_size = lora_config.vocab_size

    args.lora_config = lora_config

    config = {
        'architecture': hf_config.architectures[0] if hf_config is not None else "LlamaForCausalLM",
        'dtype': args.dtype,
        'logits_dtype': 'float32',
        'num_hidden_layers': args.n_layer,
        'num_attention_heads': args.n_head,
        'hidden_size': args.n_embd,
        'intermediate_size': args.inter_size,
        'num_key_value_heads': args.n_kv_head,
        'vocab_size': args.vocab_size,
        'position_embedding_type': 'rope_gpt_neox',
        'max_position_embeddings': args.n_positions,
        'hidden_act': args.hidden_act,
        'rotary_base': args.rotary_base,
        'rotary_scaling': args.rotary_scaling,
        'norm_epsilon': args.rms_norm_eps,
        'quantization': {
            'quant_algo': None,
            'kv_cache_quant_algo': None,
            "sq_use_plugin": False,
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        "moe_config": {
            "num_experts": args.moe_num_experts,
            "top_k": args.moe_top_k,
            "tp_mode": args.moe_tp_mode,
            "normalization_mode": args.moe_renorm_mode,
        },
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'share_embedding_table': args.use_embedding_sharing,
        'use_prompt_tuning': args.use_prompt_tuning,
        'moe_num_experts': args.moe_num_experts,
        'moe_top_k': args.moe_top_k,
        'moe_tp_mode': args.moe_tp_mode,
        'moe_normalization_mode': args.moe_renorm_mode,
        'enable_pos_shift': args.enable_pos_shift,
        'dense_context_fmha': args.dense_context_fmha,
        'max_lora_rank': args.max_lora_rank,
        'lora_target_modules': args.lora_target_modules,
        'hf_modules_to_trtllm_modules': args.lora_config.hf_modules_to_trtllm_modules,
        'trtllm_modules_to_hf_modules': args.lora_config.trtllm_modules_to_hf_modules,
        'disable_weight_only_quant_plugin': args.disable_weight_only_quant_plugin,
    }

    if args.use_weight_only:
        if args.weight_only_precision == 'int8':
            config['quantization']['quant_algo'] = 'W8A16'
        elif args.weight_only_precision == 'int4':
            config['quantization']['quant_algo'] = 'W4A16'
    elif args.smoothquant:
        config['quantization']['sq_use_plugin'] = True
        if args.per_channel:
            if args.per_token:
                config['quantization']['quant_algo'] = 'W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN'
            else:
                config['quantization']['quant_algo'] = 'W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN'
        else:
            if args.per_token:
                config['quantization']['quant_algo'] = 'W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN'
            else:
                config['quantization']['quant_algo'] = 'W8A8_SQ_PER_TENSOR_PLUGIN'
    elif args.enable_fp8:
        config['quantization']['quant_algo'] = 'FP8'
    if args.int8_kv_cache:
        config['quantization']['kv_cache_quant_algo'] = 'INT8'
    elif args.fp8_kv_cache:
        config['quantization']['kv_cache_quant_algo'] = 'FP8'

    if args.weight_only_precision == 'int4_gptq':
        config['quantization'].update(
            {
                "group_size": args.group_size,
                "has_zero_point": True,
                "pre_quant_scale": False,
                'quant_algo': 'W4A16_GPTQ',
            }
        )

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    if args.model_dir is None and args.meta_ckpt_dir is None:
        return

    if args.weight_only_precision == 'int8':
        plugin_weight_only_quant_type = torch.int8
    elif args.weight_only_precision == 'int4':
        plugin_weight_only_quant_type = torch.quint4x2

    act_range = {}
    llama_qkv_para = {}
    # smoother for inputs of self_attn.o_proj and mlp.down_proj
    llama_smoother = {}
    model = None
    if args.model_dir is not None:
        if args.model_type == "llava":
            hf_llava = LlavaForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype="auto")
            model = hf_llava.language_model
        else:
            hf_model = LlamaForCausalLM if args.model_type != "mixtral" else MixtralForCausalLM
            model = hf_model.from_pretrained(
                args.model_dir,
                # device_map={
                #     "model": "cpu",
                #     "lm_head": "cpu",
                #     "embed_tokens": "cpu",
                #     "layers": "cpu",
                #     "norm": "cpu",
                # },  # Load to CPU memory
                device_map='auto',
                torch_dtype='auto',
            )

        if args.smoothquant is not None or args.int8_kv_cache:
            os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "false")
            if args.load_model_on_cpu:
                logger.warning("Note that running capture_activation_range on cpu would be very small.")
            dataset = load_dataset("ccdv/cnn_dailymail", '3.0.0', cache_dir=args.dataset_cache_dir)

            act_range = capture_activation_range(
                model, LlamaTokenizer.from_pretrained(args.model_dir, padding_side='left'), dataset
            )
            if args.smoothquant is not None:
                smooth_llama_model(model, act_range, args.smoothquant, llama_qkv_para, llama_smoother)
    convert_args = {
        'hf_model': model,
        'act_range': act_range,
        'llama_qkv_para': llama_qkv_para,
        'llama_smoother': llama_smoother,
    }

    def covert_and_save(rank, convert_args):
        mapping = Mapping(world_size=world_size, rank=rank, tp_size=args.tp_size, pp_size=args.pp_size)

        if args.use_weight_only and args.weight_only_precision == 'int4_gptq':
            weights = load_from_gptq_llama(
                args.ammo_quant_ckpt_path, args.n_layer, args.vocab_size, mapping, dtype=args.dtype
            )

        elif args.meta_ckpt_dir is not None:
            weights = load_from_meta_llama(
                args.meta_ckpt_dir, mapping, PretrainedConfig.from_dict(copy.deepcopy(config))
            )

            if args.enable_fp8 or args.fp8_kv_cache:
                scales = load_from_fp8_llama(args.ammo_quant_ckpt_path, args.n_layer, mapping, args.fp8_kv_cache)
                weights.update(scales)

        else:
            if args.load_by_shard:
                weights = load_from_hf_checkpoint(
                    args.model_dir, mapping, PretrainedConfig.from_dict(copy.deepcopy(config)), args.lora_config
                )

            else:
                weights = convert_hf_llama(
                    convert_args['hf_model'],
                    mapping,
                    vocab_size=args.vocab_size,
                    dtype=args.dtype,
                    use_weight_only=args.use_weight_only,
                    use_gemm_woq_plugin=not args.disable_weight_only_quant_plugin,
                    plugin_weight_only_quant_type=plugin_weight_only_quant_type,
                    use_parallel_embedding=args.use_parallel_embedding,
                    sharding_dim=args.embedding_sharding_dim,
                    share_embedding_table=args.use_embedding_sharing,
                    use_smooth_quant=args.smoothquant,
                    per_channel=args.per_channel,
                    per_token=args.per_token,
                    int8_kv_cache=args.int8_kv_cache,
                    act_range=convert_args['act_range'],
                    qkv_para=convert_args['llama_qkv_para'],
                    smoother=convert_args['llama_smoother'],
                    moe_config=args.moe_config,
                    lora_config=args.lora_config,
                )

                if args.enable_fp8 or args.fp8_kv_cache:
                    scales = load_from_fp8_llama(args.ammo_quant_ckpt_path, args.n_layer, mapping, args.fp8_kv_cache)
                    weights.update(scales)

        safetensors.torch.save_file(weights, os.path.join(args.output_dir, f'rank{rank}.safetensors'))

    if args.workers == 1:
        for rank in range(world_size):
            covert_and_save(rank, convert_args)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as p:
            futures = [p.submit(covert_and_save, rank, convert_args) for rank in range(world_size)]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    exceptions.append(e)
            assert len(exceptions) == 0, "Checkpoint conversion failed, please check error log."

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
