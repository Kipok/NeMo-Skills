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

# copied from https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/qwen/convert_checkpoint.py

import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import tensorrt_llm
from tensorrt_llm._utils import release_gc
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import QWenForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None, required=True)
    parser.add_argument('--tp_size', type=int, default=1, help='N-way tensor parallelism size')
    parser.add_argument('--pp_size', type=int, default=1, help='N-way pipeline parallelism size')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float32', 'bfloat16', 'float16'])
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
        '--calib_dataset',
        type=str,
        default='ccdv/cnn_dailymail',
        help="The huggingface dataset name or the local directory of the dataset for calibration.",
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
        '--per_group',
        default=False,
        action="store_true",
        help='By default, we use a single static scaling factor to scale weights in the int4 range. '
        'per_group chooses at run time, and for each group, a custom scaling factor. '
        'The flag is built for GPTQ/AWQ quantization.',
    )

    parser.add_argument('--group_size', type=int, default=128, help='Group size used in GPTQ quantization.')

    parser.add_argument("--load_model_on_cpu", action="store_true")
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
    parser.add_argument(
        '--output_dir', type=str, default='tllm_checkpoint', help='The path to save the TensorRT-LLM checkpoint'
    )
    parser.add_argument(
        '--workers', type=int, default=1, help='The number of workers for converting checkpoint in parallel'
    )
    parser.add_argument(
        '--moe_tp_size',
        type=int,
        default=-1,
        help='N-way tensor parallelism size for MOE, default is tp_size, which will do tp-only for MoE',
    )
    parser.add_argument(
        '--moe_ep_size',
        type=int,
        default=-1,
        help='N-way expert parallelism size for MOE, default is 1, which will do tp-only for MoE',
    )
    args = parser.parse_args()
    return args


def args_to_quant_config(args: argparse.Namespace) -> QuantConfig:
    '''return config dict with quantization info based on the command line args'''
    quant_config = QuantConfig()
    if args.use_weight_only:
        if args.weight_only_precision == 'int8':
            quant_config.quant_algo = QuantAlgo.W8A16
        elif args.weight_only_precision == 'int4':
            quant_config.quant_algo = QuantAlgo.W4A16
    elif args.smoothquant:
        quant_config.smoothquant_val = args.smoothquant
        if args.per_channel:
            if args.per_token:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN
            else:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN
        else:
            if args.per_token:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN
            else:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN

    if args.int8_kv_cache:
        quant_config.kv_cache_quant_algo = QuantAlgo.INT8

    if args.weight_only_precision == 'int4_gptq':
        quant_config.group_size = args.group_size
        quant_config.has_zero_point = True
        quant_config.pre_quant_scale = False
        quant_config.quant_algo = QuantAlgo.W4A16_GPTQ

    return quant_config


def args_to_build_options(args):
    return {
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'share_embedding_table': args.use_embedding_sharing,
        'disable_weight_only_quant_plugin': args.disable_weight_only_quant_plugin,
    }


def convert_and_save_hf(args):
    model_dir = args.model_dir
    world_size = args.tp_size * args.pp_size
    # Need to convert the cli args to the kay-value pairs and override them in the generate config dict.
    # Ideally these fields will be moved out of the config and pass them into build API, keep them here for compatibility purpose for now,
    # before the refactor is done.
    override_fields = {}
    override_fields.update(args_to_build_options(args))

    # Qwen models have GPTQ-quantized checkpoint available on HF.
    use_hf_gptq_checkpoint = args.use_weight_only and args.weight_only_precision == 'int4_gptq'
    quant_config = args_to_quant_config(args)

    if args.smoothquant is not None or args.int8_kv_cache:
        mapping = Mapping(
            world_size=world_size,
            tp_size=args.tp_size,
            pp_size=args.pp_size,
            moe_tp_size=args.moe_tp_size,
            moe_ep_size=args.moe_ep_size,
        )
        QWenForCausalLM.quantize(
            args.model_dir,
            args.output_dir,
            dtype=args.dtype,
            mapping=mapping,
            quant_config=quant_config,
            calib_dataset=args.calib_dataset,
            **override_fields,
        )
    else:

        def convert_and_save_rank(args, rank):
            mapping = Mapping(
                world_size=world_size,
                rank=rank,
                tp_size=args.tp_size,
                pp_size=args.pp_size,
                moe_tp_size=args.moe_tp_size,
                moe_ep_size=args.moe_ep_size,
            )
            qwen = QWenForCausalLM.from_hugging_face(
                model_dir,
                args.dtype,
                mapping=mapping,
                quant_config=quant_config,
                use_hf_gptq_checkpoint=use_hf_gptq_checkpoint,
                **override_fields,
            )
            qwen.save_checkpoint(args.output_dir, save_config=(rank == 0))
            del qwen

        execute(args.workers, [convert_and_save_rank] * world_size, args)
        release_gc()


def execute(workers, func, args):
    if workers == 1:
        for rank, f in enumerate(func):
            f(args, rank)
    else:
        with ThreadPoolExecutor(max_workers=workers) as p:
            futures = [p.submit(f, args, rank) for rank, f in enumerate(func)]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    exceptions.append(e)
            assert len(exceptions) == 0, "Checkpoint conversion failed, please check error log."


def main():
    print(tensorrt_llm.__version__)
    args = parse_arguments()

    if args.moe_tp_size == -1 and args.moe_ep_size == -1:
        # moe default to tp-only
        args.moe_tp_size = args.tp_size
        args.moe_ep_size = 1
    elif args.moe_tp_size == -1:
        args.moe_tp_size = args.tp_size // args.moe_ep_size
    elif args.moe_ep_size == -1:
        args.moe_ep_size = args.tp_size // args.moe_tp_size
    assert args.moe_tp_size * args.moe_ep_size == args.tp_size, "moe_tp_size * moe_ep_size must equal to tp_size"

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    assert args.model_dir is not None
    convert_and_save_hf(args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
