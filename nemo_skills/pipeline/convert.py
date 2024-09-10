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

from argparse import ArgumentParser

import nemo_run as run
from huggingface_hub import get_token

from nemo_skills.pipeline import add_task, check_if_mounted, get_cluster_config, run_exp
from nemo_skills.utils import setup_logging


def get_nemo_to_hf_cmd(input_model, output_model, hf_model_name, dtype, num_gpus, num_nodes, extra_arguments):
    cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"export HF_TOKEN={get_token()} && "
        f"cd /nemo_run/code && "
        f"python nemo_skills/conversion/nemo_to_hf.py "
        f"    --in-path {input_model} "
        f"    --out-path {output_model} "
        f"    --hf-model-name {hf_model_name} "
        f"    --precision {dtype} "
        f"    --max-shard-size 10GB "
        f"    {extra_arguments} "
    )
    return cmd


def get_hf_to_trtllm_cmd(input_model, output_model, hf_model_name, dtype, num_gpus, num_nodes, extra_arguments):
    dtype = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }[dtype]
    cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"export HF_TOKEN={get_token()} && "
        f"cd /nemo_run/code && "
        f"python nemo_skills/conversion/hf_to_trtllm.py "
        f"    --model_dir {input_model} "
        f"    --output_dir {output_model}-tmp "
        f"    --dtype {dtype} "
        f"    --tp_size {num_gpus} "
        f"    --pp_size {num_nodes} &&"
        f"trtllm-build "
        f"    --checkpoint_dir {output_model}-tmp "
        f"    --output_dir {output_model} "
        f"    --gpt_attention_plugin {dtype} "
        f"    --use_paged_context_fmha enable "
        # some decent defaults, but each model needs different values for best performance
        f"    --max_input_len 4096 "
        f"    --max_seq_len 8192 "
        f"    --max_num_tokens 8192 "
        f"    --max_batch_size 128 "
        f"    {extra_arguments} && "
        f"cp {input_model}/tokenizer* {output_model} "
    )
    return cmd


def get_hf_to_nemo_cmd(input_model, output_model, hf_model_name, dtype, num_gpus, num_nodes, extra_arguments):
    cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"export HF_TOKEN={get_token()} && "
        f"cd /nemo_run/code && "
        f"python nemo_skills/conversion/hf_to_nemo.py "
        f"    --in-path {input_model} "
        f"    --out-path {output_model} "
        f"    --hf-model-name {hf_model_name} "
        f"    --precision {dtype} "
        f"    {extra_arguments} "
    )
    return cmd


if __name__ == "__main__":
    setup_logging(disable_hydra_logs=False)
    parser = ArgumentParser()
    parser.add_argument("--cluster", required=True, help="One of the configs inside cluster_configs")
    parser.add_argument("--input_model", required=True)
    parser.add_argument("--output_model", required=True, help="Where to put the final model")
    parser.add_argument("--convert_from", default="nemo", help="Format of the input model", choices=["nemo", "hf"])
    parser.add_argument(
        "--convert_to", required=True, help="Format of the output model", choices=["nemo", "hf", "tensorrt_llm"]
    )
    parser.add_argument(
        "--hf_model_name", required=False, help="Name of the model on Hugging Face Hub to convert to/from"
    )
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--expname", default="conversion", help="NeMo-Run experiment name")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_gpus", required=True, type=int)
    parser.add_argument(
        "--partition",
        required=False,
        help="Can specify if need interactive jobs or a specific non-default partition",
    )
    parser.add_argument(
        "--run_after",
        required=False,
        help="Can specify an expname that needs to be completed before this one starts (will use as slurm dependency)",
    )

    args, unknown = parser.parse_known_args()
    extra_arguments = f'{" ".join(unknown)}'

    # TODO: add support for conversion from NeMo to trtllm using nemo.export (need to test thoroughly)
    if args.convert_from == "nemo" and args.convert_to == "tensorrt_llm":
        raise ValueError("Conversion from NeMo to TensorRT LLM is not supported directly. Convert to HF first.")

    if args.convert_to != "tensorrt_llm" and args.hf_model_name is None:
        raise ValueError("--hf_model_name is required")

    cluster_config = get_cluster_config(args.cluster)
    check_if_mounted(cluster_config, args.input_model)
    check_if_mounted(cluster_config, args.output_model)

    conversion_cmd_map = {
        ("nemo", "hf"): get_nemo_to_hf_cmd,
        ("hf", "nemo"): get_hf_to_nemo_cmd,
        ("hf", "tensorrt_llm"): get_hf_to_trtllm_cmd,
    }
    container_map = {
        ("nemo", "hf"): cluster_config["containers"]["nemo"],
        ("hf", "nemo"): cluster_config["containers"]["nemo"],
        ("hf", "tensorrt_llm"): cluster_config["containers"]["tensorrt_llm"],
    }
    conversion_cmd = conversion_cmd_map[(args.convert_from, args.convert_to)](
        input_model=args.input_model,
        output_model=args.output_model,
        hf_model_name=args.hf_model_name,
        dtype=args.dtype,
        num_gpus=args.num_gpus,
        num_nodes=args.num_nodes,
        extra_arguments=extra_arguments,
    )

    with run.Experiment(args.expname) as exp:
        add_task(
            exp,
            cmd=conversion_cmd,
            task_name=f'conversion-{args.convert_from}:{args.convert_to}',
            container=container_map[(args.convert_from, args.convert_to)],
            num_gpus=args.num_gpus,
            num_nodes=args.num_nodes,
            num_tasks=1,
            cluster_config=cluster_config,
            partition=args.partition,
            run_after=args.run_after,
        )
        run_exp(exp, cluster_config)
