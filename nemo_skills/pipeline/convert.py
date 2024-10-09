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
import logging
from enum import Enum
from pathlib import Path

import nemo_run as run
import typer
from huggingface_hub import get_token

from nemo_skills.pipeline import add_task, check_if_mounted, get_cluster_config, run_exp
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.utils import setup_logging

LOG = logging.getLogger(__file__)


def get_nemo_to_hf_cmd(
    input_model, output_model, model_type, hf_model_name, dtype, num_gpus, num_nodes, extra_arguments
):
    cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"export HF_TOKEN={get_token()} && "
        f"cd /nemo_run/code && "
        f"python -m nemo_skills.conversion.nemo_to_hf "
        f"    --in-path {input_model} "
        f"    --out-path {output_model} "
        f"    --hf-model-name {hf_model_name} "
        f"    --precision {dtype} "
        f"    --max-shard-size 10GB "
        f"    {extra_arguments} "
    )
    return cmd


def get_hf_to_trtllm_cmd(
    input_model, output_model, model_type, hf_model_name, dtype, num_gpus, num_nodes, extra_arguments
):
    dtype = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }[dtype]
    script = "hf_to_trtllm" if model_type == "llama" else "hf_to_trtllm_qwen"
    cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"export HF_TOKEN={get_token()} && "
        f"cd /nemo_run/code && "
        f"python -m nemo_skills.conversion.{script} "
        f"    --model_dir {input_model} "
        f"    --output_dir {output_model}-tmp "
        f"    --dtype {dtype} "
        f"    --tp_size {num_gpus} "
        f"    --pp_size {num_nodes} && "
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


def get_hf_to_nemo_cmd(
    input_model, output_model, model_type, hf_model_name, dtype, num_gpus, num_nodes, extra_arguments
):
    # Check if the model_type is "nemo"
    
    script = "hf_to_nemo" if model_type == "llama" else "hf_to_nemo_qwen"
    cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"export HF_TOKEN={get_token()} && "
        f"cd /nemo_run/code && "
        f"python -m nemo_skills.conversion.{script} " # Use the appropriate command based on the model_type
        f"    --in-path {input_model} "
        f"    --out-path {output_model} "
        f"    --hf-model-name {hf_model_name} "
        f"    --precision {dtype} "
        f"    {extra_arguments} "
    )

    return cmd


class SupportedTypes(str, Enum):
    llama = "llama"
    qwen = "qwen"


class SupportedFormatsTo(str, Enum):
    nemo = "nemo"
    hf = "hf"
    trtllm = "trtllm"


class SupportedFormatsFrom(str, Enum):
    nemo = "nemo"
    hf = "hf"


class SupportedDtypes(str, Enum):
    bf16 = "bf16"
    fp16 = "fp16"
    fp32 = "fp32"


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def convert(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    input_model: str = typer.Option(...),
    model_type: SupportedTypes = typer.Option("llama", help="Type of the model"),
    output_model: str = typer.Option(..., help="Where to put the final model"),
    convert_from: SupportedFormatsFrom = typer.Option(..., help="Format of the input model"),
    convert_to: SupportedFormatsTo = typer.Option(..., help="Format of the output model"),
    hf_model_name: str = typer.Option(None, help="Name of the model on Hugging Face Hub to convert to/from"),
    dtype: SupportedDtypes = typer.Option("bf16", help="Data type"),
    expname: str = typer.Option("conversion", help="NeMo-Run experiment name"),
    num_nodes: int = typer.Option(1),
    num_gpus: int = typer.Option(...),
    partition: str = typer.Option(
        None, help="Can specify if need interactive jobs or a specific non-default partition"
    ),
    run_after: str = typer.Option(
        None,
        help="Can specify an expname that needs to be completed before this one starts (will use as slurm dependency)",
    ),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    log_dir: str = typer.Option(None, help="Can specify a custom location for slurm logs. "),
):
    """Convert a checkpoint from one format to another.

    All extra arguments are passed directly to the underlying conversion script (see their docs).
    """
    setup_logging(disable_hydra_logs=False)
    extra_arguments = f'{" ".join(ctx.args)}'
    LOG.info("Starting conversion job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    try:
        model_type = model_type.value
        convert_from = convert_from.value
        convert_to = convert_to.value
        dtype = dtype.value
    except AttributeError:
        pass

    # TODO: add support for qwen nemo conversion
    if model_type == "qwen":
        if convert_from == "nemo":
            raise ValueError("NeMo conversion for Qwen models from nemo to hf formate is not supported yet")

    # TODO: add support for conversion from NeMo to trtllm using nemo.export (need to test thoroughly)
    if convert_from == "nemo" and convert_to == "trtllm":
        raise ValueError("Conversion from NeMo to TensorRT LLM is not supported directly. Convert to HF first.")

    if convert_to != "trtllm" and hf_model_name is None:
        raise ValueError("--hf_model_name is required")

    cluster_config = get_cluster_config(cluster, config_dir)
    check_if_mounted(cluster_config, input_model)
    check_if_mounted(cluster_config, output_model)
    if log_dir:
        check_if_mounted(cluster_config, log_dir)
    else:
        log_dir = str(Path(output_model) / "conversion-logs")

    conversion_cmd_map = {
        ("nemo", "hf"): get_nemo_to_hf_cmd,
        ("hf", "nemo"): get_hf_to_nemo_cmd,
        ("hf", "trtllm"): get_hf_to_trtllm_cmd,
    }
    container_map = {
        ("nemo", "hf"): cluster_config["containers"]["nemo"],
        ("hf", "nemo"): cluster_config["containers"]["nemo"],
        ("hf", "trtllm"): cluster_config["containers"]["trtllm"],
    }
    conversion_cmd = conversion_cmd_map[(convert_from, convert_to)](
        input_model=input_model,
        output_model=output_model,
        model_type=model_type,
        hf_model_name=hf_model_name,
        dtype=dtype,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        extra_arguments=extra_arguments,
    )
    with run.Experiment(expname) as exp:
        LOG.info("Launching task with command %s", conversion_cmd)
        add_task(
            exp,
            cmd=conversion_cmd,
            task_name=f'conversion-{convert_from}-{convert_to}',
            log_dir=log_dir,
            container=container_map[(convert_from, convert_to)],
            num_gpus=num_gpus,
            num_nodes=1,  # always running on a single node, might need to change that in the future
            num_tasks=1,
            cluster_config=cluster_config,
            partition=partition,
            run_after=run_after,
        )
        run_exp(exp, cluster_config)


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()
