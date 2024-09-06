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

import datetime
import os
from argparse import ArgumentParser
from pathlib import Path

import nemo_run as run
import yaml
from huggingface_hub import get_token

from nemo_skills.pipeline import add_task, run_exp
from nemo_skills.utils import setup_logging


def get_training_cmd(
    cluster_config,
    partition,
    config_name,
    config_path,
    num_gpus,
    num_nodes,
    expname,
    training_algo,
    chat_format,
    validation_dataset,
    disable_wandb,
    wandb_project,
    extra_arguments,
):
    if training_algo == "dpo" and config_name is None:
        config_name = "dpo_config"

    if training_algo == "sft" and config_name is None:
        config_name = "sft_config"

    if training_algo == "dpo" and chat_format:
        raise ValueError("DPO does not support chat format")

    if chat_format:
        extra_arguments = (
            " ++model.data.chat=True "
            f" model.data.validation_ds.file_path=/code/datasets/{validation_dataset}/validation-sft-chat.jsonl "
        ) + extra_arguments
    else:
        if training_algo == "sft":
            extra_arguments = (
                " ++model.data.chat=False "
                f" model.data.validation_ds.file_path=/code/datasets/{validation_dataset}/validation-sft.jsonl "
            ) + extra_arguments

    if training_algo == "dpo":
        # TODO: for DPO currently user has to be explicit about validation/test sets
        # ++model.data.data_prefix.train='[/data/paired_all_openmath.jsonl]' \
        # ++model.data.data_prefix.validation='[/data/paired_val_openmath_train.jsonl]' \
        # ++model.data.data_prefix.test='[/data/paired_val_openmath_train.jsonl]'

        extra_arguments = " pretrained_checkpoint.restore_from_path=/nemo_model " + extra_arguments
    else:
        extra_arguments = " model.restore_from_path=/nemo_model " + extra_arguments

    if 'timeouts' not in cluster_config:
        timeout = "10000:00:00:00"
    else:
        timeout = cluster_config["timeouts"][partition or cluster_config["partition"]]
        # subtracting 15 minutes to account for the time it takes to save the model
        # the format expected by nemo is days:hours:minutes:seconds
        time_diff = datetime.strptime(timeout, "%H:%M:%S") - datetime.strptime("00:15:00", "%H:%M:%S")
        timeout = (
            f'00:{time_diff.seconds // 3600:02d}:{(time_diff.seconds % 3600) // 60:02d}:{time_diff.seconds % 60:02d}'
        )

    if not disable_wandb:
        if os.getenv('WANDB_API_KEY') is None:
            raise ValueError("WANDB_API_KEY is not set. Use --disable_wandb to disable wandb logging")
        logging_params = (
            f"exp_manager.create_wandb_logger=True "
            f"exp_manager.wandb_logger_kwargs.name={expname} "
            f"exp_manager.wandb_logger_kwargs.project={wandb_project} "
            f"+exp_manager.wandb_logger_kwargs.id={expname} "
            f"+exp_manager.wandb_logger_kwargs.resume=True "
        )
    else:
        logging_params = "exp_manager.create_wandb_logger=False +exp_manager.create_tensorboard_logger=True"

    cmd = (
        f"export WANDB_API_KEY={os.getenv('WANDB_API_KEY', '')} && "
        f"export HF_TOKEN={get_token()} && "
        f"export HYDRA_FULL_ERROR=1 && "
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"cd /nemo_run/code && "
        f"echo 'Starting training' && "
        f"python nemo_skills/finetuning/start_{training_algo}.py "
        f"    --config-name={config_name} --config-path={config_path} "
        f"    ++model.tensor_model_parallel_size={num_gpus} "
        f"    trainer.devices={num_gpus} "
        f"    trainer.num_nodes={num_nodes} "
        f"    {logging_params} "
        f"    exp_manager.name={expname} "
        f"    exp_manager.explicit_log_dir=/results "
        f"    exp_manager.exp_dir=/results "
        f"    ++exp_manager.max_time_per_run={timeout} "
        f"    {extra_arguments} "
    )
    return cmd


if __name__ == "__main__":
    setup_logging(disable_hydra_logs=False)
    parser = ArgumentParser()
    # by default we are using a shared project
    parser.add_argument("--cluster", required=True, help="One of the configs inside cluster_configs")
    parser.add_argument("--expname", required=True, help="Experiment name")
    parser.add_argument("--nemo_model", required=True)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_gpus", type=int)
    parser.add_argument("--dependent_jobs", type=int, default=0)
    parser.add_argument("--training_algo", default="sft", choices=["sft", "dpo"])
    # have to be handled explicitly since hydra requires these to be first arguments
    parser.add_argument("--config-name", "-cn", required=False, help="If not specified will use (sft/dpo)_config")
    parser.add_argument("--config-path", "-cp", default='/nemo_run/code/nemo_skills/finetuning/')
    parser.add_argument(
        "--validation_dataset",
        default="gsm8k",
        # TODO: how to disable it by default?
        help="Validation dataset to use. Make sure it exists inside datasets folder",
    )
    parser.add_argument("--wandb_project", default="nemo-skills")
    parser.add_argument(
        "--disable_wandb", action="store_true", help="Disable wandb logging and use tensorboard instead"
    )
    parser.add_argument("--chat_format", action="store_true", help="Use chat format for SFT data")
    parser.add_argument("--with_sandbox", action="store_true", help="If sandbox is required for code generation")
    parser.add_argument(
        "--partition",
        required=False,
        help="Can specify if need interactive jobs or a specific non-default partition",
    )

    args, unknown = parser.parse_known_args()
    extra_arguments = f'{" ".join(unknown)}'

    with open(Path(__file__).parents[2] / 'cluster_configs' / f'{args.cluster}.yaml', "rt", encoding="utf-8") as fin:
        cluster_config = yaml.safe_load(fin)

    train_cmd = get_training_cmd(
        cluster_config=cluster_config,
        partition=args.partition,
        config_name=args.config_name,
        config_path=args.config_path,
        num_gpus=args.num_gpus,
        num_nodes=args.num_nodes,
        expname=args.expname,
        training_algo=args.training_algo,
        chat_format=args.chat_format,
        validation_dataset=args.validation_dataset,
        disable_wandb=args.disable_wandb,
        wandb_project=args.wandb_project,
        extra_arguments=extra_arguments,
    )

    with run.Experiment(args.expname) as exp:
        for job_id in range(args.dependent_jobs + 1):
            # TODO: doesn't work currently since folder is not shared
            #       either need to use different cluster path or wait for nemorun to support it
            add_task(
                exp,
                cmd=train_cmd,
                task_name=f'{args.training_algo}-{job_id}',
                container=cluster_config["containers"]["nemo"],
                num_gpus=args.num_gpus,
                num_nodes=args.num_nodes,
                num_tasks=args.num_gpus if cluster_config["executor"] == "slurm" else 1,
                cluster_config=cluster_config,
                partition=args.partition,
                with_sandbox=args.with_sandbox,
            )
        run_exp(exp, cluster_config)

    # TODO: add prepare eval here directly, not reason to keep it separate
    # TODO: instead let's create a --depends_on or --after flag to all scripts
    #    so that users can chain them together in any way they want.
    #    It's more flexible than trying to put everything inside a "pipeline"
