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

import os
from argparse import ArgumentParser
from datetime import datetime

import nemo_run as run
from huggingface_hub import get_token

from nemo_skills.pipeline import add_task, check_if_mounted, get_cluster_config, run_exp
from nemo_skills.utils import setup_logging


def get_training_cmd(
    cluster_config,
    partition,
    config_name,
    config_path,
    nemo_model,
    output_dir,
    training_data,
    validation_data,
    num_gpus,
    num_nodes,
    expname,
    training_algo,
    disable_wandb,
    wandb_project,
    extra_arguments,
):
    if training_algo == "dpo" and config_name is None:
        config_name = "dpo_config"

    if training_algo == "sft" and config_name is None:
        config_name = "sft_config"

    if validation_data is None:
        validation_data = training_data

    if training_algo == "dpo":
        extra_arguments = (
            f" ++model.data.data_prefix.train='[{training_data}]' "
            f" ++model.data.data_prefix.validation='[{validation_data}]' "
            f" ++model.data.data_prefix.test='[{validation_data}]' "
            f" pretrained_checkpoint.restore_from_path={nemo_model} " + extra_arguments
        )
    else:
        extra_arguments = (
            f" ++model.data.train_ds.file_path='{training_data}' "
            f" ++model.data.validation_ds.file_path='{validation_data}' "
            f" model.restore_from_path={nemo_model} " + extra_arguments
        )

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
        f"    exp_manager.explicit_log_dir={output_dir}/training "
        f"    exp_manager.exp_dir={output_dir}/training "
        f"    ++exp_manager.max_time_per_run={timeout} "
        f"    {extra_arguments} "
    )
    return cmd


def get_avg_checkpoints_cmd(nemo_model, output_dir, average_steps):
    cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"cd /nemo_run/code && "
        f"python nemo_skills/finetuning/average_checkpoints.py "
        f"    --untarred_nemo_folder {nemo_model} "
        f"    --name_prefix=model "
        f"    --checkpoint_dir={output_dir}/training/checkpoints {average_steps} &&"
        f"mv {output_dir}/training/checkpoints/model-averaged.nemo {output_dir} "
    )
    return cmd


if __name__ == "__main__":
    setup_logging(disable_hydra_logs=False)
    parser = ArgumentParser()
    # by default we are using a shared project
    parser.add_argument("--config_folder", default=None, help="Path to the cluster_configs folder")
    parser.add_argument("--cluster", required=True, help="One of the configs inside cluster_configs")
    # TODO: maybe not required and reuse expname in that case?
    parser.add_argument("--output_dir", required=True, help="Where to put results")
    parser.add_argument("--expname", required=True, help="Experiment name")
    parser.add_argument("--nemo_model", required=True)
    parser.add_argument("--training_data", required=True)
    # TODO: this needs to be fixed in nemo-aligner
    parser.add_argument(
        "--validation_data", required=False, help="Will default to the training data, since we can't disable it"
    )
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_gpus", type=int)
    parser.add_argument(
        "--num_training_jobs", type=int, default=1, help="Set to 0 if you only want to convert the model."
    )
    parser.add_argument("--training_algo", default="sft", choices=["sft", "dpo"])
    # have to be handled explicitly since hydra requires these to be first arguments
    parser.add_argument("--config-name", "-cn", required=False, help="If not specified will use (sft/dpo)_config")
    parser.add_argument("--config-path", "-cp", default='/nemo_run/code/nemo_skills/finetuning/')
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
    parser.add_argument(
        "--average_steps",
        nargs="+",
        type=int,
        help="List of checkpoint steps to average. If not specified, will average all.",
    )
    parser.add_argument(
        "--run_after",
        required=False,
        help="Can specify an expname that needs to be completed before this one starts (will use as slurm dependency)",
    )

    args, unknown = parser.parse_known_args()
    extra_arguments = f'{" ".join(unknown)}'

    if not args.output_dir.startswith("/"):
        raise ValueError("output_dir must be referenced in a mounted location (mounts section in the config file)")

    cluster_config = get_cluster_config(args.cluster, args.config_folder)
    check_if_mounted(cluster_config, args.output_dir)
    check_if_mounted(cluster_config, args.nemo_model)
    check_if_mounted(cluster_config, args.training_data)
    if args.validation_data:
        check_if_mounted(cluster_config, args.validation_data)

    train_cmd = get_training_cmd(
        cluster_config=cluster_config,
        partition=args.partition,
        config_name=args.config_name,
        config_path=args.config_path,
        nemo_model=args.nemo_model,
        output_dir=args.output_dir,
        training_data=args.training_data,
        validation_data=args.validation_data,
        num_gpus=args.num_gpus,
        num_nodes=args.num_nodes,
        expname=args.expname,
        training_algo=args.training_algo,
        disable_wandb=args.disable_wandb,
        wandb_project=args.wandb_project,
        extra_arguments=extra_arguments,
    )

    with run.Experiment(args.expname) as exp:
        for job_id in range(args.num_training_jobs):
            add_task(
                exp,
                cmd=train_cmd,
                task_name=f'{args.training_algo}',  # f'{args.training_algo}-{job_id}',
                container=cluster_config["containers"]["nemo"],
                num_gpus=args.num_gpus,
                num_nodes=args.num_nodes,
                num_tasks=args.num_gpus if cluster_config["executor"] == "slurm" else 1,
                cluster_config=cluster_config,
                partition=args.partition,
                with_sandbox=args.with_sandbox,
                run_after=args.run_after,
            )

        cmd = get_avg_checkpoints_cmd(
            nemo_model=args.nemo_model,
            output_dir=args.output_dir,
            average_steps=f"--steps {' '.join(map(str, args.average_steps))} " if args.average_steps else "",
        )
        add_task(
            exp,
            cmd=cmd,
            task_name="prepare-eval",
            container=cluster_config["containers"]['nemo'],
            cluster_config=cluster_config,
            partition=args.partition,
            num_nodes=1,
            num_tasks=1,
            num_gpus=args.num_gpus,
            run_after=args.run_after,
        )

        run_exp(exp, cluster_config, sequential=True)
        # exp.dryrun()
