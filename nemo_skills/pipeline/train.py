# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from datetime import datetime
from enum import Enum

import nemo_run as run
import typer
from huggingface_hub import get_token

from nemo_skills.pipeline import add_task, check_if_mounted, get_cluster_config, run_exp
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.utils import setup_logging

LOG = logging.getLogger(__file__)


class TrainingAlgo(str, Enum):
    sft = "sft"
    dpo = "dpo"


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
    timeout = f'00:{time_diff.seconds // 3600:02d}:{(time_diff.seconds % 3600) // 60:02d}:{time_diff.seconds % 60:02d}'

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
        f"python -m nemo_skills.training.start_{training_algo} "
        f" --config-name={config_name} --config-path={config_path} "
        f" ++model.tensor_model_parallel_size={num_gpus} "
        f" trainer.devices={num_gpus} "
        f" trainer.num_nodes={num_nodes} "
        f" {logging_params} "
        f" exp_manager.name={expname} "
        f" exp_manager.explicit_log_dir={output_dir}/training "
        f" exp_manager.exp_dir={output_dir}/training "
        f" ++exp_manager.max_time_per_run={timeout} "
        f" {extra_arguments} "
    )

    return cmd


def get_avg_checkpoints_cmd(nemo_model, output_dir, final_nemo_path, average_steps):
    name = "model" + ("-".join(average_steps[len('--steps ') :].split()) if average_steps else '') + "-averaged"
    cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"cd /nemo_run/code && "
        f"python -m nemo_skills.training.average_checkpoints "
        f" --untarred_nemo_dir {nemo_model} "
        f" --name_prefix=model "
        f" --checkpoint_dir={output_dir}/training/checkpoints {average_steps} && "
        f"mkdir -p {os.path.dirname(final_nemo_path)} && "
        f"mv {output_dir}/training/checkpoints/{name} {final_nemo_path} "
    )
    return cmd


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def train(
    ctx: typer.Context,
    config_dir: str = typer.Option(None, help="Path to the cluster_configs dir"),
    cluster: str = typer.Option(..., help="One of the configs inside cluster_configs"),
    output_dir: str = typer.Option(..., help="Where to put results"),
    final_nemo_path: str = typer.Option(None, help="Where to put the final checkpoint"),
    expname: str = typer.Option(..., help="Experiment name"),
    nemo_model: str = typer.Option(..., help="Path to the NeMo model"),
    training_data: str = typer.Option(None, help="Path to the training data"),
    validation_data: str = typer.Option(None, help="Path to the validation data"),
    num_nodes: int = typer.Option(1, help="Number of nodes"),
    num_gpus: int = typer.Option(..., help="Number of GPUs"),
    num_training_jobs: int = typer.Option(1, help="Number of training jobs"),
    training_algo: TrainingAlgo = typer.Option(TrainingAlgo.sft, help="Training algorithm"),
    config_name: str = typer.Option(None, help="Config name"),
    config_path: str = typer.Option('/nemo_run/code/nemo_skills/training/', help="Config path"),
    wandb_project: str = typer.Option("nemo-skills", help="Weights & Biases project name"),
    disable_wandb: bool = typer.Option(False, help="Disable wandb logging"),
    chat_format: bool = typer.Option(False, help="Use chat format for SFT data"),
    with_sandbox: bool = typer.Option(False, help="If sandbox is required for code generation"),
    partition: str = typer.Option(None, help="Specify partition for jobs"),
    average_steps: list[int] = typer.Option(None, help="List of checkpoint steps to average"),
    run_after: str = typer.Option(None, help="Experiment to run after"),
):
    """Train (SFT or DPO) an LLM model."""
    setup_logging(disable_hydra_logs=False)
    extra_arguments = f'{" ".join(ctx.args)}'
    LOG.info("Starting training job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    cluster_config = get_cluster_config(cluster, config_dir)
    check_if_mounted(cluster_config, output_dir)
    check_if_mounted(cluster_config, nemo_model)

    if num_training_jobs > 0:
        if training_data is None:
            raise ValueError("training_data is required when num_training_jobs > 0")
        check_if_mounted(cluster_config, training_data)

    if not final_nemo_path:
        final_nemo_path = f"{output_dir}/model-averaged-nemo"
    check_if_mounted(cluster_config, final_nemo_path)

    if validation_data:
        check_if_mounted(cluster_config, validation_data)

    train_cmd = get_training_cmd(
        cluster_config=cluster_config,
        partition=partition,
        config_name=config_name,
        config_path=config_path,
        nemo_model=nemo_model,
        output_dir=output_dir,
        training_data=training_data,
        validation_data=validation_data,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        expname=expname,
        training_algo=training_algo,
        disable_wandb=disable_wandb,
        wandb_project=wandb_project,
        extra_arguments="",
    )

    with run.Experiment(expname) as exp:
        for job_id in range(num_training_jobs):
            add_task(
                exp,
                cmd=train_cmd,
                task_name=f'{training_algo}-{job_id}',
                log_dir=f"{output_dir}/training-logs",
                container=cluster_config["containers"]["nemo"],
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                num_tasks=num_gpus if cluster_config["executor"] == "slurm" else 1,
                cluster_config=cluster_config,
                partition=partition,
                with_sandbox=with_sandbox,
                run_after=run_after,
            )

        cmd = get_avg_checkpoints_cmd(
            nemo_model=nemo_model,
            output_dir=output_dir,
            final_nemo_path=final_nemo_path,
            average_steps=f"--steps {' '.join(map(str, average_steps))} " if average_steps else "",
        )

        add_task(
            exp,
            cmd=cmd,
            task_name="prepare-eval",
            log_dir=f"{output_dir}/prepare-eval-logs",
            container=cluster_config["containers"]['nemo'],
            cluster_config=cluster_config,
            partition=partition,
            num_nodes=1,
            num_tasks=1,
            num_gpus=num_gpus,
            run_after=run_after,
        )

        run_exp(exp, cluster_config, sequential=True)


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()
