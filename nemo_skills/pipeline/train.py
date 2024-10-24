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

import abc
import logging
import os
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import TypeVar

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
    rm = "rm"


class BashCommand:
    def __init__(self, base_command):
        self.commands = [base_command]

    def add_command(self, command):
        """Add a new command to be executed after the current commands using '&&'."""
        self.commands.append(command)
        return self

    def extend_command(self, *args):
        """Extend the last command with additional arguments."""
        self.commands[-1] += ' ' + ' '.join(args)
        return self

    def __str__(self):
        """Return the full command string with commands joined by '&&'."""
        return ' && '.join(self.commands)


@dataclass
class TrainingCmdBuilder(abc.ABC):

    config_name: str
    config_path: str
    nemo_model: str
    output_dir: str
    training_data: str
    validation_data: str
    num_gpus: int
    num_nodes: int
    expname: str
    training_algo: str
    disable_wandb: bool
    wandb_project: str
    extra_arguments: str
    timeout: str

    def base_command(self):
        return BashCommand(
            f"export WANDB_API_KEY={os.getenv('WANDB_API_KEY', '')} && "
            f"export HF_TOKEN={get_token()} && "
            f"export HYDRA_FULL_ERROR=1 && "
            f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
            f"cd /nemo_run/code && "
            f"echo 'Starting training' ")

    def add_parallel_device_configs(self, cmd: BashCommand, **kwargs):
        return cmd.extend_command(
            f"++model.tensor_model_parallel_size={self.num_gpus} "
            f"trainer.devices={self.num_gpus} "
            f"trainer.num_nodes={self.num_nodes} "
        )

    def add_logging_params(self, cmd: BashCommand, **kwargs) -> BashCommand:
        if not self.disable_wandb:
            if os.getenv('WANDB_API_KEY') is None:
                raise ValueError("WANDB_API_KEY is not set. Use --disable_wandb to disable wandb logging")
            logging_params = (
                f"exp_manager.create_wandb_logger=True "
                f"exp_manager.wandb_logger_kwargs.name={self.expname} "
                f"exp_manager.wandb_logger_kwargs.project={self.wandb_project} "
                f"+exp_manager.wandb_logger_kwargs.id={self.expname} "
                f"+exp_manager.wandb_logger_kwargs.resume=True "
            )
        else:
            logging_params = "exp_manager.create_wandb_logger=False +exp_manager.create_tensorboard_logger=True"

        logging_params = BashCommand(logging_params).extend_command(
                f"    exp_manager.name={self.expname} "
                f"    exp_manager.explicit_log_dir={self.output_dir}/training "
                f"    exp_manager.exp_dir={self.output_dir}/training "
                f"    ++exp_manager.max_time_per_run={self.timeout} "
        )
        return cmd.extend_command(logging_params)

    @abc.abstractmethod
    def add_training_script(self, cmd: BashCommand, **kwargs) -> BashCommand:
        pass

    @abc.abstractmethod
    def add_config(self, cmd: BashCommand, **kwargs) -> BashCommand:
        pass

    @abc.abstractmethod
    def add_extra_arguments(self, cmd: BashCommand, **kwargs) -> BashCommand:
        pass


CmdBuilderType = TypeVar("CmdBuilderType", bound=TrainingCmdBuilder)

class TrainingCmdDirector:

    def __init__(self, builder: TrainingCmdBuilder):
        self.builder = builder

    def build_command(self) -> str:
        command = self.builder.base_command()
        command = self.builder.add_training_script(command)
        command = self.builder.add_config(command)
        command = self.builder.add_parallel_device_configs(command)
        command = self.builder.add_logging_params(command)
        command = self.builder.add_extra_arguments(command)
        return str(command)


class SFTTrainingCmdBuilder(TrainingCmdBuilder):
    def add_training_script(self, cmd: BashCommand, **kwargs):
        new_cmd = cmd.add_command(
           f"python -m nemo_skills.training.start_{self.training_algo}  "
        )
        return new_cmd

    def add_config(self, cmd: BashCommand, **kwargs):
        return cmd.extend_command(
            f"--config-name=sft_config --config-path={self.config_path}"
        )

    def add_extra_arguments(self, cmd: BashCommand, **kwargs):
        return cmd.extend_command(
            f" ++model.data.train_ds.file_path='{self.training_data}' "
            f" ++model.data.validation_ds.file_path='{self.validation_data}' "
            f" model.restore_from_path={self.nemo_model} " + self.extra_arguments
        )

class DPOTrainingCmdBuilder(TrainingCmdBuilder):
    def add_training_script(self, cmd: BashCommand, **kwargs):
        new_cmd = cmd.add_command(
           f"python -m nemo_skills.training.start_{self.training_algo}  "
        )
        return new_cmd

    def add_config(self, cmd: BashCommand, **kwargs):
        return cmd.extend_command(
            f"--config-name=dpo_config --config-path={self.config_path}"
        )

    def add_extra_arguments(self, cmd: BashCommand, **kwargs):
        return cmd.extend_command(
            f" ++model.data.data_prefix.train='[{self.training_data}]' "
            f" ++model.data.data_prefix.validation='[{self.validation_data}]' "
            f" ++model.data.data_prefix.test='[{self.validation_data}]' "
            f" pretrained_checkpoint.restore_from_path={self.nemo_model} " + self.extra_arguments
        )


class RMTrainingCmdBuilder(TrainingCmdBuilder):
    def add_training_script(self, cmd, **kwargs):
        return cmd.add_command(
            "python -u /opt/NeMo-Aligner/examples/nlp/gpt/train_reward_model.py"
        )

    def add_config(self, cmd, **kwargs):
        return cmd

    def add_extra_arguments(self, cmd, **kwargs):
        return cmd.extend_command(
            f" ++model.data.data_prefix.train='[{self.training_data}]' "
            f" ++model.data.data_prefix.validation='[{self.validation_data}]' "
            f" ++model.data.data_prefix.test='[{self.validation_data}]' "
            f" pretrained_checkpoint.restore_from_path={self.nemo_model} "
            f" {self.extra_arguments}"
        )

builder_classes = {
    TrainingAlgo.sft: SFTTrainingCmdBuilder,
    TrainingAlgo.dpo: DPOTrainingCmdBuilder,
    TrainingAlgo.rm: RMTrainingCmdBuilder,
}

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
    if validation_data is None:
        validation_data = training_data

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

    builder_cls: CmdBuilderType = builder_classes.get(training_algo)
    if builder_cls is None:
        raise ValueError(f"Unsupported training algorithm: {training_algo}")

    builder = builder_cls(
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
        extra_arguments=extra_arguments,
        timeout=timeout,
    )
    cmd = TrainingCmdDirector(builder).build_command()

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
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
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
    with_sandbox: bool = typer.Option(False, help="If sandbox is required for code generation"),
    partition: str = typer.Option(None, help="Specify partition for jobs"),
    average_steps: str = typer.Option(
        None, help="List of commas separated checkpoint steps to average. E.g 1000,5000"
    ),
    run_after: str = typer.Option(None, help="Experiment to run after"),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    log_dir: str = typer.Option(None, help="Can specify a custom location for slurm logs. "),
):
    """Train (SFT or DPO) an LLM model.

    All extra arguments are passed directly to the training script
    (need to be prefixed with ++, since NeMo uses Hydra).
    """
    setup_logging(disable_hydra_logs=False)
    extra_arguments = f'{" ".join(ctx.args)}'
    LOG.info("Starting training job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    try:
        training_algo = training_algo.value
    except AttributeError:
        pass

    cluster_config = get_cluster_config(cluster, config_dir)
    check_if_mounted(cluster_config, output_dir)
    check_if_mounted(cluster_config, nemo_model)
    if log_dir:
        check_if_mounted(cluster_config, log_dir)
    else:
        log_dir = output_dir

    if num_training_jobs > 0:
        if training_data is None:
            raise ValueError("training_data is required when num_training_jobs > 0")
        check_if_mounted(cluster_config, training_data)

    if not final_nemo_path:
        final_nemo_path = f"{output_dir}/model-averaged-nemo"
    check_if_mounted(cluster_config, final_nemo_path)

    if validation_data:
        check_if_mounted(cluster_config, validation_data)

    if " " in str(average_steps):
        raise ValueError("average steps should be separated with commas")

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
        extra_arguments=extra_arguments,
    )

    with run.Experiment(expname) as exp:
        prev_task = None
        for job_id in range(num_training_jobs):
            prev_task = add_task(
                exp,
                cmd=train_cmd,
                task_name=f'{training_algo}-{job_id}',
                log_dir=f"{log_dir}/training-logs",
                container=cluster_config["containers"]["nemo"],
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                num_tasks=num_gpus if cluster_config["executor"] == "slurm" else 1,
                cluster_config=cluster_config,
                partition=partition,
                with_sandbox=with_sandbox,
                run_after=run_after,
                task_dependencies=[prev_task] if prev_task is not None else None,
            )

        cmd = get_avg_checkpoints_cmd(
            nemo_model=nemo_model,
            output_dir=output_dir,
            final_nemo_path=final_nemo_path,
            average_steps=f"--steps {' '.join(average_steps.split(','))} " if average_steps else "",
        )

        add_task(
            exp,
            cmd=cmd,
            task_name="prepare-eval",
            log_dir=f"{log_dir}/prepare-eval-logs",
            container=cluster_config["containers"]['nemo'],
            cluster_config=cluster_config,
            partition=partition,
            num_nodes=1,
            num_tasks=1,
            num_gpus=num_gpus,
            run_after=run_after,
            task_dependencies=[prev_task] if prev_task is not None else None,
        )

        # explicitly setting sequential to False since we set dependencies directly
        run_exp(exp, cluster_config, sequential=False)


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()
