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

import nemo_run as run
import typer

from nemo_skills.pipeline import add_task, check_if_mounted, get_cluster_config, get_generation_command, run_exp
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.utils import setup_logging

LOG = logging.getLogger(__file__)


def get_cmd(output_dir, extra_arguments, random_seed=None, eval_args=None):
    if random_seed is not None:
        output_file = f"{output_dir}/generation/output-rs{random_seed}.jsonl"
    else:
        output_file = f"{output_dir}/generation/output.jsonl"
    cmd = f"python -m nemo_skills.inference.generate ++skip_filled=True ++output_file={output_file} "
    if random_seed is not None:
        cmd += (
            f"    ++inference.random_seed={random_seed} "
            f"    ++inference.temperature=1.0 "
            f"    ++inference.top_k=0 "
            f"    ++inference.top_p=0.95 "
        )
    cmd += f" {extra_arguments} "
    if eval_args:
        cmd += (
            f" && python -m nemo_skills.evaluation.evaluate_results "
            f"    ++input_files={output_file} "
            f"    {eval_args} "
        )
    return cmd


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def prepare_sft_data(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    output_dir: str = typer.Option(..., help="Where to put results"),
    expname: str = typer.Option("prepare_sft_data", help="Nemo run experiment name"),
    model: str = typer.Option(None, help="Path to the model or model name in API"),
    partition: str = typer.Option(
        None, help="Can specify if need interactive jobs or a specific non-default partition"
    ),
    run_after: str = typer.Option(
        None, help="Can specify an expname that needs to be completed before this one starts"
    ),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    log_dir: str = typer.Option(None, help="Can specify a custom location for slurm logs. "),
):
    """Generate LLM completions for a given input file.

    Run `python -m nemo_skills.inference.generate --help` for other supported arguments
    (need to be prefixed with ++, since we use Hydra for that script).
    """
    setup_logging(disable_hydra_logs=False)
    extra_arguments = f'{" ".join(ctx.args)}'

    cluster_config = get_cluster_config(cluster, config_dir)
    check_if_mounted(cluster_config, output_dir)
    if log_dir:
        check_if_mounted(cluster_config, log_dir)
    else:
        log_dir = f"{output_dir}/generation-logs"

    with run.Experiment(expname) as exp:
        # prev_tasks = None
        _ = add_task(
            exp,
            cmd="echo 'foo' && pip install transformers", #get_generation_command(server_address=server_address, generation_commands=cmd),
            task_name="prepare_sft_data",
            log_dir=log_dir,
            container=cluster_config["containers"]["nemo-skills"],
            cluster_config=cluster_config,
            partition=partition, # TODO make it cpu by default?
            run_after=run_after,
            extra_srun_args=(
                '--cpu-bind=cores',
                '--mem-bind=local',
                '--cpu-freq=highm1',
                '--cpus-per-task=45',
                '--mem=170GB', # TODO figure out which of these options are necessary and which are not
                # TODO also figure out if any of these need to be configurable
            )
            # task_dependencies=prev_tasks,
        )
        # prev_tasks = [new_task]
        run_exp(exp, cluster_config)


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()

