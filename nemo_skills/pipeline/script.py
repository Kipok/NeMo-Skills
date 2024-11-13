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
from pathlib import Path

import nemo_run as run
import typer

from nemo_skills.pipeline import add_task, check_if_mounted, get_cluster_config, get_generation_command, run_exp
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.utils import setup_logging

LOG = logging.getLogger(__file__)


def get_cmd(module, extra_arguments):
    cmd = "pip install transformers && "
    cmd += f"time python -m {module} "
    cmd += f" {extra_arguments} "
    cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"cd /nemo_run/code && "
        # might be required if we are not hosting server ourselves
        f"export NVIDIA_API_KEY={os.getenv('NVIDIA_API_KEY', '')} && "
        f"export OPENAI_API_KEY={os.getenv('OPENAI_API_KEY', '')} && "
        f"{cmd}"
    )
    return cmd


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def script(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    expname: str = typer.Option("script", help="Nemo run experiment name"),
    partition: str = typer.Option(
        None, help="Can specify if need interactive jobs or a specific non-default partition"
    ),
    module: str = typer.Option(help=(
        "Searches sys.path in the NeMo-Skills Container for the named module and runs the "
        "corresponding .py file as a script."
    )),
    num_gpus: int = typer.Option(1, help="Number of GPUs to use"),
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
    if log_dir:
        check_if_mounted(cluster_config, log_dir)
    else:
        log_dir = str(Path('.').absolute() / 'script-logs')

    with run.Experiment(expname) as exp:
        _ = add_task(
            exp,
            cmd=get_cmd(module=module, extra_arguments=extra_arguments),
            task_name="script",
            log_dir=log_dir,
            container=cluster_config["containers"]["nemo-skills"],
            cluster_config=cluster_config,
            partition=partition,
            run_after=run_after,
            num_gpus=num_gpus,
        )
        run_exp(exp, cluster_config)


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()

