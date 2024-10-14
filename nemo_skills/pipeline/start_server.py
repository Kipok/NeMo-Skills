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
from enum import Enum

import nemo_run as run
import typer

from nemo_skills.pipeline import add_task, check_if_mounted, get_cluster_config
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.utils import setup_logging


class SupportedServers(str, Enum):
    trtllm = "trtllm"
    vllm = "vllm"
    nemo = "nemo"


@app.command()
@typer_unpacker
def start_server(
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    model: str = typer.Option(..., help="Path to the model"),
    server_type: SupportedServers = typer.Option('trtllm', help="Type of server to use"),
    server_gpus: int = typer.Option(..., help="Number of GPUs to use for hosting the model"),
    server_nodes: int = typer.Option(1, help="Number of nodes to use for hosting the model"),
    server_args: str = typer.Option("", help="Additional arguments for the server"),
    partition: str = typer.Option(None, help="Cluster partition to use"),
    with_sandbox: bool = typer.Option(
        False, help="Starts a sandbox (set this flag if model supports calling Python interpreter)"
    ),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    log_dir: str = typer.Option(
        None,
        help="Can specify a custom location for slurm logs. "
        "If not specified, will be inside `ssh_tunnel.job_dir` part of your cluster config.",
    ),
):
    """Self-host a model server."""
    setup_logging(disable_hydra_logs=False)

    cluster_config = get_cluster_config(cluster, config_dir)

    try:
        server_type = server_type.value
    except AttributeError:
        pass

    if log_dir:
        check_if_mounted(cluster_config, log_dir)

    server_config = {
        "model_path": model,
        "server_type": server_type,
        "num_gpus": server_gpus,
        "num_nodes": server_nodes,
        "server_args": server_args,
    }

    with run.Experiment("server") as exp:
        add_task(
            exp,
            cmd="",  # not running anything except the server
            task_name='server',
            log_dir=log_dir,
            container=cluster_config["containers"]["nemo-skills"],
            cluster_config=cluster_config,
            partition=partition,
            server_config=server_config,
            with_sandbox=with_sandbox,
        )
        # we don't want to detach in this case even on slurm, so not using run_exp
        exp.run(detach=False, tail_logs=True)
        # TODO: seems like not being killed? If nemorun doesn't do this, we can catch the signal and kill the server ourselves


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()
