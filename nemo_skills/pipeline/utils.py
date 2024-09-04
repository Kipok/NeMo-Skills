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
import os
from pathlib import Path

import nemo_run as run
from nemo_run.core.execution.slurm import JobPaths

LOG = logging.getLogger(__file__)


# TODO: should we fill up this vars not at import time?
GENERATION_CMD = (
    "export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
    "cd /nemo_run/code && "
    # might be required if we are not hosting server ourselves
    f"export NVIDIA_API_KEY={os.getenv('NVIDIA_API_KEY', '')} && "
    f"export OPENAI_API_KEY={os.getenv('OPENAI_API_KEY', '')} && "
    # this will try to handshake in a loop and unblock when the server responds
    "echo 'Waiting for the server to start' && "
    "while [ $(curl -X PUT {server_address} >/dev/null 2>&1; echo $?) -ne 0 ]; do sleep 3; done"
    # will run in a single task always (no need to check mpi env vars)
    "{generation_commands}"
)


def get_server_command(server_type: str, num_gpus: int, num_nodes: int, model_path: str, cluster_config: dict):
    num_tasks = num_gpus
    if server_type == 'nemo':
        server_start_cmd = (
            f"python /nemo_run/code/nemo_skills/inference/server/serve_nemo.py gpt_model_file={model_path} "
            f"trainer.devices={num_gpus} "
            f"trainer.num_nodes={num_nodes} "
            f"tensor_model_parallel_size={num_gpus} "
            f"pipeline_model_parallel_size={num_nodes} "
        )
        # somehow on slurm nemo needs multiple tasks, but locally only 1
        if cluster_config["executor"] == "local":
            num_tasks = 1

    elif server_type == 'vllm':
        server_start_cmd = (
            f"NUM_GPUS={num_gpus} bash /nemo_run/code/nemo_skills/inference/server/serve_vllm.sh "
            f"{model_path} {os.path.basename(model_path)} 0 openai 5000"
        )

        if os.environ.get("MAX_SEQ_LEN", None) is not None:
            server_start_cmd = f"export MAX_SEQ_LEN={os.environ['MAX_SEQ_LEN']} && {server_start_cmd}"

        num_tasks = 1
    else:
        # adding sleep to ensure the logs file exists
        # need this flag for stable Nemotron-4-340B deployment
        server_start_cmd = (
            f"FORCE_NCCL_ALL_REDUCE_STRATEGY=1 python /nemo_run/code/nemo_skills/inference/server/serve_trt.py "
            f"--model_path {model_path}"
        )
        num_tasks = num_gpus

    server_cmd = (
        "nvidia-smi && "
        "cd /nemo_run/code && "
        "export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"export HF_TOKEN={os.getenv('HF_TOKEN', '')} && "
        f"{server_start_cmd} "
    )
    return server_cmd, num_tasks


def get_sandox_command():
    return "/entrypoint.sh && /start.sh"


def get_sandbox_executor(executor, cluster_config):
    sandbox_executor = executor.clone()
    sandbox_executor.container_image = cluster_config["containers"]["sandbox"]
    sandbox_executor.container_mounts = []
    sandbox_executor.srun_args += [f"--ntasks={sandbox_executor.nodes}", "--overlap", '--wait=1']
    # sandbox_executor.job_paths_cls = SandboxJobPaths
    return sandbox_executor


def get_client_executor(executor, cluster_config):
    client_executor = executor.clone()
    # TODO: change to "main", which is nemo-skills only container
    client_executor.container_image = cluster_config["containers"]["tensorrt_llm"]
    client_executor.container_mounts = []
    client_executor.srun_args += [f"--ntasks=1", "--nodes=1", '--wait=1']
    # client_executor.job_paths_cls = SandboxJobPaths
    return client_executor


# def log_path()
class MainJobPaths(JobPaths):
    @property
    def stdout(self) -> Path:
        return Path(self.folder / "slurm-logs" / "sbatch.txt")

    @property
    def srun_stdout(self) -> Path:
        return Path(self.folder / "slurm-logs" / "job_logs.txt")


def get_server_executor(
    cluster_config,
    container,
    num_nodes,
    tasks_per_node,
    gpus_per_node,
    partition=None,
):
    mounts = mounts or []
    partition = partition or cluster_config.get("partition")
    if 'timeouts' not in cluster_config:
        timeout = "10000:00:00:00"
    else:
        timeout = cluster_config["timeouts"][partition]

    # return run.LocalExecutor(
    #     container_image=container,
    #     container_mounts=cluster_config.get('mounts', []),
    #     ntasks_per_node=tasks_per_node,
    #     gpus_per_node=gpus_per_node,
    # )

    return run.SlurmExecutor(
        account=cluster_config["account"],
        partition=partition,
        nodes=num_nodes,
        ntasks_per_node=tasks_per_node,
        tunnel=run.SSHTunnel(**cluster_config["ssh_tunnel"]),
        container_image=container,
        container_mounts=cluster_config.get('mounts', []),
        time=timeout,
        packager=run.GitArchivePackager(include_pattern='nemo_skills/dataset/**/*.jsonl'),
        gpus_per_node=gpus_per_node,
        job_name_prefix=cluster_config["job_name_prefix"],
        srun_args=["--no-container-mount-home", "--overlap", "--mpi=pmix", '--wait=10'],
        exclusive=True,
        mem=0,
        job_paths_cls=MainJobPaths,
        wait_time_for_group_job=0.01,
        monitor_group_job_wait_time=20,
    )


def add_task(
    exp,
    cmd,
    task_name,
    cluster_config,
    num_tasks,
    num_gpus,
    num_nodes,
    container,
    partition=None,
    with_sandbox=False,
    server_config=None,
):
    commands = [cmd]
    executors = [...]
    if server_config is not None:
        server_cmd, num_server_tasks = get_server_command(**server_config, cluster_config=cluster_config)
        if 'container' not in server_config:
            server_container = cluster_config["containers"][server_config['server_type']]
        server_executor = get_server_executor(
            cluster_config=cluster_config,
            container=server_container,
            num_nodes=server_config['num_nodes'],
            tasks_per_node=num_server_tasks,
            gpus_per_node=server_config['num_gpus'],
            partition=partition,
        )

    commands = [cmd.replace("$", "\\$") for cmd in commands]
    if with_sandbox:
        sandbox_executor = get_sandbox_executor(main_executor, cluster_config)
        client_executor = get_client_executor(main_executor, cluster_config)
        exp.add(
            [run.Script(inline=cmds[0]), run.Script(inline=get_sandox_command()), run.Script(inline=cmds[1])],
            executor=[main_executor, sandbox_executor, client_executor],
            name=task_name,
        )
    else:
        exp.add(run.Script(inline=cmds[0]), executor=main_executor, name=task_name)
