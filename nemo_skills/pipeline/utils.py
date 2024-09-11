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

import json
import logging
import os
import shlex
import subprocess
from functools import lru_cache
from pathlib import Path

import nemo_run as run
import yaml
from huggingface_hub import get_token
from nemo_run.config import NEMORUN_HOME
from nemo_run.core.execution.docker import DockerExecutor
from nemo_run.core.execution.slurm import JobPaths
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer

LOG = logging.getLogger(__file__)


def check_if_mounted(cluster_config, path_to_check):
    """Will check that path_to_check is referenced inside one of the mounts."""
    for mount in cluster_config.get('mounts', []):
        if path_to_check.startswith(mount.split(":")[1]):
            return
    raise ValueError(f"The path '{path_to_check}' is not mounted. Check cluster config.")


# TODO: How this function is expected to work if we install nemo-skills as a package?
def check_uncommitted_changes(path: Path) -> bool:
    # Check for modified files
    cmd_modified = f"cd {shlex.quote(str(path))} && git diff --name-only nemo_skills"
    result_modified = subprocess.run(
        cmd_modified, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Check for staged files
    cmd_staged = f"cd {shlex.quote(str(path))} && git diff --name-only --cached nemo_skills"
    result_staged = subprocess.run(cmd_staged, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    changed_files = result_modified.stdout.strip().split("\n") + result_staged.stdout.strip().split("\n")
    changed_files = [f for f in changed_files if f]
    if changed_files:
        raise ValueError(
            f"There are uncommitted changes in the repository: {changed_files}. Please commit or stash them before running the experiment."
        )


def _get_latest_dir(path) -> str:
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    latest_dir = max(dirs, key=lambda d: os.path.getctime(os.path.join(path, d)))
    return os.path.join(path, latest_dir)


def get_exp_handles(expname):
    # TODO: remove this after we can properly use .from_title api
    parent_dir = os.path.join(NEMORUN_HOME, "experiments", expname)
    exp_dir = _get_latest_dir(parent_dir)

    with open(os.path.join(exp_dir, '_TASKS')) as f:
        serialized_jobs = json.load(f)

    serializer = ZlibJSONSerializer()
    handles = []
    for job in serialized_jobs:
        handles.append(serializer.deserialize(job[0]).handle)
    return handles


def get_generation_command(server_address, generation_commands):
    cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"cd /nemo_run/code && "
        # might be required if we are not hosting server ourselves
        f"export NVIDIA_API_KEY={os.getenv('NVIDIA_API_KEY', '')} && "
        f"export OPENAI_API_KEY={os.getenv('OPENAI_API_KEY', '')} && "
        # this will try to handshake in a loop and unblock when the server responds
        f"echo 'Waiting for the server to start' && "
        f"while [ $(curl -X PUT {server_address} >/dev/null 2>&1; echo $?) -ne 0 ]; do sleep 3; done && "
        # will run in a single task always (no need to check mpi env vars)
        f"{generation_commands}"
    )
    return cmd


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
            f"{model_path} self-hosted-model 0 openai 5000"
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
        f"nvidia-smi && "
        f"cd /nemo_run/code && "
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"export HF_TOKEN={get_token()} && "
        f"{server_start_cmd} "
    )
    return server_cmd, num_tasks


def get_sandox_command():
    return "/entrypoint.sh && /start.sh"


# def get_logs_cls(cluster_config, expname):
#     class MainJobPaths(JobPaths):
#         @property
#         def stdout(self) -> Path:
#             return Path(f"{cluster_config['workspace']}/{expname}" / "slurm-logs" / "sbatch.txt")

#         @property
#         def srun_stdout(self) -> Path:
#             return Path(f"{cluster_config['workspace']}/{expname}" / "slurm-logs" / "job_logs.txt")

#     return MainJobPaths


# class MainJobPaths(JobPaths):
#     @property
#     def stdout(self) -> Path:
#         return Path(self.folder / "slurm-logs" / "sbatch.txt")

#     @property
#     def srun_stdout(self) -> Path:
#         return Path(self.folder / "slurm-logs" / "job_logs.txt")


# a very hacky way to cache cluster config - is there a better way to do this?
class hashabledict(dict):
    def __hash__(self):
        return hash(frozenset(self))


def get_cluster_config(cluster, config_folder=None):
    if config_folder is None:
        config_folder = Path(__file__).parents[2] / 'cluster_configs'
    else:
        config_folder = Path(config_folder)

    with open(config_folder / f'{cluster}.yaml', "rt", encoding="utf-8") as fin:
        cluster_config = yaml.safe_load(fin)

    return hashabledict(cluster_config)


@lru_cache
def get_tunnel(cluster_config):
    return run.SSHTunnel(**cluster_config["ssh_tunnel"])


@lru_cache
def get_executor(
    cluster_config,
    container,
    num_nodes,
    tasks_per_node,
    gpus_per_node,
    mounts=None,
    partition=None,
    dependencies=None,
):
    config_mounts = cluster_config.get('mounts', [])
    mounts = mounts or config_mounts
    if cluster_config["executor"] == "local":
        if num_nodes > 1:
            raise ValueError("Local executor does not support multi-node execution")
        return DockerExecutor(
            container_image=container,
            packager=run.GitArchivePackager(include_pattern='nemo_skills/dataset/**/*.jsonl'),
            ipc_mode="host",
            volumes=mounts,
            ntasks_per_node=1,
            num_gpus=gpus_per_node,
            network="host",
            env_vars={"PYTHONUNBUFFERED": "1"},  # this makes sure logs are streamed right away
        )

    partition = partition or cluster_config.get("partition")
    if 'timeouts' not in cluster_config:
        timeout = "10000:00:00:00"
    else:
        timeout = cluster_config["timeouts"][partition]

    return run.SlurmExecutor(
        account=cluster_config["account"],
        partition=partition,
        nodes=num_nodes,
        ntasks_per_node=tasks_per_node,
        tunnel=get_tunnel(cluster_config),
        container_image=container,
        container_mounts=mounts,
        time=timeout,
        packager=run.GitArchivePackager(include_pattern='nemo_skills/dataset/**/*.jsonl'),
        gpus_per_node=gpus_per_node,
        job_name_prefix=cluster_config["job_name_prefix"],
        srun_args=[
            "--no-container-mount-home",
            "--overlap",
            "--mpi=pmix",
            '--wait=10',
            # we need to be explicit about this in srun as commands might need to run in parallel
            f"--ntasks={tasks_per_node * num_nodes}",
            f"--nodes={num_nodes}",
        ],
        # TODO: can we relax this to allow partial node allocation?
        exclusive=True,
        mem=0,
        # job_paths_cls=get_logs_cls(cluster_config, expname),
        # job_paths_cls=MainJobPaths,
        wait_time_for_group_job=0.01,
        monitor_group_job_wait_time=20,
        dependencies=dependencies,
    )


def add_task(
    exp,
    cmd,
    task_name,
    cluster_config,
    container,
    # TODO: these are good defaults for generation jobs, but probably not the best overall?
    num_tasks=1,
    num_gpus=1,
    num_nodes=1,
    partition=None,
    with_sandbox=False,
    server_config=None,
    run_after=None,
):
    if run_after is not None and cluster_config["executor"] == "slurm":
        dependencies = tuple(get_exp_handles(run_after))
    else:
        dependencies = None
    commands = []
    executors = []
    # assuming server always has the largest resources request, so it needs to go first
    if server_config is not None:
        server_cmd, num_server_tasks = get_server_command(**server_config, cluster_config=cluster_config)
        if 'container' not in server_config:
            server_container = cluster_config["containers"][server_config['server_type']]
        server_executor = get_executor(
            cluster_config=cluster_config,
            container=server_container,
            num_nodes=server_config['num_nodes'],
            tasks_per_node=num_server_tasks,
            gpus_per_node=server_config['num_gpus'],
            partition=partition,
            dependencies=dependencies,
        )
        if cluster_config["executor"] == "local" and num_server_tasks > 1:
            server_cmd = f"mpirun --allow-run-as-root -np {num_server_tasks} bash -c {shlex.quote(server_cmd)}"
        commands.append(server_cmd)
        executors.append(server_executor)

    # then goes the main task unless it's empty
    if cmd:
        if cluster_config["executor"] == "local" and num_tasks > 1:
            cmd = f"mpirun --allow-run-as-root -np {num_tasks} bash -c {shlex.quote(cmd)}"
        commands.append(cmd)
        executors.append(
            get_executor(
                cluster_config=cluster_config,
                container=container,
                num_nodes=num_nodes,
                tasks_per_node=num_tasks,
                gpus_per_node=num_gpus,
                partition=partition,
                dependencies=dependencies,
            )
        )

    # finally a sandbox if needed
    if with_sandbox:
        sandbox_executor = get_executor(
            cluster_config=cluster_config,
            container=cluster_config["containers"]["sandbox"],
            num_nodes=executors[0].nodes if cluster_config["executor"] == "slurm" else 1,
            tasks_per_node=1,
            gpus_per_node=num_gpus,
            partition=partition,
            mounts=tuple(),  # we don't want to mount anything
            dependencies=dependencies,
        )
        commands.append(get_sandox_command())
        executors.append(sandbox_executor)

    if len(commands) == 1:
        # to keep sbatch script simpler, we don't wrap in a list in this case
        exp.add(run.Script(inline=commands[0]), executor=executors[0], name=task_name)
    else:
        exp.add(
            [run.Script(inline=command) for command in commands],
            executor=executors,
            name=task_name,
        )


def run_exp(exp, cluster_config, sequential=False):
    if cluster_config['executor'] == 'local':
        # locally we are always running sequentially - does that need to be changed?
        exp.run(detach=False, tail_logs=True, sequential=True)
    else:
        exp.run(detach=True, sequential=sequential)
