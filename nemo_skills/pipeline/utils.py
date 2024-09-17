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
import tarfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import nemo_run as run
import yaml
from huggingface_hub import get_token
from nemo_run.config import NEMORUN_HOME
from nemo_run.core.execution.docker import DockerExecutor
from nemo_run.core.execution.slurm import SlurmJobDetails
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer

LOG = logging.getLogger(__file__)


def check_if_mounted(cluster_config, path_to_check):
    """Will check that path_to_check is referenced inside one of the mounts."""
    for mount in cluster_config.get('mounts', []):
        if path_to_check.startswith(mount.split(":")[1]):
            return
    raise ValueError(f"The path '{path_to_check}' is not mounted. Check cluster config.")


def get_unmounted_path(cluster_config, path):
    """Will return the path on the filesystem before it's mounted."""
    if path is None:
        return None
    for mount in cluster_config.get('mounts', []):
        if path.startswith(mount.split(":")[1]):
            return mount.split(":")[0] + path[len(mount.split(":")[1]) :]
    raise ValueError(f"The path '{path}' is not mounted. Check cluster config.")


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
            f"python -m nemo_skills.inference.server.serve_nemo "
            f"    gpt_model_file={model_path} "
            f"    trainer.devices={num_gpus} "
            f"    trainer.num_nodes={num_nodes} "
            f"    tensor_model_parallel_size={num_gpus} "
            f"    pipeline_model_parallel_size={num_nodes} "
        )
        # somehow on slurm nemo needs multiple tasks, but locally only 1
        if cluster_config["executor"] == "local":
            num_tasks = 1

    elif server_type == 'vllm':
        server_start_cmd = (
            f"NUM_GPUS={num_gpus} bash nemo_skills/inference/server/serve_vllm.sh "
            f"{model_path} self-hosted-model 0 openai 5000"
        )

        if os.environ.get("MAX_SEQ_LEN", None) is not None:
            server_start_cmd = f"export MAX_SEQ_LEN={os.environ['MAX_SEQ_LEN']} && {server_start_cmd}"

        num_tasks = 1
    else:
        # adding sleep to ensure the logs file exists
        # need this flag for stable Nemotron-4-340B deployment
        server_start_cmd = (
            f"FORCE_NCCL_ALL_REDUCE_STRATEGY=1 python -m nemo_skills.inference.server.serve_trt "
            f"    --model_path {model_path}"
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


@dataclass(kw_only=True)
class CustomJobDetails(SlurmJobDetails):
    log_prefix: str = "main"

    @property
    def stdout(self) -> Path:
        return Path(self.folder) / f"{self.log_prefix}_sbatch.log"

    @property
    def srun_stdout(self) -> Path:
        return Path(self.folder) / f"{self.log_prefix}_srun.log"

    @property
    def stderr(self) -> Path:
        return Path(self.folder) / f"{self.log_prefix}_sbatch.log"

    @property
    def srun_stderr(self) -> Path:
        return Path(self.folder) / f"{self.log_prefix}_srun.log"


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


def cluster_download(tunnel, remote_dir, local_dir):
    remote_dir = remote_dir.rstrip('/')
    remote_tar = f"{remote_dir}.tar.gz"
    local_tar = os.path.join(local_dir, os.path.basename(remote_tar))

    # Create tarball of the remote directory
    tunnel.run(
        f"cd {os.path.dirname(remote_dir)} && tar -czf {remote_tar} {os.path.basename(remote_dir)}",
        hide=True,
    )

    # Download the tarball to the local directory
    tunnel.get(remote_tar, local_tar)

    # Extract the tarball locally
    os.makedirs(local_dir, exist_ok=True)
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=local_dir)

    # Clean up the tarball from the remote server
    tunnel.run(f'rm {remote_tar}', hide=True)

    # Clean up the local tarball
    os.remove(local_tar)


@lru_cache
def get_executor(
    cluster_config,
    container,
    num_nodes,
    tasks_per_node,
    gpus_per_node,
    job_name,
    log_folder,
    log_prefix: str = "main",
    mounts=None,
    partition=None,
    dependencies=None,
):
    config_mounts = cluster_config.get('mounts', [])
    mounts = mounts or config_mounts
    packager = run.GitArchivePackager(include_pattern='nemo_skills/dataset/**/*.jsonl', check_uncommitted_changes=True)
    if cluster_config["executor"] == "local":
        if num_nodes > 1:
            raise ValueError("Local executor does not support multi-node execution")
        return DockerExecutor(
            container_image=container,
            packager=packager,
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
        packager=packager,
        gpus_per_node=gpus_per_node if not cluster_config.get("disable_gpus_per_node", False) else None,
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
        job_details=CustomJobDetails(
            job_name=cluster_config.get("job_name_prefix", "") + job_name,
            folder=get_unmounted_path(cluster_config, log_folder),
            log_prefix=log_prefix,
        ),
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
    log_folder=None,
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
            job_name=task_name,
            log_folder=log_folder,
            log_prefix="server",
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
                job_name=task_name,
                log_folder=log_folder,
                log_prefix="main",
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
            job_name=task_name,
            log_folder=log_folder,
            log_prefix="sandbox",
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
