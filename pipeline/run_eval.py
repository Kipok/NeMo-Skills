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
import sys
from argparse import ArgumentParser
from pathlib import Path

import yaml

# adding nemo_skills to python path to avoid requiring installation
sys.path.append(str(Path(__file__).absolute().parents[1]))

from launcher import NEMO_SKILLS_CODE, WRAPPER_HELP, get_server_command, launch_job

try:
    from nemo_skills.inference.generate import HELP_MESSAGE
except (ImportError, TypeError):
    HELP_MESSAGE = """
TODO
"""
from nemo_skills.evaluation.settings import EXTRA_EVAL_ARGS, EXTRA_GENERATION_ARGS
from nemo_skills.utils import setup_logging

SCRIPT_HELP = """
TODO
"""


def get_greedy_cmd(benchmark, output_name='output-greedy.jsonl', extra_eval_args="", extra_arguments=""):
    extra_eval_args = f"{EXTRA_EVAL_ARGS.get(benchmark, '')} {extra_eval_args}"
    extra_arguments = f"{EXTRA_GENERATION_ARGS.get(benchmark, '')} {extra_arguments}"
    return f"""echo "Evaluating benchmark {benchmark}" && \
python nemo_skills/inference/generate.py \
    ++server.server_type={{server_type}} \
    ++dataset={benchmark} \
    ++output_file=/results/{benchmark}/{output_name} \
    {extra_arguments} && \
python nemo_skills/evaluation/evaluate_results.py \
    ++prediction_jsonl_files=/results/{benchmark}/{output_name} {extra_eval_args} && \
"""


def get_sampling_cmd(benchmark, random_seed, extra_eval_args="", extra_arguments=""):
    extra_arguments = f" inference.random_seed={random_seed} inference.temperature=0.7 {extra_arguments}"
    return get_greedy_cmd(
        benchmark,
        output_name=f"output-rs{random_seed}.jsonl",
        extra_eval_args=extra_eval_args,
        extra_arguments=extra_arguments,
    )


# TODO: can we use something generic instead of SLURM_PROCID?
CMD = (
    # boilerplate code to setup the environment
    "nvidia-smi && "
    "cd /code && "
    "export PYTHONPATH=$PYTHONPATH:/code && "
    "export HF_TOKEN={HF_TOKEN} && "
    "export NVIDIA_API_KEY={NVIDIA_API_KEY} && "
    "export OPENAI_API_KEY={OPENAI_API_KEY} && "
    # launching the server and waiting for it to start
    # server_start_cmd = "sleep infinity" if host/port are not specified (so server is running elsewhere)
    # but of course in that case this script itself is meant to be run locally and not on a GPU cluster
    "if [ $SLURM_PROCID -eq 0 ]; then "
    #    running in background to be able to communicate with the server
    "    {server_start_cmd} & "
    "    echo 'Waiting for the server to start' && "
    "    while [ $(curl -X PUT {server_address} >/dev/null 2>&1; echo $?) -ne 0 ]; do sleep 3; done && "
    "    {eval_cmds} "
    #    command to kill the server if it was started by this script
    "    {server_end_cmd}; "
    "else "
    #    this is a blocking call to keep non-zero ranks from exiting and killing the job
    "    {server_start_cmd}; "
    "fi "
)


if __name__ == "__main__":
    setup_logging(disable_hydra_logs=False)
    parser = ArgumentParser(usage=WRAPPER_HELP + '\n\n' + SCRIPT_HELP + '\n\nscript arguments:\n\n' + HELP_MESSAGE)
    wrapper_args = parser.add_argument_group('wrapper arguments')
    wrapper_args.add_argument("--cluster", required=True, help="One of the configs inside cluster_configs")
    wrapper_args.add_argument("--model", required=False, help="Path to the model or model name in API.")
    wrapper_args.add_argument(
        "--server_address",
        required=False,
        help="Use ip:port for self-hosted models or the API url if using model providers.",
    )
    # TODO: let's make it not needed - we just need to unify our api calls
    wrapper_args.add_argument(
        "--server_type",
        choices=('nemo', 'tensorrt_llm', 'vllm', 'openai'),
        default='tensorrt_llm',
        help="Type of the server to start. This parameter is ignored if server_address is specified.",
    )
    wrapper_args.add_argument("--output_dir", required=True)
    wrapper_args.add_argument("--num_gpus", type=int, required=True)
    wrapper_args.add_argument("--starting_seed", type=int, default=0)
    wrapper_args.add_argument(
        "--benchmarks",
        nargs="+",
        help="Need to be in a format <benchmark>:<num samples for majority voting>. "
        "Use <benchmark>:0 to only run greedy decoding.",
    )
    wrapper_args.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes required for hosting LLM server.",
    )
    wrapper_args.add_argument(
        "--num_jobs",
        type=int,
        default=-1,
        help="Will launch this many separate jobs and split the benchmarks across them. "
        "Set -1 to run each benchmark / random seed as a separate job.",
    )
    wrapper_args.add_argument(
        "--partition",
        required=False,
        help="Can specify if need interactive jobs or a specific non-default partition",
    )
    wrapper_args.add_argument(
        "--extra_eval_args",
        default="",
        help="Any extra arguments to pass to nemo_skills/evaluation/evaluate_results.py",
    )

    args, unknown = parser.parse_known_args()

    extra_arguments = f'{" ".join(unknown)}'

    with open(Path(__file__).parents[1] / 'cluster_configs' / f'{args.cluster}.yaml', "rt", encoding="utf-8") as fin:
        cluster_config = yaml.safe_load(fin)

    args.output_dir = Path(args.output_dir).absolute()

    # model is hosted elsewhere
    if args.server_address is not None:
        server_start_cmd = "sleep infinity"
        server_end_cmd = "echo 'done'"
        job_name = "eval-remote"
        mounts = f"{NEMO_SKILLS_CODE}:/code,{args.output_dir}:/results"
        num_tasks = 1
        if args.server_type == "openai":
            extra_arguments += f" ++server.base_url={args.server_address} ++server.model={args.model}"
        # TODO: run without container if remote server?
        container = cluster_config["containers"]['nemo']
    else:
        model_path = Path(args.model).absolute()
        server_start_cmd, num_tasks = get_server_command(
            args.server_type, args.num_gpus, args.num_nodes, model_path.name
        )
        server_end_cmd = "pkill -f nemo_skills/inference/server"
        job_name = f"eval-{model_path.name}"
        # also mounting the model in this case
        mounts = f"{NEMO_SKILLS_CODE}:/code,{args.output_dir}:/results,{model_path}:/model"
        args.server_address = "localhost:5000"
        container = cluster_config["containers"][args.server_type]

    format_dict = {
        "server_start_cmd": server_start_cmd,
        "server_end_cmd": server_end_cmd,
        "server_type": args.server_type,
        "server_address": args.server_address,
        "NEMO_SKILLS_CODE": NEMO_SKILLS_CODE,
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "NVIDIA_API_KEY": os.getenv("NVIDIA_API_KEY", ""),
    }

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    # if benchmarks are specified, only run those
    BENCHMARKS = {k: int(v) for k, v in [b.split(":") for b in args.benchmarks]}

    eval_cmds = [
        get_greedy_cmd(benchmark, extra_eval_args=args.extra_eval_args, extra_arguments=extra_arguments)
        for benchmark in BENCHMARKS.keys()
    ]
    eval_cmds += [
        get_sampling_cmd(benchmark, rs, extra_eval_args=args.extra_eval_args, extra_arguments=extra_arguments)
        for benchmark, rs_num in BENCHMARKS.items()
        for rs in range(args.starting_seed, args.starting_seed + rs_num)
    ]
    if args.num_jobs == -1:
        args.num_jobs = len(eval_cmds)

    # splitting eval cmds equally across num_jobs nodes
    eval_cmds = [" ".join(eval_cmds[i :: args.num_jobs]) for i in range(args.num_jobs)]

    for idx, eval_cmd in enumerate(eval_cmds):
        extra_sbatch_args = ["--parsable", f"--output={args.output_dir}/slurm_logs_eval{idx}.log"]
        launch_job(
            cluster_config=cluster_config,
            cmd=CMD.format(**format_dict, eval_cmds=eval_cmd.format(**format_dict)),
            num_nodes=args.num_nodes,
            tasks_per_node=num_tasks,
            gpus_per_node=args.num_gpus,
            job_name=job_name,
            container=container,
            mounts=mounts,
            partition=args.partition,
            with_sandbox=True,
            extra_sbatch_args=extra_sbatch_args,
        )
