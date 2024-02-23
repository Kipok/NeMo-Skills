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

import sys
from argparse import ArgumentParser
from pathlib import Path

# adding nemo_skills to python path to avoid requiring installation
sys.path.append(str(Path(__file__).absolute().parents[1]))

from launcher import CLUSTER_CONFIG, get_server_command, launch_job

from nemo_skills.inference.prompt.utils import examples_map, prompt_types
from nemo_skills.utils import setup_logging


def get_greedy_cmd(benchmark, output_name='output-greedy.jsonl', extra_arguments=""):
    return f"""echo "Evaluating benchmark {benchmark}" && \
python nemo_skills/inference/generate_solutions.py \
    server.server_type={{server_type}} \
    prompt.context_type=empty \
    +dataset={benchmark} \
    output_file=/results/{benchmark}/{output_name} \
    {extra_arguments} && \
python nemo_skills/evaluation/evaluate_results.py \
    prediction_jsonl_files=/results/{benchmark}/{output_name} && \
"""


def get_sampling_cmd(benchmark, random_seed, extra_arguments=""):
    extra_arguments = f" inference.random_seed={random_seed} inference.temperature=0.7 {extra_arguments}"
    return get_greedy_cmd(benchmark, output_name=f"output-rs{random_seed}.jsonl", extra_arguments=extra_arguments)


# default number of samples for majority voting
BENCHMARKS = {
    "gsm8k": 8,
    "math": 4,
}

SLURM_CMD = """
nvidia-smi && \
cd /code && \
export PYTHONPATH=$PYTHONPATH:/code && \
{server_start_cmd} && \
if [ $SLURM_LOCALID -eq 0 ]; then \
    echo "Waiting for the server to start" && \
    tail -n0 -f /tmp/server_logs.txt | sed '/Running on all addresses/ q' && \
    {eval_cmds} \
    kill %1; \
else \
    sleep infinity; \
fi \
"""

MOUNTS = "{NEMO_SKILLS_CODE}:/code,{model_path}:/model,{output_dir}:/results"
JOB_NAME = "eval-{model_name}"


if __name__ == "__main__":
    setup_logging(disable_hydra_logs=False)
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--server_type", choices=('nemo', 'tensorrt_llm'), default='tensorrt_llm')
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--starting_seed", type=int, default=0)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=[],
        help="Need to be in a format <benchmark>:<num samples for majority voting>. "
        "Use <benchmark>:0 to only run greedy decoding.",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=-1,
        help="Will parallelize across this number of nodes. Set -1 to run each decoding on a separate node.",
    )
    parser.add_argument(
        "--partition",
        required=False,
        help="Can specify if need interactive jobs or a specific non-default partition",
    )
    args, unknown = parser.parse_known_args()

    extra_arguments = f'{" ".join(unknown)}'

    args.model_path = Path(args.model_path).absolute()
    args.output_dir = Path(args.output_dir).absolute()

    server_start_cmd = get_server_command(args.server_type, args.num_gpus)

    format_dict = {
        "model_path": args.model_path,
        "model_name": args.model_path.name,
        "output_dir": args.output_dir,
        "num_gpus": args.num_gpus,
        "server_start_cmd": server_start_cmd,
        "server_type": args.server_type,
        "NEMO_SKILLS_CODE": str(Path(__file__).absolute().parents[1]),
    }

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    # if benchmarks are specified, only run those
    if args.benchmarks:
        BENCHMARKS = {k: int(v) for k, v in [b.split(":") for b in args.benchmarks]}

    eval_cmds = [get_greedy_cmd(benchmark, extra_arguments=extra_arguments) for benchmark in BENCHMARKS.keys()]
    eval_cmds += [
        get_sampling_cmd(benchmark, rs, extra_arguments=extra_arguments)
        for benchmark, rs_num in BENCHMARKS.items()
        for rs in range(args.starting_seed, args.starting_seed + rs_num)
    ]
    if args.num_nodes == -1:
        args.num_nodes = len(eval_cmds)

    # splitting eval cmds equally across num_nodes nodes
    eval_cmds = [" ".join(eval_cmds[i :: args.num_nodes]) for i in range(args.num_nodes)]

    num_tasks = format_dict["num_gpus"]
    # somehow on slurm nemo needs multiple tasks, but locally only 1
    if args.server_type == "nemo" and CLUSTER_CONFIG["cluster"] == "local":
        num_tasks = 1

    for idx, eval_cmd in enumerate(eval_cmds):
        extra_sbatch_args = ["--parsable", f"--output={args.output_dir}/slurm_logs_eval{idx}.log"]
        launch_job(
            cmd=SLURM_CMD.format(**format_dict, eval_cmds=eval_cmd.format(**format_dict)),
            num_nodes=1,
            tasks_per_node=num_tasks,
            gpus_per_node=format_dict["num_gpus"],
            job_name=JOB_NAME.format(**format_dict),
            container=CLUSTER_CONFIG["containers"][args.server_type],
            mounts=MOUNTS.format(**format_dict),
            partition=args.partition,
            with_sandbox=True,
            extra_sbatch_args=extra_sbatch_args,
        )
