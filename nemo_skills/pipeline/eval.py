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

from argparse import ArgumentParser

import nemo_run as run

from nemo_skills.evaluation.settings import EXTRA_EVAL_ARGS, EXTRA_GENERATION_ARGS
from nemo_skills.pipeline import add_task, check_if_mounted, get_cluster_config, get_generation_command, run_exp
from nemo_skills.utils import setup_logging


def get_greedy_cmd(benchmark, output_dir, output_name='output-greedy.jsonl', extra_eval_args="", extra_arguments=""):
    extra_eval_args = f"{EXTRA_EVAL_ARGS.get(benchmark, '')} {extra_eval_args}"
    extra_arguments = f"{EXTRA_GENERATION_ARGS.get(benchmark, '')} {extra_arguments}"
    cmd = (
        f'echo "Evaluating benchmark {benchmark}" && '
        f'python -m nemo_skills.inference.generate '
        f'    ++dataset={benchmark} '
        f'    ++output_file={output_dir}/eval-results/{benchmark}/{output_name} '
        f'    {extra_arguments} && '
        f'python -m nemo_skills.evaluation.evaluate_results '
        f'    ++input_files={output_dir}/eval-results/{benchmark}/{output_name} {extra_eval_args}'
    )
    return cmd


def get_sampling_cmd(benchmark, random_seed, extra_eval_args="", extra_arguments=""):
    extra_arguments = f" inference.random_seed={random_seed} inference.temperature=0.7 {extra_arguments}"
    return get_greedy_cmd(
        benchmark,
        output_name=f"output-rs{random_seed}.jsonl",
        extra_eval_args=extra_eval_args,
        extra_arguments=extra_arguments,
    )


if __name__ == "__main__":
    setup_logging(disable_hydra_logs=False)
    parser = ArgumentParser(usage="TODO")
    wrapper_args = parser.add_argument_group('wrapper arguments')
    wrapper_args.add_argument("--config_folder", default=None, help="Path to the cluster_configs folder")
    wrapper_args.add_argument("--cluster", required=True, help="One of the configs inside cluster_configs")
    wrapper_args.add_argument("--output_dir", required=True, help="Where to put results")
    wrapper_args.add_argument("--expname", default="eval", help="Nemo run experiment name")
    wrapper_args.add_argument("--model", required=False, help="Path to the model or model name in API.")
    # TODO: should all this be inside a single dictionary config?
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
    wrapper_args.add_argument("--server_gpus", type=int, required=False)
    wrapper_args.add_argument(
        "--server_nodes",
        type=int,
        default=1,
        help="Number of nodes required for hosting LLM server.",
    )
    wrapper_args.add_argument("--starting_seed", type=int, default=0)
    wrapper_args.add_argument(
        "--benchmarks",
        nargs="+",
        help="Need to be in a format <benchmark>:<num samples for majority voting>. "
        "Use <benchmark>:0 to only run greedy decoding.",
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
    wrapper_args.add_argument(
        "--run_after",
        required=False,
        help="Can specify an expname that needs to be completed before this one starts (will use as slurm dependency)",
    )

    args, unknown = parser.parse_known_args()

    extra_arguments = f'{" ".join(unknown)}'

    cluster_config = get_cluster_config(args.cluster, args.config_folder)
    check_if_mounted(cluster_config, args.output_dir)

    if args.server_address is None:  # we need to host the model
        assert args.server_gpus is not None, "Need to specify server_gpus if hosting the model"
        args.server_address = "localhost:5000"
        check_if_mounted(cluster_config, args.model)
        server_config = {
            "model_path": args.model,
            "server_type": args.server_type,
            "num_gpus": args.server_gpus,
            "num_nodes": args.server_nodes,
        }
        extra_arguments += f" ++server.server_type={args.server_type} "
    else:  # model is hosted elsewhere
        server_config = None
        extra_arguments += (
            f" ++server.server_type={args.server_type} "
            f" ++server.base_url={args.server_address} "
            f" ++server.model={args.model} "
        )

    # if benchmarks are specified, only run those
    BENCHMARKS = {k: int(v) for k, v in [b.split(":") for b in args.benchmarks]}

    eval_cmds = [
        get_greedy_cmd(
            benchmark, args.output_dir, extra_eval_args=args.extra_eval_args, extra_arguments=extra_arguments
        )
        for benchmark in BENCHMARKS.keys()
    ]
    eval_cmds += [
        get_sampling_cmd(
            benchmark, args.output_dir, rs, extra_eval_args=args.extra_eval_args, extra_arguments=extra_arguments
        )
        for benchmark, rs_num in BENCHMARKS.items()
        for rs in range(args.starting_seed, args.starting_seed + rs_num)
    ]
    if args.num_jobs == -1:
        args.num_jobs = len(eval_cmds)

    # splitting eval cmds equally across num_jobs nodes
    eval_cmds = [" && ".join(eval_cmds[i :: args.num_jobs]) for i in range(args.num_jobs)]

    with run.Experiment(args.expname) as exp:
        for idx, eval_cmd in enumerate(eval_cmds):
            add_task(
                exp,
                cmd=get_generation_command(server_address=args.server_address, generation_commands=eval_cmd),
                # TODO: has to be the same currently to reuse the code, need a fix in nemo.run
                task_name="eval",  # f'eval-{idx}',
                container=cluster_config["containers"]["nemo-skills"],
                cluster_config=cluster_config,
                partition=args.partition,
                server_config=server_config,
                with_sandbox=True,
                run_after=args.run_after,
            )
        run_exp(exp, cluster_config)
        # exp.dryrun()
