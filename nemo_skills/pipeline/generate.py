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

from nemo_skills.pipeline import add_task, check_if_mounted, get_cluster_config, get_generation_command, run_exp
from nemo_skills.utils import setup_logging


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


if __name__ == "__main__":
    setup_logging(disable_hydra_logs=False)
    parser = ArgumentParser()
    parser.add_argument("--config_dir", default=None, help="Path to the cluster_configs dir")
    parser.add_argument("--cluster", required=True, help="One of the configs inside cluster_configs")
    parser.add_argument("--output_dir", required=True, help="Where to put results")
    parser.add_argument("--expname", default="generate", help="Nemo run experiment name")
    parser.add_argument("--model", required=False, help="Path to the model or model name in API.")
    # TODO: should all this be inside a single dictionary config?
    parser.add_argument(
        "--server_address",
        required=False,
        help="Use ip:port for self-hosted models or the API url if using model providers.",
    )
    parser.add_argument(
        "--server_type",
        choices=('nemo', 'trtllm', 'vllm', 'openai'),
        default='trtllm',
        help="Type of the server to start.",
    )
    parser.add_argument("--server_gpus", type=int, required=False)
    parser.add_argument(
        "--server_nodes",
        type=int,
        default=1,
        help="Number of nodes required for hosting LLM server.",
    )
    parser.add_argument("--server_args", default="", help="Any extra arguments to pass to the server.")
    parser.add_argument(
        "--dependent_jobs",
        type=int,
        default=0,
        help="Specify this to launch that number of dependent jobs. Useful for large datasets, "
        "where you're not able to process everything before slurm timeout.",
    )
    parser.add_argument(
        "--num_random_seeds",
        type=int,
        required=False,
        help="Specify if want to run many generations with high temperature for the same input.",
    )
    parser.add_argument("--starting_seed", type=int, default=0)
    parser.add_argument(
        "--partition",
        required=False,
        help="Can specify if need interactive jobs or a specific non-default partition",
    )
    parser.add_argument(
        "--eval_args",
        required=False,
        help="Specify if need to run nemo_skills/evaluation/evaluate_results.py on the generation outputs.",
    )
    parser.add_argument(
        "--run_after",
        required=False,
        help="Can specify an expname that needs to be completed before this one starts (will use as slurm dependency)",
    )
    args, unknown = parser.parse_known_args()

    extra_arguments = f'{" ".join(unknown)}'

    cluster_config = get_cluster_config(args.cluster, args.config_dir)
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
            "server_args": args.server_args,
        }
        extra_arguments += f" ++server.server_type={args.server_type} "
    else:  # model is hosted elsewhere
        server_config = None
        extra_arguments += (
            f" ++server.server_type={args.server_type} "
            f" ++server.base_url={args.server_address} "
            f" ++server.model={args.model} "
        )

    with run.Experiment(args.expname) as exp:
        if args.num_random_seeds:
            for seed in range(args.starting_seed, args.starting_seed + args.num_random_seeds):
                # TODO: needs support on nemorun side
                assert args.dependent_jobs == 0
                cmd = get_cmd(
                    random_seed=seed,
                    output_dir=args.output_dir,
                    extra_arguments=extra_arguments,
                    eval_args=args.eval_args,
                )
                add_task(
                    exp,
                    cmd=get_generation_command(server_address=args.server_address, generation_commands=cmd),
                    task_name=f'generate-rs{seed}',
                    log_dir=f"{args.output_dir}/generation-logs",
                    container=cluster_config["containers"]["nemo-skills"],
                    cluster_config=cluster_config,
                    partition=args.partition,
                    server_config=server_config,
                    with_sandbox=True,
                    run_after=args.run_after,
                )
        else:
            # TODO: needs support on nemorun side
            assert args.dependent_jobs == 0
            cmd = get_cmd(
                random_seed=None,
                output_dir=args.output_dir,
                extra_arguments=extra_arguments,
                eval_args=args.eval_args,
            )
            add_task(
                exp,
                cmd=get_generation_command(server_address=args.server_address, generation_commands=cmd),
                task_name="generate",
                log_dir=f"{args.output_dir}/generation-logs",
                container=cluster_config["containers"]["nemo-skills"],
                cluster_config=cluster_config,
                partition=args.partition,
                server_config=server_config,
                with_sandbox=True,
                run_after=args.run_after,
            )
        run_exp(exp, cluster_config)
