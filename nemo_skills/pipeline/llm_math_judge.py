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


def get_judge_cmd(input_files, extra_arguments=""):
    return f'python -m nemo_skills.inference.llm_math_judge ++input_files={input_files} {extra_arguments}'


if __name__ == "__main__":
    setup_logging(disable_hydra_logs=False)
    parser = ArgumentParser(usage="TODO")
    wrapper_args = parser.add_argument_group('wrapper arguments')
    wrapper_args.add_argument("--config_dir", default=None, help="Path to the cluster_configs dir")
    wrapper_args.add_argument("--log_dir", required=False, help="Can specify a custom location for slurm logs")
    wrapper_args.add_argument("--cluster", required=True, help="One of the configs inside cluster_configs")
    wrapper_args.add_argument(
        "--input_files",
        required=True,
        nargs="+",
        help="Can also specify multiple glob patterns, like output-rs*.jsonl. Will add judgement field to each file",
    )
    wrapper_args.add_argument("--expname", default="llm-math-judge", help="Nemo run experiment name")
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
        choices=('nemo', 'trtllm', 'vllm', 'openai'),
        default='trtllm',
        help="Type of the server to start. This parameter is ignored if server_address is specified.",
    )
    wrapper_args.add_argument("--server_gpus", type=int, required=False)
    wrapper_args.add_argument(
        "--server_nodes",
        type=int,
        default=1,
        help="Number of nodes required for hosting LLM server.",
    )
    parser.add_argument("--server_args", default="", help="Any extra arguments to pass to the server.")
    # TODO: support this
    # wrapper_args.add_argument(
    #     "--num_jobs",
    #     type=int,
    #     default=-1,
    #     help="Will launch this many separate jobs and split the benchmarks across them. "
    #     "Set -1 to run each benchmark / random seed as a separate job.",
    # )
    wrapper_args.add_argument(
        "--partition",
        required=False,
        help="Can specify if need interactive jobs or a specific non-default partition",
    )
    wrapper_args.add_argument(
        "--run_after",
        required=False,
        help="Can specify an expname that needs to be completed before this one starts (will use as slurm dependency)",
    )

    args, unknown = parser.parse_known_args()

    extra_arguments = f'{" ".join(unknown)}'

    cluster_config = get_cluster_config(args.cluster, args.config_dir)
    for input_file in args.input_files:
        check_if_mounted(cluster_config, input_file)
    if args.log_dir:
        check_if_mounted(cluster_config, args.log_dir)
    args.input_files = f'"{" ".join(args.input_files)}"'

    if args.server_address is None:  # we need to host the model
        assert args.server_gpus is not None, "Need to specify server_gpus if hosting the model"
        args.server_address = "localhost:5000"

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
        add_task(
            exp,
            cmd=get_generation_command(
                server_address=args.server_address,
                generation_commands=get_judge_cmd(args.input_files, extra_arguments),
            ),
            task_name="llm-math-judge",
            log_dir=args.log_dir,
            container=cluster_config["containers"]["nemo-skills"],
            cluster_config=cluster_config,
            partition=args.partition,
            server_config=server_config,
            with_sandbox=True,
            run_after=args.run_after,
        )
        run_exp(exp, cluster_config)
