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
from pathlib import Path

import nemo_run as run
import yaml

from nemo_skills.pipeline import add_task, get_generation_command, run_exp
from nemo_skills.utils import setup_logging


def get_cmd(random_seed, extra_arguments, extra_eval_args):
    cmd = (
        f"python nemo_skills/inference/generate.py "
        f"    skip_filled=True "
        f"    inference.random_seed={random_seed} "
        f"    inference.temperature=1.0 "
        f"    inference.top_k=0 "
        f"    inference.top_p=0.95 "
        f"    output_file=/exp/generation/output-rs{random_seed}.jsonl "
        f"    {extra_arguments} && "
        f"python nemo_skills/evaluation/evaluate_results.py "
        f"    prediction_jsonl_files=/exp/generation/output-rs{random_seed}.jsonl {extra_eval_args}"
    )
    return cmd


if __name__ == "__main__":
    setup_logging(disable_hydra_logs=False)
    parser = ArgumentParser()
    parser.add_argument("--cluster", required=True, help="One of the configs inside cluster_configs")
    parser.add_argument("--expname", required=True, help="Experiment name")
    parser.add_argument("--model", required=False, help="Path to the model or model name in API.")
    # TODO: should all this be inside a single dictionary config?
    parser.add_argument(
        "--server_address",
        required=False,
        help="Use ip:port for self-hosted models or the API url if using model providers.",
    )
    # TODO: let's make it not needed - we just need to unify our api calls
    parser.add_argument(
        "--server_type",
        choices=('nemo', 'tensorrt_llm', 'vllm', 'openai'),
        default='tensorrt_llm',
        help="Type of the server to start. This parameter is ignored if server_address is specified.",
    )
    parser.add_argument("--server_gpus", type=int, required=False)
    parser.add_argument(
        "--server_nodes",
        type=int,
        default=1,
        help="Number of nodes required for hosting LLM server.",
    )
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument(
        "--dependent_jobs",
        type=int,
        default=0,
        help="Specify this to launch that number of dependent jobs. Useful for large datasets, "
        "where you're not able to process everything before slurm timeout.",
    )
    parser.add_argument("--starting_seed", type=int, default=0)
    parser.add_argument(
        "--partition",
        required=False,
        help="Can specify if need interactive jobs or a specific non-default partition",
    )
    parser.add_argument(
        "--extra_eval_args",
        default="",
        help="Any extra arguments to pass to nemo_skills/evaluation/evaluate_results.py",
    )
    args, unknown = parser.parse_known_args()

    extra_arguments = f'{" ".join(unknown)}'

    with open(Path(__file__).parents[2] / 'cluster_configs' / f'{args.cluster}.yaml', "rt", encoding="utf-8") as fin:
        cluster_config = yaml.safe_load(fin)

    if args.server_address is None:  # we need to host the model
        assert args.server_gpus is not None, "Need to specify server_gpus if hosting the model"
        args.server_address = "localhost:5000"
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
            f"++server.server_type={args.server_type} "
            f"++server.base_url={args.server_address} "
            f"++server.model={args.model} "
        )

    with run.Experiment(args.expname) as exp:
        for seed in range(args.starting_seed, args.starting_seed + args.num_runs):
            # TODO: needs support on nemorun side
            assert args.dependent_jobs == 0
            cmd = get_cmd(random_seed=seed, extra_arguments=extra_arguments, extra_eval_args=args.extra_eval_args)
            add_task(
                exp,
                cmd=get_generation_command(server_address=args.server_address, generation_commands=cmd),
                # TODO: has to be the same currently to reuse the code, need a fix in nemo.run
                task_name="generate",  # f'generate-rs{seed}',
                container=cluster_config["containers"]["nemo-skills"],
                cluster_config=cluster_config,
                partition=args.partition,
                server_config=server_config,
                with_sandbox=True,
            )
        run_exp(exp, cluster_config)
        # exp.dryrun()
