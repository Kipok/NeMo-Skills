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

from nemo_skills.pipeline import add_task, check_if_mounted, get_cluster_config
from nemo_skills.utils import setup_logging

if __name__ == "__main__":
    setup_logging(disable_hydra_logs=False)
    parser = ArgumentParser()
    parser.add_argument("--config_dir", default=None, help="Path to the cluster_configs dir")
    parser.add_argument("--log_dir", required=False, help="Can specify a custom location for slurm logs")
    parser.add_argument("--cluster", required=True, help="One of the configs inside cluster_configs")
    parser.add_argument("--model", required=True, help="Path to the model.")
    parser.add_argument(
        "--server_type",
        choices=('nemo', 'trtllm', 'vllm'),
        default='trtllm',
        help="Type of the server to start. This parameter is ignored if server_address is specified.",
    )
    parser.add_argument("--server_gpus", type=int, required=True)
    parser.add_argument(
        "--server_nodes",
        type=int,
        default=1,
        help="Number of nodes required for hosting LLM server.",
    )
    parser.add_argument("--server_args", default="", help="Any extra arguments to pass to the server.")
    parser.add_argument(
        "--partition",
        required=False,
        help="Can specify if need interactive jobs or a specific non-default partition",
    )
    parser.add_argument(
        "--with_sandbox", action="store_true", help="Enables local sandbox if code execution is required."
    )
    args = parser.parse_args()

    cluster_config = get_cluster_config(args.cluster, args.config_dir)
    check_if_mounted(cluster_config, args.model)
    if args.log_dir:
        check_if_mounted(cluster_config, args.log_dir)

    server_config = {
        "model_path": args.model,
        "server_type": args.server_type,
        "num_gpus": args.server_gpus,
        "num_nodes": args.server_nodes,
        "server_args": args.server_args,
    }

    with run.Experiment("server") as exp:
        add_task(
            exp,
            cmd="",  # not running anything except the server
            task_name=f'server-{args.model.replace("/", "-")}',
            log_dir=args.log_dir,
            container=cluster_config["containers"]["nemo-skills"],
            cluster_config=cluster_config,
            partition=args.partition,
            server_config=server_config,
            with_sandbox=args.with_sandbox,
        )
        # we don't want to detach in this case even on slurm, so not using run_exp
        exp.run(detach=False, tail_logs=True)
        # TODO: seems like not being killed? If nemorun doesn't do this, we can catch the signal and kill the server ourselves
        # TODO: logs not streamed, probably a bug with custom log path
