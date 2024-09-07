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

from nemo_skills.pipeline import add_task, get_cluster_config, get_generation_command, run_exp
from nemo_skills.utils import setup_logging


def get_cmd(training_folder, nemo_model, average_steps, conversion_step):
    cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"cd /nemo_run/code && "
        f"python /code/nemo_skills/finetuning/average_checkpoints.py "
        f"    --untarred_nemo_folder {nemo_model} "
        f"    --name_prefix=model "
        f"    --checkpoint_dir={training_folder} {average_steps} &&"
        f"{conversion_step}"
    )
    return cmd


if __name__ == "__main__":
    setup_logging(disable_hydra_logs=False)
    parser = ArgumentParser()
    parser.add_argument("--training_folder", required=True)
    parser.add_argument("--cluster", required=True, help="One of the configs inside cluster_configs")
    parser.add_argument("--server_type", choices=('nemo',), default='nemo')
    parser.add_argument("--output_path", required=True, help="Path to save the prepared model")
    parser.add_argument("--nemo_model", required=True, help="Only need this to get the config file")
    parser.add_argument("--num_gpus", required=True, type=int)
    parser.add_argument(
        "--average_steps",
        nargs="+",
        type=int,
        help="List of checkpoint steps to average. If not specified, will average all.",
    )
    parser.add_argument(
        "--partition",
        required=False,
        help="Can specify if need interactive jobs or a specific non-default partition",
    )
    args = parser.parse_args()

    conversion_step = f"mv {args.training_folder}/model-averaged.nemo {args.output_path}"
    # TODO: add NeMo to TensorRT-LLM conversion step

    cluster_config = get_cluster_config(args.cluster)

    with run.Experiment(args.expname) as exp:
        cmd = get_cmd(
            training_folder=args.training_folder,
            nemo_model=args.nemo_model,
            average_steps=f"--steps {' '.join(map(str, args.average_steps))} " if args.average_steps else "",
            conversion_step=conversion_step,
        )
        add_task(
            exp,
            cmd=cmd,
            task_name="prepare-eval",
            container=cluster_config["containers"][args.server_type],
            cluster_config=cluster_config,
            partition=args.partition,
            num_nodes=1,
            num_tasks=1,
            num_gpus=args.num_gpus,
        )
        run_exp(exp, cluster_config)
