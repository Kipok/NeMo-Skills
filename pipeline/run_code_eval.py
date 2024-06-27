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

"""
Local launch command
CODE_TEMPLATE = "[nemo-chat]" or "[nemo-wizardcoder]"

CODE_TEMPLATE="[nemo-chat]" python run_evals.py \
    --model-path "path to .nemo file" \
    --output-path "./results/<nemo filename>/" \
    --evals "code" \
    --cluster_config "local.yaml" \
    --num_gpus 2

"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

# adding nemo_skills to python path to avoid requiring installation
sys.path.append(str(Path(__file__).absolute().parents[1]))

from launcher import CLUSTER_CONFIG, NEMO_SKILLS_CODE, get_server_command, launch_job

SLURM_CMD = """
nvidia-smi && \
cd /code && \
export PYTHONPATH=$PYTHONPATH:/code && \
export HF_TOKEN={HF_TOKEN} && \
if [ $SLURM_LOCALID -eq 0 ]; then \
    {{ {server_start_cmd} 2>&1 | tee /tmp/server_logs.txt & }} && sleep 1 && \
    echo "Waiting for the server to start" && \
    tail -n0 -f /tmp/server_logs.txt | sed '/{server_wait_string}/ q' && \
    pip install evalplus && \
    python nemo_skills/inference/human_eval.py \
        --output_path /results/humaneval \
        --N 1 --batchsize 128 --greedy_decode \
        --temperature 1.0 --top_p 1.0 --max_len 1280 \
        --nemo_port 5000 && \
    python nemo_skills/inference/process_humaneval.py \
        --path /results/humaneval \
        --out_path /results/humaneval/results.jsonl \
        --add_prompt --ast_check && \
    evalplus.evaluate --dataset humaneval --samples /results/humaneval/results.jsonl > /results/humaneval/scores.txt && \
    pkill -f nemo_skills/inference/server; \
else \
    {server_start_cmd}; \
fi \
"""

# python generate_mbpp_nemo.py \
#     --output_path /results/mbpp \
#     --N 1 --batchsize 25 --greedy_decode \
#     --temperature 1.0 --top_p 1.0 --max_len 2048 \
#     --model_type nemo \
#     --template {template} \
#     --nemo_port 5000 && \
# python process_mbpp_nemo.py \
#     --path /results/mbpp \
#     --out_path /results/mbpp/results.jsonl \
#     --add_prompt --ast_check && \
# evalplus.evaluate --dataset mbpp --samples /results/mbpp/results.jsonl > /results/mbpp/scores.txt && \


MOUNTS = "{NEMO_SKILLS_CODE}:/code,{model_path}:/model,{output_dir}:/results"
JOB_NAME = "code-eval-{model_name}"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--server_type", choices=('nemo',), default='nemo')
    parser.add_argument(
        "--partition",
        required=False,
        help="Can specify if need interactive jobs or a specific non-default partition",
    )
    args, unknown = parser.parse_known_args()

    extra_arguments = f'{" ".join(unknown)}'

    args.model_path = Path(args.model_path).absolute()
    args.output_dir = Path(args.output_dir).absolute()

    server_start_cmd, num_tasks, server_wait_string = get_server_command(
        args.server_type, args.num_gpus, args.num_nodes, args.model_path.name
    )

    format_dict = {
        "model_path": args.model_path,
        "model_name": args.model_path.name,
        "output_dir": args.output_dir,
        "num_gpus": args.num_gpus,
        "server_start_cmd": server_start_cmd,
        "server_type": args.server_type,
        "NEMO_SKILLS_CODE": NEMO_SKILLS_CODE,
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),  # needed for some of the models, so making an option to pass it in
        "server_wait_string": server_wait_string,
    }

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    launch_job(
        cmd=SLURM_CMD.format(**format_dict),
        num_nodes=args.num_nodes,
        tasks_per_node=num_tasks,
        gpus_per_node=format_dict["num_gpus"],
        job_name=JOB_NAME.format(**format_dict),
        container=CLUSTER_CONFIG["containers"][args.server_type],
        mounts=MOUNTS.format(**format_dict),
        partition=args.partition,
        with_sandbox=True,
        extra_sbatch_args=["--parsable", f"--output={args.output_dir}/slurm_logs_code_eval.log"],
    )
