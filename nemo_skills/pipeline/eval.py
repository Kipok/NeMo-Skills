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

import importlib
from argparse import ArgumentParser

import nemo_run as run
import typer

from nemo_skills.pipeline import add_task, check_if_mounted, get_cluster_config, get_generation_command, run_exp
from nemo_skills.utils import setup_logging

app = typer.Typer()


def get_greedy_cmd(
    benchmark, split, output_dir, output_name='output-greedy.jsonl', extra_eval_args="", extra_arguments=""
):
    benchmark_module = importlib.import_module(f"nemo_skills.dataset.{benchmark}")

    extra_eval_args = f"{benchmark_module.DEFAULT_EVAL_ARGS} {extra_eval_args}"
    extra_arguments = f"{benchmark_module.DEFAULT_GENERATION_ARGS} {extra_arguments}"
    cmd = (
        f'echo "Evaluating benchmark {benchmark}" && '
        f'python -m nemo_skills.inference.generate '
        f'    ++dataset={benchmark} '
        f'    ++split={split} '
        f'    ++output_file={output_dir}/eval-results/{benchmark}/{output_name} '
        f'    {extra_arguments} && '
        f'python -m nemo_skills.evaluation.evaluate_results '
        f'    ++input_files={output_dir}/eval-results/{benchmark}/{output_name} {extra_eval_args}'
    )
    return cmd


def get_sampling_cmd(benchmark, split, output_dir, random_seed, extra_eval_args="", extra_arguments=""):
    extra_arguments = f" inference.random_seed={random_seed} inference.temperature=0.7 {extra_arguments}"
    return get_greedy_cmd(
        benchmark=benchmark,
        split=split,
        output_dir=output_dir,
        output_name=f"output-rs{random_seed}.jsonl",
        extra_eval_args=extra_eval_args,
        extra_arguments=extra_arguments,
    )


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def eval(
    ctx: typer.Context,
    cluster: str = typer.Option(),
    output_dir: str = typer.Option(),
    benchmarks: list[str] = typer.Option(),
    config_dir: str | None = None,
    expname: str = "eval",
    model: str | None = None,
    server_address: str | None = None,
    server_type: str = 'trtllm',
    server_gpus: int | None = None,
    server_nodes: int = 1,
    server_args: str = "",
    starting_seed: int = 0,
    split: str = 'test',
    num_jobs: int = -1,
    partition: str | None = None,
    extra_eval_args: str = "",
    skip_greedy: bool = False,
    run_after: str | None = None,
):
    """
    Evaluate a model using specified benchmarks and configurations.

    Args:
        ctx (typer.Context): Context for passing extra options to the underlying script.
        cluster (str): Cluster configuration name.
        output_dir (str): Directory to store evaluation outputs.
        benchmarks (list[str]): List of benchmarks in the format "benchmark_name:num_repeats".
        config_dir (str | None, optional): Directory containing cluster configuration files. Defaults to None.
        expname (str, optional): Name of the experiment. Defaults to "eval".
        model (str | None, optional): Path to the model to be evaluated. Defaults to None.
        server_address (str | None, optional): Address of the server hosting the model. Defaults to None.
        server_type (str, optional): Type of server to use. Defaults to 'trtllm'.
        server_gpus (int | None, optional): Number of GPUs to use if hosting the model. Defaults to None.
        server_nodes (int, optional): Number of nodes to use if hosting the model. Defaults to 1.
        server_args (str, optional): Additional arguments for the server. Defaults to "".
        starting_seed (int, optional): Starting seed for random sampling. Defaults to 0.
        split (str, optional): Data split to use for evaluation. Defaults to 'test'.
        num_jobs (int, optional): Number of jobs to split the evaluation into. Defaults to -1.
        partition (str | None, optional): Cluster partition to use. Defaults to None.
        extra_eval_args (str, optional): Additional arguments for evaluation. Defaults to "".
        skip_greedy (bool, optional): Whether to skip greedy evaluation. Defaults to False.
        run_after (str | None, optional): Task to run after the evaluation. Defaults to None.
    """

    setup_logging(disable_hydra_logs=False)

    extra_arguments = f'{" ".join(ctx.args)}'

    cluster_config = get_cluster_config(cluster, config_dir)
    check_if_mounted(cluster_config, output_dir)

    if server_address is None:  # we need to host the model
        assert server_gpus is not None, "Need to specify server_gpus if hosting the model"
        server_address = "localhost:5000"

        server_config = {
            "model_path": model,
            "server_type": server_type,
            "num_gpus": server_gpus,
            "num_nodes": server_nodes,
            "server_args": server_args,
        }
        extra_arguments += f" ++server.server_type={server_type} "
    else:  # model is hosted elsewhere
        server_config = None
        extra_arguments += (
            f" ++server.server_type={server_type} " f" ++server.base_url={server_address} " f" ++server.model={model} "
        )

    benchmarks = {k: int(v) for k, v in [b.split(":") for b in benchmarks]}

    eval_cmds = (
        [
            get_greedy_cmd(
                benchmark,
                split,
                output_dir,
                extra_eval_args=extra_eval_args,
                extra_arguments=extra_arguments,
            )
            for benchmark in benchmarks.keys()
        ]
        if not skip_greedy
        else []
    )
    eval_cmds += [
        get_sampling_cmd(
            benchmark,
            split,
            output_dir,
            rs,
            extra_eval_args=extra_eval_args,
            extra_arguments=extra_arguments,
        )
        for benchmark, rs_num in benchmarks.items()
        for rs in range(starting_seed, starting_seed + rs_num)
    ]
    if num_jobs == -1:
        num_jobs = len(eval_cmds)

    # splitting eval cmds equally across num_jobs nodes
    eval_cmds = [" && ".join(eval_cmds[i::num_jobs]) for i in range(num_jobs)]

    with run.Experiment(expname) as exp:
        for idx, eval_cmd in enumerate(eval_cmds):
            add_task(
                exp,
                cmd=get_generation_command(server_address=server_address, generation_commands=eval_cmd),
                task_name=f'eval-{idx}',
                log_dir=f"{output_dir}/eval-logs",
                container=cluster_config["containers"]["nemo-skills"],
                cluster_config=cluster_config,
                partition=partition,
                server_config=server_config,
                with_sandbox=True,
                run_after=run_after,
            )
        run_exp(exp, cluster_config)


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
