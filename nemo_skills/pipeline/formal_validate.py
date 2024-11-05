import logging
from enum import Enum

import nemo_run as run
import typer

from nemo_skills.pipeline import (
    add_task,
    check_if_mounted,
    get_cluster_config,
    get_generation_command,
    run_exp,
)
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.utils import setup_logging

LOG = logging.getLogger(__file__)


class SupportedServers(str, Enum):
    trtllm = "trtllm"
    vllm = "vllm"
    nemo = "nemo"
    openai = "openai"


def get_formal_validate_cmd(
    output_dir,
    extra_arguments_first,
    extra_arguments_second,
    max_samples=-1,
    random_seed=None,
    server_type=None,
    cluster=None,
):
    # Handle file naming based on random_seed
    if random_seed is not None:
        seed_suffix = f"-rs{random_seed}"
    else:
        seed_suffix = ""

    # All outputs are in the 'generation' folder
    generation_dir = f"{output_dir}/generation"

    # Temporary output file for the first generate command
    temp_output_file = f"{generation_dir}/temp_output{seed_suffix}.jsonl"

    # Final output files after processing
    output_file = f"{generation_dir}/temp-output{seed_suffix}.jsonl"
    con_output_file = f"{generation_dir}/temp-con-output{seed_suffix}.jsonl"

    # Build the first generate command
    cmd1 = (
        f"python -m nemo_skills.inference.generate ++skip_filled=True "
        f"++output_file={temp_output_file} ++max_samples={max_samples} "
        f"{extra_arguments_first} "
    )
    if random_seed is not None:
        cmd1 += (
            f"++inference.random_seed={random_seed} "
            f"++inference.temperature=1.0 "
            f"++inference.top_k=0 "
            f"++inference.top_p=0.95 "
        )

    cmd1_trans_eval = (
        f"python -m nemo_skills.evaluation.evaluate_results "
        f"++input_files={temp_output_file} ++eval_type=lean4-stat "
    )

    # Process the outputs using custom scripts to produce the final outputs
    cmd2 = (
        f"python nemo_skills/inference/lean4/process_contr.py "
        f"{temp_output_file} {con_output_file} && "
        f"python nemo_skills/inference/lean4/process_formal.py "
        f"{temp_output_file} {output_file} "
    )

    # Remove the temporary file
    # cmd_cleanup = f"rm {temp_output_file} {con_output_file} {output_file}"
    cmd_cleanup = f"rm {temp_output_file}"


    # Generate commands on the processed files and run evaluations
    # For con-output
    final_con_output_file = f"{generation_dir}/dirty-con-output{seed_suffix}.jsonl"
    cmd3_con_generate = (
        f"python -m nemo_skills.inference.generate "
        f"++input_file={con_output_file} "
        f"++output_file={final_con_output_file} "
        f"++max_samples={max_samples} "
        f"{extra_arguments_second} "
    )
    if random_seed is not None:
        cmd3_con_generate += (
            f"++inference.random_seed={random_seed} "
            f"++inference.temperature=1.0 "
            f"++inference.top_k=0 "
            f"++inference.top_p=0.95 "
        )
    cmd3_con_eval = (
        f"python -m nemo_skills.evaluation.evaluate_results "
        f"++input_files={final_con_output_file} ++eval_type=lean4 "
    )

    # For output
    final_output_file = f"{generation_dir}/dirty-output{seed_suffix}.jsonl"
    cmd3_formal_generate = (
        f"python -m nemo_skills.inference.generate "
        f"++input_file={output_file} "
        f"++output_file={final_output_file} "
        f"++max_samples={max_samples} "
        f"{extra_arguments_second} "
    )
    if random_seed is not None:
        cmd3_formal_generate += (
            f"++inference.random_seed={random_seed} "
            f"++inference.temperature=1.0 "
            f"++inference.top_k=0 "
            f"++inference.top_p=0.95 "
        )
    cmd3_formal_eval = (
        f"python -m nemo_skills.evaluation.evaluate_results "
        f"++input_files={final_output_file} ++eval_type=lean4 "
    )
    final_con_output_file_clean = f"{generation_dir}/con-output{seed_suffix}.jsonl"
    final_output_file_clean = f"{generation_dir}/output{seed_suffix}.jsonl"


    cmd4 = (
        f"python nemo_skills/inference/lean4/drop_trivials.py "
        f"{final_output_file} {final_con_output_file} {final_output_file_clean} {final_con_output_file_clean}"
    )

    cmd_cleanup = f"rm {temp_output_file} {con_output_file} {output_file} {final_output_file} {final_con_output_file}"


    # Combine all commands
    cmd = (
        f"{cmd1} &&{cmd1_trans_eval} && {cmd2} && "
        f"{cmd3_con_generate} && {cmd3_con_eval} && "
        f"{cmd3_formal_generate} && {cmd3_formal_eval} && "
        f"{cmd4} && "
        f"{cmd_cleanup}"
    )
    return cmd


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
@typer_unpacker
def formal_validate(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="Cluster configuration.",
    ),
    output_dir: str = typer.Option(..., help="Where to put results"),
    expname: str = typer.Option("formal_validate", help="Experiment name"),
    model: str = typer.Option(None, help="Path to the model or model name in API"),
    server_address: str = typer.Option(None, help="Address of the model server"),
    server_type: SupportedServers = typer.Option(
        ..., help="Type of server to use"
    ),
    server_gpus: int = typer.Option(
        None, help="Number of GPUs for hosting the model"
    ),
    server_nodes: int = typer.Option(
        1, help="Number of nodes for hosting the model"
    ),
    server_args: str = typer.Option("", help="Extra arguments for the server"),
    partition: str = typer.Option(None, help="Cluster partition to use"),
    run_after: str = typer.Option(
        None, help="Specify an experiment that needs to be completed before this one"
    ),
    config_dir: str = typer.Option(None, help="Directory for cluster configs"),
    log_dir: str = typer.Option(None, help="Directory for logs"),
    extra_arguments_first: str = typer.Option(
        "++dataset=math-translate ++split=test ++prompt_config=lean/nat-to-lean4 "
        "++examples_type=math_to_lean4_fewshot ++prompt_template=deepseek-prover-translation "
        "++inference.tokens_to_generate=512",
        help="Extra arguments for the first generate command",
    ),
    extra_arguments_second: str = typer.Option(
        "++prompt_config=lean/lean4-false-proof ++examples_type=minif2f_deepseek_fewshot "
        "++prompt_template=deepseek-prover-translation ++inference.tokens_to_generate=512",
        help="Extra arguments for the second generate command",
    ),
    max_samples: int = typer.Option(-1, help="Maximum number of samples"),
    num_random_seeds: int = typer.Option(
        None, help="Number of random seeds for pass@k evaluation"
    ),
    starting_seed: int = typer.Option(0, help="Starting seed for random sampling"),
):
    """Pipeline to generate translations with proof "sorry" and then attempt to prove statements."""
    setup_logging(disable_hydra_logs=False)
    extra_arguments = " ".join(ctx.args)

    try:
        server_type = server_type.value
    except AttributeError:
        pass

    cluster_config = get_cluster_config(cluster, config_dir)
    check_if_mounted(cluster_config, output_dir)
    if log_dir:
        check_if_mounted(cluster_config, log_dir)
    else:
        log_dir = f"{output_dir}/formal_validate-logs"

    if server_address is None:
        assert (
            server_gpus is not None
        ), "Need to specify server_gpus if hosting the model"
        server_address = "localhost:5000"

        server_config = {
            "model_path": model,
            "server_type": server_type,
            "num_gpus": server_gpus,
            "num_nodes": server_nodes,
            "server_args": server_args,
        }
        extra_arguments += f" ++server.server_type={server_type} "
    else:
        server_config = None
        extra_arguments += (
            f" ++server.server_type={server_type} ++server.base_url={server_address} ++server.model={model} "
        )

    # Remove ++dataset and ++split from extra_arguments for the second generate command
    extra_arguments_second_filtered = " ".join(
        arg
        for arg in extra_arguments.split()
        if not arg.startswith("++dataset=") and not arg.startswith("++split=")
    )

    # Combine server configurations with extra arguments
    extra_arguments_first = f"{extra_arguments} {extra_arguments_first}"
    extra_arguments_second = (
        f"{extra_arguments_second_filtered} {extra_arguments_second}"
    )

    with run.Experiment(expname) as exp:
        # Generate outputs without random seeds
        cmd = get_formal_validate_cmd(
            output_dir=output_dir,
            extra_arguments_first=extra_arguments_first,
            extra_arguments_second=extra_arguments_second,
            max_samples=max_samples,
            random_seed=None,
            server_type=server_type,
            cluster=cluster,
        )
        add_task(
            exp,
            cmd=get_generation_command(
                server_address=server_address, generation_commands=cmd
            ),
            task_name="formal_validate",
            log_dir=log_dir,
            container=cluster_config["containers"]["nemo-skills"],
            cluster_config=cluster_config,
            partition=partition,
            server_config=server_config,
            with_sandbox=True,
            run_after=run_after,
        )

        # Generate outputs with random seeds if specified
        if num_random_seeds:
            for seed in range(starting_seed, starting_seed + num_random_seeds):
                cmd = get_formal_validate_cmd(
                    output_dir=output_dir,
                    extra_arguments_first=extra_arguments_first,
                    extra_arguments_second=extra_arguments_second,
                    max_samples=max_samples,
                    random_seed=seed,
                    server_type=server_type,
                    cluster=cluster,
                )
                add_task(
                    exp,
                    cmd=get_generation_command(
                        server_address=server_address, generation_commands=cmd
                    ),
                    task_name=f"formal_validate_rs{seed}",
                    log_dir=log_dir,
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
