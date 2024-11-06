import os
import subprocess
from pathlib import Path
import typer
from nemo_skills.pipeline.app import typer_unpacker

app = typer.Typer()

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def formal_validate_on_folder(
    ctx: typer.Context,
    input_folder: str = typer.Option(..., "--input_folder", "-i", help="Path to the folder containing JSON files."),
    input_relative_path: str = typer.Option(..., "--input_relative_path", "-r", help="Relative path to the folder containing JSON files mounted on docker."),
    output_base_dir: str = typer.Option(..., "--output_base_dir", "-o", help="Base directory for output directories."),
    cluster: str = typer.Option("local", help="Cluster configuration."),
    server_type: str = typer.Option("vllm", "--server_type", help="Type of server to use."),
    model: str = typer.Option("/models/DeepSeek-Prover-V1.5-RL", help="Path to the model."),
    server_gpus: int = typer.Option(1, "--server_gpus", help="Number of GPUs for the server."),
    server_nodes: int = typer.Option(1, "--server_nodes", help="Number of nodes for the server."),
    num_random_seeds: int = typer.Option(1, "--num_random_seeds", help="Number of random seeds."),
    max_samples: int = typer.Option(10, "--max_samples", help="Maximum number of samples."),
):

    """
    Runs the formal_validate.py script on all JSON files in the input folder.
    """
    # Convert input paths to Path objects
    input_folder = Path(input_folder)
    output_base_dir = Path(output_base_dir)

    # Find all JSONL files in the input folder
    json_files = list(input_folder.glob("*.jsonl"))

    # Collect any extra arguments as a string
    extra_arguments = ' '.join(ctx.args)

    for json_file in json_files:
        # Compute the relative path and create output directory name
        relative_path = json_file.relative_to(input_folder)
        output_dir_name = '-'.join(relative_path.with_suffix('').parts)
        output_dir = output_base_dir / output_dir_name


        docker_input_file = os.path.join(input_relative_path, str(relative_path))

    # Build the command to run formal_validate.py
        cmd = (
            f"python nemo_skills/pipeline/formal_validate.py "
            f"--cluster {cluster} "
            f"--server_type {server_type} "
            f"--model {model} "
            f"--server_gpus {server_gpus} "
            f"--server_nodes {server_nodes} "
            f"--output_dir {output_dir} "
            f"--input_file {docker_input_file} "
            f"--num_random_seeds {num_random_seeds} "
            f"--max_samples {max_samples} "
            f"{extra_arguments}"
        )

        # Print and run the command
        print(f"Running command for {json_file}: {cmd}")
        subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
