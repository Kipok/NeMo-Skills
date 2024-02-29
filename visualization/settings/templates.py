import os
from pathlib import Path

path = os.path.abspath(Path(__file__).parent.parent.parent)
header = f"export PYTHONPATH='{path}:$PYTHONPATH'"

generate_solution_template = (
    header
    + """
python3 nemo_skills/inference/generate_solutions.py \\
    output_file={output_file} \\
    ++dataset={dataset} \\
    ++split_name={split_name} \\
    ++batch_size={batch_size} \\
    ++max_samples={max_samples} \\
    ++skip_filled={skip_filled} \\
    ++offset={offset} \\
    ++context={context_templates} \\
    ++prompt.delimiter="{delimiter}" \\
    ++prompt.prefix="{prefix}" \\
    ++prompt.examples_type={examples_type} \\
    ++prompt.template="{template}" \\
    ++prompt.num_few_shots={num_few_shots} \\
    ++prompt.context_type={context_type} \\
    ++server.server_type={server_type} \\
    ++server.host={host} \\
    ++server.port={port} \\
    ++server.ssh_server={ssh_server} \\
    ++server.ssh_key_path={ssh_key_path} \\
    ++sandbox.host={sandbox_host} \\
    ++inference.temperature={temperature} \\
    ++inference.top_k={top_k} \\
    ++inference.top_p={top_p} \\
    ++inference.random_seed={random_seed} \\
    ++inference.tokens_to_generate={tokens_to_generate} \\
    ++inference.repetition_penalty={repetition_penalty} \\
"""
)


evaluate_results_template = (
    header
    + """
python3 nemo_skills/evaluation/evaluate_results.py \\
    prediction_jsonl_files={prediction_jsonl_files} \\
    ++sandbox.host={sandbox_host} \\
    ++sandbox.ssh_server={ssh_server} \\
    ++sandbox.ssh_key_path={ssh_key_path} \\
"""
)

compute_metrics_template = (
    header
    + """
python3 pipeline/compute_metrics.py \\
  --prediction_jsonl_files {prediction_jsonl_files} \\
  --save_metrics_file {save_metrics_file}
"""
)
