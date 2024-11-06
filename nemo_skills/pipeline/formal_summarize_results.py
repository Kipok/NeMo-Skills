import glob
import importlib
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import typer

from nemo_skills.evaluation.metrics import Lean4Metrics, compute_metrics
from nemo_skills.pipeline import (
    check_if_mounted,
    cluster_download,
    get_cluster_config,
    get_tunnel,
    get_unmounted_path,
)
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.utils import setup_logging


@app.command()
@typer_unpacker
def formal_summarize_results(
    results_dir: str = typer.Argument(
        ...,
        help="Path to the directory containing generation outputs (e.g., output.jsonl, con-output.jsonl). "
        "If cluster is specified, will fetch the results from there.",
    ),
    cluster: Optional[str] = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument. "
        "If not specified, will assume the results are in the local filesystem.",
    ),
    config_dir: Optional[str] = typer.Option(
        None, help="Can customize where we search for cluster configs"
    ),
    debug: bool = typer.Option(False, help="Print debug information"),
    max_samples: int = typer.Option(-1, help="Limit metric computation only to first `max_samples`"),
):
    """Summarize formalization results in the specified directory."""
    setup_logging(disable_hydra_logs=False, log_level=logging.INFO if not debug else logging.DEBUG)

    cluster = cluster or os.environ.get("NEMO_SKILLS_CONFIG")

    # Copying results from the cluster if necessary
    if cluster is not None:
        cluster_config = get_cluster_config(cluster, config_dir)
        check_if_mounted(cluster_config, results_dir)
        if cluster == "local":
            results_dir = get_unmounted_path(cluster_config, results_dir)
        else:
            tunnel = get_tunnel(cluster_config)
            temp_dir = tempfile.mkdtemp()
            print(f"Copying results from {results_dir} on cluster {cluster} to {temp_dir}")
            os.makedirs(temp_dir, exist_ok=True)
            cluster_download(tunnel, get_unmounted_path(cluster_config, results_dir), temp_dir)
            tunnel.cleanup()
            results_dir = Path(temp_dir) / Path(results_dir).name

    results_dir = Path(results_dir)

    if not results_dir.exists():
        print(f"The specified results_dir {results_dir} does not exist.")
        return

    # Find all jsonl files in the directory
    jsonl_files = list(results_dir.glob('*.jsonl'))

    if not jsonl_files:
        print(f"No jsonl files found in {results_dir}")
        return

    # Group files by their prefix (e.g., con- and regular)
    file_groups = defaultdict(list)
    for jsonl_file in jsonl_files:
        filename = jsonl_file.name
        if filename.startswith('con-'):
            group = 'con'
        else:
            group = 'regular'
        file_groups[group].append(jsonl_file)

    metrics_calculator = Lean4Metrics()

    results = defaultdict(dict)
    for group, files in file_groups.items():
        # Now, within each group, further group files by their evaluation mode
        # For example, greedy (output.jsonl), and sampling (output-rs*.jsonl)

        greedy_files = [f for f in files if 'rs' not in f.name]
        sampling_files = [f for f in files if 'rs' in f.name]

        group_name = results_dir.name
        if group == 'con':
            group_name += ' (con)'

        results[group_name] = {}

        if greedy_files:
            metrics = compute_metrics(
                input_files=[str(f) for f in greedy_files],
                metrics_calculator=metrics_calculator,
                max_samples=max_samples,
                aggregation_mode='first',
                allow_incomplete=True,
            )
            results[group_name]['greedy'] = metrics

        if sampling_files:
            num_samples = len(set(f.name.split('-rs')[1].split('.')[0] for f in sampling_files))
            sampling_files_sorted = sorted(sampling_files, key=lambda f: f.name)
            metrics_best = compute_metrics(
                input_files=[str(f) for f in sampling_files_sorted],
                metrics_calculator=metrics_calculator,
                max_samples=max_samples,
                aggregation_mode='best',
                allow_incomplete=True,
            )
            results[group_name][f'pass@{num_samples}'] = metrics_best

    # Now, print the results
    for group_name, group_results in results.items():
        if not group_results:
            continue
        # Calculate max widths for formatting
        max_widths = {}
        max_widths['evaluation_mode'] = len('evaluation_mode')
        for eval_mode, metrics in group_results.items():
            for metric_key, metric_value in metrics.items():
                max_widths[metric_key] = max(
                    max_widths.get(metric_key, len(metric_key)),
                    len(f"{metric_value:.2f}" if isinstance(metric_value, float) else str(metric_value)),
                )
            max_widths['evaluation_mode'] = max(max_widths['evaluation_mode'], len(eval_mode))

        total_width = sum(max_widths.values()) + (len(max_widths) - 1) * 3
        print(f' {group_name} '.center(total_width, '-'))
        headers = ['evaluation_mode'] + list(list(group_results.values())[0].keys())
        print(' | '.join([f'{header:<{max_widths[header]}}' for header in headers]))

        for eval_mode, metrics in group_results.items():
            values = [f'{eval_mode:<{max_widths["evaluation_mode"]}}']
            for metric_key, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    metric_value = f"{metric_value:.2f}"
                values.append(f'{str(metric_value):<{max_widths[metric_key]}}')
            print(' | '.join(values))

        print('\n')


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()
