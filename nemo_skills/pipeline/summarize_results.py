import glob
import importlib
import json
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import typer

from nemo_skills.evaluation.metrics import MathMetrics
from nemo_skills.pipeline import (
    check_if_mounted,
    cluster_download,
    cluster_upload,
    get_cluster_config,
    get_tunnel,
    get_unmounted_path,
)
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.compute_metrics import compute_metrics
from nemo_skills.utils import setup_logging


@app.command()
@typer_unpacker
def summarize_results(
    results_dir: str = typer.Argument(
        ...,
        help="Path to the dir with results. Needs to contain <benchmark> dirs inside, or output files directly if eval_type is specified. "
        "If cluster is specified, will fetch the results from there.",
    ),
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument. "
        "If not specified, will assume the results are in the local filesystem.",
    ),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    benchmarks: Optional[str] = typer.Option(
        None,
        help="Specify benchmarks to run (comma separated). "
        "If not specified, all benchmarks in the results_dir will be used.",
    ),
    remote_tar_dir: str = typer.Option(None, help="Directory where remote tar files are created on clusters"),
    debug: bool = typer.Option(False, help="Print debug information"),
    max_samples: int = typer.Option(-1, help="Limit metric computation only to first `max_samples`"),
    eval_type: Optional[str] = typer.Option(
        None,
        help="Specify the evaluation type (e.g., 'lean4-statement'). If specified, results_dir is processed directly and benchmarks are ignored.",
    ),
):
    """Summarize results of an evaluation job."""
    setup_logging(disable_hydra_logs=False, log_level=logging.INFO if not debug else logging.DEBUG)

    if benchmarks and " " in str(benchmarks):
        raise ValueError("benchmarks should be separated with commas")

    cluster = cluster or os.environ.get("NEMO_SKILLS_CONFIG")

    # copying results from the cluster if necessary
    upload_path = None
    if cluster is not None:
        cluster_config = get_cluster_config(cluster, config_dir)
        check_if_mounted(cluster_config, results_dir)
    if cluster == "local":
        results_dir = get_unmounted_path(cluster_config, results_dir)
    elif cluster is not None:
        upload_path = results_dir
        tunnel = get_tunnel(cluster_config)
        temp_dir = tempfile.mkdtemp()
        print(f"Copying results from {results_dir} on cluster {cluster} to {temp_dir}")
        os.makedirs(temp_dir, exist_ok=True)
        cluster_download(
            tunnel,
            get_unmounted_path(cluster_config, results_dir),
            temp_dir,
            remote_tar_dir=get_unmounted_path(cluster_config, remote_tar_dir),
        )
        results_dir = Path(temp_dir) / Path(results_dir).name

    if eval_type is not None:
        # Process results_dir as containing output files directly

        # Import the appropriate metrics calculator
        if eval_type == 'lean4':
            from nemo_skills.evaluation.metrics import Lean4Metrics
            metrics_calculator = Lean4Metrics()
        else:
            raise ValueError(f"Unknown eval_type: {eval_type}")

        results_dir = Path(results_dir)

        if not results_dir.exists():
            print(f"The specified results_dir {results_dir} does not exist.")
            return

        # Find all jsonl files in the directory
        jsonl_files = list(results_dir.glob('*.jsonl'))

        if not jsonl_files:
            print(f"No jsonl files found in {results_dir}")
            return

        # Group files by their prefix (e.g., con-, ex-con-, and regular)
        file_groups = defaultdict(list)
        for jsonl_file in jsonl_files:
            filename = jsonl_file.name
            if filename.startswith('ex-con-'):
                group = 'ex-con'
            elif filename.startswith('con-'):
                group = 'con'
            else:
                group = 'regular'
            file_groups[group].append(jsonl_file)

        # Modify 'con-' files and save as 'ex-con-' files if needed
        def modify_con_files():
            """Modify the 'con-' files according to the given rule and save as 'ex-con-' files."""
            import json

            # Collect all non-con files (regular files)
            non_con_files = file_groups.get('regular', [])

            # Build regular_proofs_dict with key as line_index and value as 'proof_status'
            regular_proofs_dict = {}

            # First, process non-con files to initialize the dictionary
            for non_con_file in non_con_files:
                with open(non_con_file, 'r') as f:
                    for line_index, line in enumerate(f):
                        line_json = json.loads(line.strip())
                        proof_status = line_json.get('proof_status')
                        # Initialize the dictionary with the proof_status
                        if line_index not in regular_proofs_dict:
                            regular_proofs_dict[line_index] = proof_status
                        else:
                            # Update the status to 'completed' if any proof_status is 'completed'
                            if regular_proofs_dict[line_index] != 'completed' and proof_status == 'completed':
                                regular_proofs_dict[line_index] = 'completed'

            # For each 'con-' file
            con_files = file_groups.get('con', [])
            for con_file in con_files:
                # Corresponding ex-con- file
                base_name = con_file.name[len('con-'):]  # Remove 'con-' prefix
                ex_con_file = con_file.parent / ('ex-con-' + base_name)

                with open(con_file, 'r') as cf_in, open(ex_con_file, 'w') as cf_out:
                    for line_index, con_line in enumerate(cf_in):
                        con_json = json.loads(con_line.strip())
                        # If the corresponding line in regular_proofs_dict is 'completed', update proof_status
                        if regular_proofs_dict.get(line_index) == 'completed':
                            con_json['proof_status'] = 'failed'
                        # Write the modified (or unmodified) line to the ex-con- file
                        cf_out.write(json.dumps(con_json) + '\n')

        modify_con_files()

        # After modification, include the 'ex-con-' files in the file_groups
        ex_con_files = list(results_dir.glob('ex-con-*.jsonl'))
        if ex_con_files:
            file_groups['ex-con'] = ex_con_files

        # Now, compute metrics for each group
        results = defaultdict(dict)
        for group, files in file_groups.items():
            greedy_files = [f for f in files if 'rs' not in f.name]
            sampling_files = [f for f in files if 'rs' in f.name]

            group_name = results_dir.name
            if group == 'con':
                group_name += ' (con)'
            elif group == 'ex-con':
                group_name += ' (ex-con)'

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

        # Save metrics.json
        try:
            with open(Path(results_dir) / 'metrics.json', 'wt', encoding='utf-8') as fout:
                json.dump(results, fout, indent=2)
            if upload_path is not None:
                cluster_upload(
                    tunnel,
                    Path(results_dir) / 'metrics.json',
                    Path(get_unmounted_path(cluster_config, upload_path)) / 'metrics.json',
                )
                print("Metrics are saved to", str(Path(get_unmounted_path(cluster_config, upload_path)) / 'metrics.json'))
                tunnel.cleanup()
            else:
                print("Metrics are saved to", str(Path(results_dir) / 'metrics.json'))
        except PermissionError:
            print(f"Could not save metrics.json to {Path(results_dir) / 'metrics.json'}. Please check the permissions.")

    else:
        # existing code for benchmarks
        # Check if there is an eval-results dir inside the results_dir
        eval_results_dir = Path(results_dir) / 'eval-results'
        if eval_results_dir.exists() and eval_results_dir.is_dir():
            results_dir = eval_results_dir
        benchmarks_paths = glob.glob(f'{results_dir}/*')

        if benchmarks:
            benchmarks_paths = [b for b in benchmarks_paths if Path(b).name in benchmarks.split(",")]

        results = defaultdict(dict)
        max_metrics_to_print = {}
        for benchmark_path in benchmarks_paths:
            benchmark = str(Path(benchmark_path).name)
            if not Path(benchmark_path).is_dir():
                continue
            try:
                benchmark_module = importlib.import_module(f"nemo_skills.dataset.{benchmark}")
                metrics_calculator = benchmark_module.METRICS_CLASS()
                results[benchmark] = {}
                max_metrics_to_print[benchmark] = metrics_calculator.max_metrics_to_print()
                # TODO: we should just return all available aggregations from compute_metrics directly
                if not isinstance(metrics_calculator, MathMetrics):
                    if Path(f'{benchmark_path}/output-greedy.jsonl').exists():
                        results[benchmark]['greedy'] = compute_metrics(
                            input_files=[f"{benchmark_path}/output-greedy.jsonl"],
                            metrics_calculator=metrics_calculator,
                            max_samples=max_samples,
                        )
                    sampling_outputs = glob.glob(f'{benchmark_path}/output-rs*.jsonl')
                    if len(sampling_outputs) > 0:
                        results[benchmark][f'pass@{len(sampling_outputs)}'] = compute_metrics(
                            input_files=sampling_outputs,
                            metrics_calculator=metrics_calculator,
                            aggregation_mode="best",
                            max_samples=max_samples,
                        )
                else:
                    if Path(f'{benchmark_path}/output-greedy.jsonl').exists():
                        results[benchmark]['greedy'] = compute_metrics(
                            input_files=[f"{benchmark_path}/output-greedy.jsonl"],
                            metrics_calculator=metrics_calculator,
                            max_samples=max_samples,
                        )

                    sampling_outputs = glob.glob(f'{benchmark_path}/output-rs*.jsonl')
                    if len(sampling_outputs) > 0:
                        results[benchmark][f'majority@{len(sampling_outputs)}'] = compute_metrics(
                            input_files=sampling_outputs,
                            metrics_calculator=metrics_calculator,
                            aggregation_mode="majority",
                            max_samples=max_samples,
                        )
                        results[benchmark][f'pass@{len(sampling_outputs)}'] = compute_metrics(
                            input_files=sampling_outputs,
                            metrics_calculator=metrics_calculator,
                            aggregation_mode="best",
                            max_samples=max_samples,
                        )
            except Exception as e:
                print(f"Error running compute_metrics.py for {benchmark}: {e}")

        for benchmark, benchmark_results in results.items():
            if not benchmark_results:
                continue
            max_widths = {}
            max_widths['evaluation_mode'] = len('evaluation_mode')
            for eval_mode, metrics in benchmark_results.items():
                if max_metrics_to_print[benchmark] is None:
                    max_metrics_to_print[benchmark] = len(metrics)
                for metric_key, metric_value in list(metrics.items())[: max_metrics_to_print[benchmark]]:
                    max_widths[metric_key] = max(
                        max_widths.get(metric_key, len(metric_key)),
                        len(f"{metric_value:.2f}" if isinstance(metric_value, float) else str(metric_value)),
                    )
                max_widths['evaluation_mode'] = max(max_widths['evaluation_mode'], len(eval_mode))

            total_width = sum(max_widths.values()) + (len(max_widths) - 1) * 3
            print(f' {benchmark} '.center(total_width, '-'))
            headers = ['evaluation_mode'] + list(list(benchmark_results.values())[0].keys())[
                : max_metrics_to_print[benchmark]
            ]
            print(' | '.join([f'{header:<{max_widths[header]}}' for header in headers]))

            for eval_mode, metrics in benchmark_results.items():
                values = [f'{eval_mode:<{max_widths["evaluation_mode"]}}']
                for metric_key, metric_value in list(metrics.items())[: max_metrics_to_print[benchmark]]:
                    if isinstance(metric_value, float):
                        metric_value = f"{metric_value:.2f}"
                    values.append(f'{str(metric_value):<{max_widths[metric_key]}}')
                print(' | '.join(values))

            print('\n')

        try:
            with open(Path(results_dir) / 'metrics.json', 'wt', encoding='utf-8') as fout:
                json.dump(results, fout, indent=2)
            if upload_path is not None:
                cluster_upload(
                    tunnel,
                    Path(results_dir) / 'metrics.json',
                    Path(get_unmounted_path(cluster_config, upload_path)) / 'metrics.json',
                )
                print("Metrics are saved to", str(Path(get_unmounted_path(cluster_config, upload_path)) / 'metrics.json'))
                tunnel.cleanup()
            else:
                print("Metrics are saved to", str(Path(results_dir) / 'metrics.json'))
        except PermissionError:
            print(f"Could not save metrics.json to {Path(results_dir) / 'metrics.json'}. Please check the permissions.")


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()
