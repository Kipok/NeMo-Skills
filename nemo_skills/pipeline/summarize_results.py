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

# will run compute metrics on all relevant files and summarize results in a .csv file

import argparse
import glob
import importlib
import json
import logging
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# adding nemo_skills to python path to avoid requiring installation
sys.path.append(str(Path(__file__).absolute().parents[1]))
sys.path.append(str(Path(__file__).absolute().parents[0]))

from nemo_skills.evaluation.metrics import MathMetrics
from nemo_skills.pipeline import check_if_mounted, cluster_download, get_cluster_config, get_tunnel, get_unmounted_path
from nemo_skills.pipeline.compute_metrics import compute_metrics
from nemo_skills.utils import setup_logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'results_folder',
        help=(
            "Path to the folder with results. Needs to contain <benchmark> folders inside. "
            "If cluster is specified, will fetch the results from there."
        ),
    )
    parser.add_argument(
        '--cluster',
        required=False,
        help="Cluster configuration to take results from. If 'local' is explicitly specified, "
        "we assume the location is relative to one of the mounted folders.",
    )
    parser.add_argument('--config_folder', default=None, help="Path to the cluster_configs folder.")
    parser.add_argument(
        '--benchmarks',
        nargs="+",
        default=[],
        help="Specify benchmarks to run. If not specified, all benchmarks in the results_folder will be used.",
    )
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    args = parser.parse_args()

    setup_logging(disable_hydra_logs=False, log_level=logging.INFO if not args.debug else logging.DEBUG)

    # copying results from the cluster
    cluster_config = get_cluster_config(args.cluster, args.config_folder)
    if args.cluster is not None:
        check_if_mounted(cluster_config, args.results_folder)
    if args.cluster == "local":
        args.results_folder = get_unmounted_path(cluster_config, args.results_folder)
    else:
        tunnel = get_tunnel(cluster_config)
        temp_dir = tempfile.mkdtemp()
        print(f"Copying results from {args.results_folder} on cluster {args.cluster} to {temp_dir}")
        os.makedirs(temp_dir, exist_ok=True)
        cluster_download(tunnel, get_unmounted_path(cluster_config, args.results_folder), temp_dir)
        tunnel.cleanup()
        args.results_folder = Path(temp_dir) / Path(args.results_folder).name

    # running compute_metrics.py to get greedy, majority and pass @k results for all benchmarks available
    # Check if there is an eval-results folder inside the results_folder
    eval_results_folder = Path(args.results_folder) / 'eval-results'
    if eval_results_folder.exists() and eval_results_folder.is_dir():
        args.results_folder = eval_results_folder
    benchmarks = glob.glob(f'{args.results_folder}/*')

    if args.benchmarks:
        for benchmark in benchmarks.copy():
            if Path(benchmark).name not in args.benchmarks:
                benchmarks.remove(benchmark)

    current_folder = Path(__file__).absolute().parent
    results = defaultdict(dict)
    for benchmark_path in benchmarks:
        benchmark = str(Path(benchmark_path).name)
        if not Path(benchmark_path).is_dir():
            continue
        try:
            benchmark_module = importlib.import_module(f"nemo_skills.dataset.{benchmark}")
            metrics_calculator = benchmark_module.METRICS_CLASS()
            results[benchmark] = {}
            # TODO: we should just return all available aggregations from compute_metrics directly
            if metrics_calculator is not MathMetrics:
                if Path(f'{benchmark_path}/output-greedy.jsonl').exists():
                    results[benchmark]['greedy'] = compute_metrics(
                        input_files=[f"{benchmark_path}/output-greedy.jsonl"],
                        metrics_calculator=metrics_calculator,
                    )
                sampling_outputs = glob.glob(f'{benchmark_path}/output-rs*.jsonl')
                if len(sampling_outputs) > 0:
                    results[benchmark][f'pass@{len(sampling_outputs)}'] = compute_metrics(
                        input_files=sampling_outputs,
                        metrics_calculator=metrics_calculator,
                        aggregation_mode="best",
                    )
            else:
                if Path(f'{benchmark_path}/output-greedy.jsonl').exists():
                    results[benchmark]['greedy'] = compute_metrics(
                        input_files=[f"{benchmark_path}/output-greedy.jsonl"],
                        metrics_calculator=metrics_calculator,
                    )

                sampling_outputs = glob.glob(f'{benchmark_path}/output-rs*.jsonl')
                if len(sampling_outputs) > 0:
                    results[benchmark][f'majority@{len(sampling_outputs)}'] = compute_metrics(
                        input_files=sampling_outputs,
                        metrics_calculator=metrics_calculator,
                        aggregation_mode="majority",
                    )
                    results[benchmark][f'pass@{len(sampling_outputs)}'] = compute_metrics(
                        input_files=sampling_outputs,
                        metrics_calculator=metrics_calculator,
                        aggregation_mode="best",
                    )
        except Exception as e:
            print(f"Error running compute_metrics.py for {benchmark}: {e}")

    lines_to_write = []
    for benchmark, benchmark_results in results.items():
        if not benchmark_results:
            continue
        max_widths = {}
        max_widths['evaluation_mode'] = len('evaluation_mode')
        for eval_mode, metrics in benchmark_results.items():
            for metric_key, metric_value in metrics.items():
                max_widths[metric_key] = max(
                    max_widths.get(metric_key, len(metric_key)),
                    len(f"{metric_value:.2f}" if isinstance(metric_value, float) else str(metric_value)),
                )
            max_widths['evaluation_mode'] = max(max_widths['evaluation_mode'], len(eval_mode))

        total_width = sum(max_widths.values()) + (len(max_widths) - 1) * 3
        print(f' {benchmark} '.center(total_width, '-'))
        headers = ['evaluation_mode'] + list(list(benchmark_results.values())[0].keys())
        print(' | '.join([f'{header:<{max_widths[header]}}' for header in headers]))

        for eval_mode, metrics in benchmark_results.items():
            values = [f'{eval_mode:<{max_widths["evaluation_mode"]}}']
            for metric_key, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    metric_value = f"{metric_value:.2f}"
                values.append(f'{str(metric_value):<{max_widths[metric_key]}}')
            print(' | '.join(values))

        print('\n')
