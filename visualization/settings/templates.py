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

evaluate_results_template = """
python nemo_skills/evaluation/evaluate_results.py \\
    prediction_jsonl_files={prediction_jsonl_files} \\
    ++sandbox.host={host} \\
    ++sandbox.ssh_server={ssh_server} \\
    ++sandbox.ssh_key_path={ssh_key_path} \\
"""

compute_metrics_template = """
python pipeline/compute_metrics.py \\
  --prediction_jsonl_files {prediction_jsonl_files} \\
  --save_metrics_file {save_metrics_file}
"""
