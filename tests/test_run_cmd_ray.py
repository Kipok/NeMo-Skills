# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nemo_skills.pipeline.run_cmd import run_cmd


@pytest.mark.parametrize("with_ray", [False, True])
def test_run_cmd_forwards_ray_to_existing_task_builder(with_ray):
    # The CLI option chooses normal Nemo-Run metadata/executor handling; it
    # does not invoke Slurm or implement a second Ray bootstrap in Python.
    config = {"executor": "slurm", "containers": {}}
    experiment = MagicMock()
    with (
        patch("nemo_skills.pipeline.run_cmd.setup_logging"),
        patch("nemo_skills.pipeline.run_cmd.pipeline_utils.get_cluster_config", return_value=config),
        patch("nemo_skills.pipeline.run_cmd.pipeline_utils.resolve_mount_paths", return_value=config),
        patch("nemo_skills.pipeline.run_cmd.check_mounts", side_effect=lambda config, path, **kwargs: path),
        patch("nemo_skills.pipeline.run_cmd.get_exp", return_value=experiment),
        patch("nemo_skills.pipeline.run_cmd.add_task") as add_task,
        patch("nemo_skills.pipeline.run_cmd.run_exp") as run_exp,
    ):
        run_cmd(
            ctx=SimpleNamespace(args=[]),
            command="python driver.py",
            container="public-image",
            cluster="dfw",
            num_nodes=4,
            num_gpus=8,
            with_ray=with_ray,
            ray_template="/public/templates/ray.sub.j2",
            dry_run=True,
        )
    assert add_task.call_args.kwargs["ray_template"] == "/public/templates/ray.sub.j2"
    assert add_task.call_args.kwargs["with_ray"] is with_ray
    assert add_task.call_args.kwargs["num_nodes"] == 4
    assert add_task.call_args.kwargs["num_gpus"] == 8
    assert add_task.call_args.kwargs["cmd"] == [
        "export HYDRA_FULL_ERROR=1 && export PYTHONPATH=$PYTHONPATH:/nemo_run/code && cd /nemo_run/code && python driver.py "
    ]
    assert run_exp.call_args.kwargs["dry_run"] is True


def test_checked_in_ray_template_preserves_installed_source_and_environment(tmp_path):
    from pathlib import Path

    from nemo_run import SlurmExecutor
    from nemo_run.run.ray.slurm import SlurmRayRequest

    template = Path(__file__).parents[1] / "nemo_skills/pipeline/nemo_rl/ray_templates/nemo_skills_sandbox_ray.sub.j2"
    executor = SlurmExecutor(
        account="render-only",
        partition="batch",
        nodes=4,
        gpus_per_node=8,
        container_image="public-carrier",
        env_vars={
            "RAY_EXECUTABLE": "/public/driver-venv/bin/ray",
            "NRL_CONTAINER": "",
            "RAY_NODE_SETUP_SCRIPT": "/public/site/setup_image_packages.sh",
            "PYTHONPATH": "",
        },
    )
    request = SlurmRayRequest(
        name="speechlm-render-only",
        cluster_dir=str(tmp_path / "ray"),
        template_name=template.name,
        template_dir=str(template.parent),
        executor=executor,
        command="/public/driver-venv/bin/python driver.py",
        launch_cmd=["sbatch"],
    )
    rendered = request.materialize()
    assert rendered.count('if ! bash "\\$RAY_NODE_SETUP_SCRIPT"; then') == 2
    assert rendered.count("Node package setup failed; Ray was not started") == 2
    assert "export RAY_NODE_SETUP_SCRIPT=/public/site/setup_image_packages.sh" in rendered
    assert "nsight.py" not in rendered
    assert "sed -i" not in rendered
    assert "export RAY_EXECUTABLE=/public/driver-venv/bin/ray" in rendered
    assert '"${RAY_EXECUTABLE:-ray}" start --head' in rendered
    assert '"${RAY_EXECUTABLE:-ray}" start --address "$ip_head"' in rendered
    assert rendered.count('"${RAY_EXECUTABLE:-ray}" status') == 2
    assert "/public/driver-venv/bin/python driver.py" in rendered
    assert "#SBATCH --nodes=4" in rendered
    assert "#SBATCH --gpus-per-node=8" in rendered

    # The upstream Ray materializer ignores executor.srun_args for initial
    # head/worker containers. The template must override image defaults with
    # every explicitly declared key, including deliberately empty values.
    container_env = "--container-env=RAY_EXECUTABLE,NRL_CONTAINER,RAY_NODE_SETUP_SCRIPT,PYTHONPATH"
    for role in ("ray-head", "ray-worker-$i"):
        launch = next(
            line for line in rendered.splitlines() if "--container-name=" + role in line and "bash -x -c" in line
        )
        assert container_env in launch
    assert "export NRL_CONTAINER=" in rendered
    assert "export PYTHONPATH=" in rendered
    import subprocess

    subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
