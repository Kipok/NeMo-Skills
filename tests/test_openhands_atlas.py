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

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nemo_skills.inference.eval.openhands_atlas import patch_atlas_local_runtime

RUNNER = """import os

def get_instance_docker_image(instance_id, swebench_official_image=False):
    repo, name = instance_id.split('__')
    return f'{repo}/{name}'

def get_config(instance, metadata):
    use_swebench_official_image = True
    base_container_image = get_instance_docker_image(
        instance['instance_id'],
        swebench_official_image=use_swebench_official_image,
    )
    return base_container_image
"""


class TestAtlasLocalRuntime(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.runner = Path(directory.name) / "run_infer.py"
        self.runner.write_text(RUNNER)

    def test_local_runtime_accepts_original_atlas_id(self):
        patch_atlas_local_runtime(self.runner)
        namespace = {}
        exec(compile(self.runner.read_text(), str(self.runner), "exec"), namespace)
        with patch.dict(os.environ, {"RUNTIME": "local"}):
            self.assertIsNone(namespace["get_config"]({"instance_id": "6905333b74f22949d97ba9be"}, None))

    def test_docker_runtime_keeps_image_selection(self):
        patch_atlas_local_runtime(self.runner)
        namespace = {}
        exec(compile(self.runner.read_text(), str(self.runner), "exec"), namespace)
        with patch.dict(os.environ, {"RUNTIME": "docker"}):
            self.assertEqual(namespace["get_config"]({"instance_id": "django__django-123"}, None), "django/django-123")

    def test_patch_is_idempotent(self):
        patch_atlas_local_runtime(self.runner)
        first = self.runner.read_text()
        patch_atlas_local_runtime(self.runner)
        self.assertEqual(self.runner.read_text(), first)

    def test_unknown_runner_is_rejected_without_changes(self):
        source = RUNNER.replace(
            "base_container_image = get_instance_docker_image(", "base_container_image = other_image("
        )
        self.runner.write_text(source)
        with self.assertRaisesRegex(RuntimeError, "Unrecognized"):
            patch_atlas_local_runtime(self.runner)
        self.assertEqual(self.runner.read_text(), source)


if __name__ == "__main__":
    unittest.main()
