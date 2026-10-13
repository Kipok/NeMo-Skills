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

"""Adapt the staged OpenHands SWE-bench runner for Atlas' local runtime."""

import argparse
import ast
from pathlib import Path


def patch_atlas_local_runtime(runner: Path) -> None:
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    config = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_config"), None
    )
    if config is None:
        raise RuntimeError("OpenHands runner does not define get_config.")
    assignments = [
        node
        for node in config.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "base_container_image"
    ]
    if len(assignments) != 1:
        raise RuntimeError("Expected one OpenHands base_container_image assignment.")
    original = assignments[0].value
    condition = ast.parse("os.environ.get('RUNTIME') == 'local'", mode="eval").body
    if (
        isinstance(original, ast.IfExp)
        and ast.dump(original.test) == ast.dump(condition)
        and isinstance(original.body, ast.Constant)
        and original.body.value is None
    ):
        return
    if not (
        isinstance(original, ast.Call)
        and isinstance(original.func, ast.Name)
        and original.func.id == "get_instance_docker_image"
    ):
        raise RuntimeError("Unrecognized OpenHands image selection; refusing to patch.")

    # NeMo-Skills already started the Atlas task container with Apptainer.
    # LocalRuntime does not need a Docker image, and Atlas IDs are not SWE-bench IDs.
    replacement = ast.unparse(ast.IfExp(test=condition, body=ast.Constant(value=None), orelse=original))
    lines = source.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: original.lineno - 1]) + original.col_offset
    end = sum(len(line) for line in lines[: original.end_lineno - 1]) + original.end_col_offset
    patched = source[:start] + replacement + source[end:]
    ast.parse(patched)
    runner.write_text(patched, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    patch_atlas_local_runtime(parser.parse_args().runner)
