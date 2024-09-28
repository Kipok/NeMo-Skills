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


import typer

from nemo_skills.pipeline.app import app

# need the imports to make sure the commands are registered
from nemo_skills.pipeline.eval import eval

typer.main.get_command_name = lambda name: name


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    app()
