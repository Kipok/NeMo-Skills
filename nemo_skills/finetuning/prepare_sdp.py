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

import sys

import hydra
from data_preparation_utils.run_processors import run_processors


@hydra.main(version_base=None)
def main(cfg):
    run_processors(cfg)


if __name__ == "__main__":
    # hacking the arguments to always disable hydra's output
    # TODO: maybe better to copy-paste hydra_runner from nemo if there are
    #    any problems with this approach
    sys.argv.extend(
        ["hydra.run.dir=.", "hydra.output_subdir=null", "hydra/job_logging=none", "hydra/hydra_logging=none"]
    )
    main()
