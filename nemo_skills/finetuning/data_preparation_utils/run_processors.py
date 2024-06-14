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

import logging
import os
import tempfile
import uuid
from typing import List

import hydra
from omegaconf import OmegaConf, open_dict
from sdp.logging import logger

# registering new resolvers to simplify config files
OmegaConf.register_new_resolver("subfield", lambda node, field: node[field])
OmegaConf.register_new_resolver("not", lambda x: not x)
OmegaConf.register_new_resolver("equal", lambda field, value: field == value)


# customizing logger
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '[SDP %(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.handlers
logger.addHandler(handler)
logger.propagate = False


def select_subset(input_list: List, select_str: str) -> List:
    """This function parses a string and selects objects based on that.

    The string is expected to be a valid representation of Python slice. The
    only difference with using an actual slice is that we are always returning
    a list, never a single element. See examples below for more details.

    Examples::

        >>> processors_to_run = [1, 2, 3, 4, 5]
        >>> select_subset(processors_to_run, "3:") # to exclude first 3 objects
        [4, 5]

        >>> select_subset(processors_to_run, ":-1") # to select all but last
        [1, 2, 3, 4]

        >>> select_subset(processors_to_run, "2:5") # to select 3rd to 5th
        [3, 4, 5]

        >>> # note that unlike normal slice, we still return a list here
        >>> select_subset(processors_to_run, "0") # to select only the first
        [1]

        >>> select_subset(processors_to_run, "-1") # to select only the last
        [5]

    Args:
        input_list (list): input list to select objects from.
        select_str (str): string representing Python slice.

    Returns:
        list: a subset of the input according to the ``select_str``

    """
    if ":" not in select_str:
        selected_objects = [input_list[int(select_str)]]
    else:
        slice_obj = slice(*map(lambda x: int(x.strip()) if x.strip() else None, select_str.split(":")))
        selected_objects = input_list[slice_obj]
    return selected_objects


def run_processors(cfg):
    logger.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")
    processors_to_run = cfg.get("processors_to_run", "all")

    if processors_to_run == "all":
        processors_to_run = ":"
    selected_cfgs = select_subset(cfg.processors, processors_to_run)
    # filtering out any processors that have should_run=False
    processors_cfgs = []
    for processor_cfg in selected_cfgs:
        with open_dict(processor_cfg):
            should_run = processor_cfg.pop("should_run", True)
        if should_run:
            processors_cfgs.append(processor_cfg)

    logger.info(
        "Specified to run the following processors: %s ",
        [cfg["_target_"] for cfg in processors_cfgs],
    )
    processors = []
    # let's build all processors first to automatically check
    # for errors in parameters
    with tempfile.TemporaryDirectory() as tmp_dir:
        # special check for the first processor.
        # In case user selected something that does not start from
        # manifest creation we will try to infer the input from previous
        # output file
        if processors_cfgs[0] is not cfg.processors[0] and "input_manifest_file" not in processors_cfgs[0]:
            # locating starting processor
            for idx, processor in enumerate(cfg.processors):
                if processor is processors_cfgs[0]:  # we don't do a copy, so can just check object ids
                    if "output_manifest_file" in cfg.processors[idx - 1]:
                        with open_dict(processors_cfgs[0]):
                            processors_cfgs[0]["input_manifest_file"] = cfg.processors[idx - 1]["output_manifest_file"]
                    break

        for idx, processor_cfg in enumerate(processors_cfgs):
            logger.info('=> Building processor "%s"', processor_cfg["_target_"])

            # we assume that each processor defines "output_manifest_file"
            # and "input_manifest_file" keys, which can be optional. In case they
            # are missing, we create tmp files here for them
            # (1) first use a temporary file for the "output_manifest_file" if it is unspecified
            if "output_manifest_file" not in processor_cfg:
                tmp_file_path = os.path.join(tmp_dir, str(uuid.uuid4()))
                with open_dict(processor_cfg):
                    processor_cfg["output_manifest_file"] = tmp_file_path

            # (2) then link the current processor's output_manifest_file to the next processor's input_manifest_file
            # if it hasn't been specified (and if you are not on the last processor)
            if idx != len(processors_cfgs) - 1 and "input_manifest_file" not in processors_cfgs[idx + 1]:
                with open_dict(processors_cfgs[idx + 1]):
                    processors_cfgs[idx + 1]["input_manifest_file"] = processor_cfg["output_manifest_file"]

            processor = hydra.utils.instantiate(processor_cfg)
            # running runtime tests to fail right-away if something is not
            # matching users expectations
            processor.test()
            processors.append(processor)

        for processor in processors:
            # TODO: add proper str method to all classes for good display
            logger.info('=> Running processor "%s"', processor)
            processor.process()
