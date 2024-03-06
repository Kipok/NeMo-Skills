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

import os
from dataclasses import asdict, fields
from pathlib import Path

import dash_bootstrap_components as dbc
import hydra
from dash import Dash
from flask import Flask
from omegaconf import MISSING, MissingMandatoryValue, OmegaConf
from settings.config import Config

from nemo_skills.inference.prompt.utils import get_prompt_config
from visualization.settings.constants import UNDEFINED

config_path = os.path.join(os.path.abspath(Path(__file__).parent.parent), "settings")

config = {}


@hydra.main(version_base=None, config_path=config_path, config_name="config")
def set_config(cfg: Config) -> None:
    global config
    cfg.output_file = ""
    new_prompt = get_prompt_config('code_sfted')
    for param in fields(new_prompt):
        try:
            setattr(new_prompt, param.name, getattr(cfg.prompt, param.name))
        except MissingMandatoryValue:
            pass
    cfg.prompt = new_prompt
    if not cfg.data_file and not cfg.dataset and not cfg.split_name:
        cfg.data_file = UNDEFINED

    config['data_explorer'] = asdict(OmegaConf.to_object(cfg))

    for param in ['host', 'ssh_server', 'ssh_key_path']:
        if param not in config['data_explorer']['sandbox'] and param in config['data_explorer']['server']:
            config['data_explorer']['sandbox'][param] = config['data_explorer']['server'][param]
    # All parameters in config can be modified through the application except Server and Sandbox configs
    # Following parameters are not used and should not be modified
    config['data_explorer'].pop('output_file')
    config['data_explorer'].pop('dataset')
    config['data_explorer'].pop('split_name')
    config['data_explorer']['prompt'].pop(
        "examples"
    )  # Can be modified separately through Few Shots accordion component
    config['data_explorer']['prompt'].pop("context")


set_config()
server = Flask(__name__)
server.config.update(config)

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    server=server,
)

from callbacks.base_callback import nav_click
from callbacks.run_prompt_callbacks import add_example
from callbacks.table_callbacks import choose_base_model
