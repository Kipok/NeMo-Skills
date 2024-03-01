import os
from pathlib import Path

import dash_bootstrap_components as dbc
import hydra
from dash import Dash
from flask import Flask
from omegaconf import OmegaConf
from settings.config import Config

from nemo_skills.utils import unroll_files

config_path = os.path.join(os.path.abspath(Path(__file__).parent.parent), "settings")

config = {}


@hydra.main(version_base=None, config_path=config_path, config_name="config")
def set_config(cfg: Config) -> None:
    global config
    config['prompt_explorer'] = OmegaConf.to_container(cfg)

    config['prompt_explorer']['inference']['start_random_seed'] = (
        config['prompt_explorer']['inference']['start_random_seed']
        if 'start_random_seed' in config['prompt_explorer']['inference']
        else 0
    )
    config['prompt_explorer']['visualization_params']['prediction_jsonl_files'] = {
        model_name: list(unroll_files(file_path.split(" ")))
        for model_name, file_path in config['prompt_explorer']['visualization_params'][
            'prediction_jsonl_files'
        ].items()
    }


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
