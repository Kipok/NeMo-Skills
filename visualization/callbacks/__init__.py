from dash import Dash
import dash_bootstrap_components as dbc

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

from callbacks.base_callback import nav_click
from callbacks.run_prompt_callbacks import (
    add_example,
)
from callbacks.table_callbacks import (
    choose_base_model,
)
