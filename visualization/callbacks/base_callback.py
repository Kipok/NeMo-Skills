from typing import Tuple

from dash import html
from dash.dependencies import Input, Output

from callbacks import app
from layouts import (
    get_compare_test_layout,
    get_run_test_layout,
)


@app.callback(
    [
        Output("page_content", "children"),
        Output("run_mode_link", "active"),
        Output("analyze_link", "active"),
    ],
    [Input("url", "pathname")],
)
def nav_click(url: str) -> Tuple[html.Div, bool, bool]:
    if url == "/":
        return get_run_test_layout(), True, False
    elif url == "/analyze":
        return get_compare_test_layout(), False, True
