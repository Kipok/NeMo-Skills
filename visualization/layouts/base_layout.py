from dash import html, dcc
import dash_bootstrap_components as dbc


def get_main_page_layout() -> html.Div:
    return html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dbc.NavbarSimple(
                children=[
                    dbc.NavItem(
                        dbc.NavLink(
                            "Run model",
                            id="run_mode_link",
                            href="/",
                            active=True,
                        )
                    ),
                    dbc.NavItem(
                        dbc.NavLink(
                            "Analyze", id="analyze_link", href="/analyze"
                        )
                    ),
                ],
                brand="Data Explorer",
                sticky="top",
                color="blue",
                dark=True,
                class_name="mb-2",
            ),
            html.Div(id='dummy_output', style={'display': 'none'}, children=""),
            dbc.Container(id="page_content"),
        ]
    )
