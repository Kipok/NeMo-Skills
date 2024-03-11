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

import random
import re
import string

from ansi2html import Ansi2HTMLConverter
from dash import dcc, html
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


def design_text_output(text: str, style={}):
    conv = Ansi2HTMLConverter()
    text = conv.convert(text, full=False)

    def add_marker_to_lines(match):
        lines = match.group(1).strip().split('\n')
        if len(lines) == 1:
            marked_lines = lines
        else:
            marked_lines = [
                ("$$" + line if i == len(lines) - 1 else line + "$$" if i == 0 else "$$" + line + "$$")
                for i, line in enumerate(lines)
            ]
        return '$$' + '\n'.join(marked_lines) + '$$'

    display_math_pattern = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)

    formatted_text = re.sub(
        display_math_pattern,
        add_marker_to_lines,
        text.replace("*", "\*")
        .replace("#", "\#")
        .replace("_", "\_")
        .replace("[", "\[")
        .replace("]", "\]")
        .replace("\\\\", "\\"),
    )
    return html.Div(
        [
            dcc.Markdown(
                line,
                mathjax=True,
                dangerously_allow_html=True,
            )
            for line in formatted_text.split('\n')
        ],
        style=style,
    )


def highlight_code(code: str) -> html.Iframe:
    highlighted_code = highlight(code, PythonLexer(), HtmlFormatter())

    iframe_style = {
        "width": "100%",
        "border": "none",
        "overflow": "hidden",
        "border": "black 1px solid",
        "background-color": "#ebecf0d8",
    }

    iframe_id = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
    return html.Iframe(
        id=iframe_id,
        srcDoc=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                {HtmlFormatter().get_style_defs()}
            </style>
        </head>
        <body>
            <pre class="highlight">{highlighted_code}</pre>
            <script>
                function updateHeight() {{
                    var body = document.body,
                        html = document.documentElement;
                    
                    var height = Math.max(body.scrollHeight, body.offsetHeight,
                                          html.clientHeight, html.scrollHeight, html.offsetHeight);
                    
                    parent.postMessage({{ frameHeight: height, frameId: '{iframe_id}' }}, '*');
                }}
                window.onload = updateHeight;
                window.onresize = updateHeight;
            </script>
        </body>
        </html>
        """,
        style=iframe_style,
    )
