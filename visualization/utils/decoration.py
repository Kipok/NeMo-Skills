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
from typing import Dict

from ansi2html import Ansi2HTMLConverter
from dash import dcc, html
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


def design_text_output(text: str, style={}):
    conv = Ansi2HTMLConverter()
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')

    if bool(ansi_escape.search(text)):
        text = conv.convert(text, full=False)
        return html.Div(
            iframe_template(
                '<link rel="stylesheet" type="text/css" href="/assets/ansi_styles.css">',
                f'<pre>{text}</pre>',
            ),
            style=style,
        )
    elif 'ipython-input' in text or 'Traceback' in text:
        text = conv.convert(text.replace('[', '\u001b['), full=False)
        return html.Div(
            iframe_template(
                '<link rel="stylesheet" type="text/css" href="/assets/ansi_styles.css">',
                f'<pre>{text}</pre>',
            ),
            style=style,
        )

    def preprocess_latex(text):
        text = text.replace('\\[', '\n$$\n').replace('\\]', '\n$$\n')

        special_chars = set(r'*_{}[]()#+-.!')
        begin_tag = '\\begin'
        end_tag = '\\end'
        begin_tag_index = -1
        math_block_flags = {'$$': False, '$': False}
        begin_end_counter = 0
        index = 0
        string_modifications = [(0, "")]

        def handle_braces(index):
            while index < len(text) and text[index] == '{':
                while index < len(text) and text[index] != '}':
                    index += 1
                if index < len(text) - 1 and text[index + 1] == '{':
                    index += 1
            return index

        while index < len(text):
            if text.startswith(begin_tag, index):
                if begin_end_counter == 0:
                    begin_tag_index = index
                begin_end_counter += 1
                index += len(begin_tag)
                index = handle_braces(index)
            elif text.startswith(end_tag, index):
                begin_end_counter -= 1
                index += len(end_tag)
                index = handle_braces(index)
                if begin_end_counter == 0:
                    string_modifications.append((begin_tag_index, "\n$$\n"))
                    string_modifications.append((index + 1, "\n$$\n"))
            elif text.startswith("$$", index):
                math_block_flags['$$'] = not math_block_flags['$$']
                index += 1
            elif text[index] == "$":
                math_block_flags['$'] = not math_block_flags['$']
            elif not any(math_block_flags.values()) and begin_end_counter == 0:
                if text[index] in special_chars:
                    string_modifications.append((index, "\\"))
            index += 1

        string_pieces = []
        string_modifications.sort()
        for i in range(len(string_modifications)):
            if i != 0:
                start, end = string_modifications[i - 1][0], string_modifications[i][0]
                string_pieces.append(text[start:end])
                string_pieces.append(string_modifications[i][1])
        string_pieces.append(text[string_modifications[-1][0] :])

        return "".join(string_pieces)

    return html.Div(
        dcc.Markdown(preprocess_latex(text), mathjax=True, dangerously_allow_html=True),
        style=style,
    )


def update_height_js(iframe_id: str) -> str:
    return f"""
        function updateHeight() {{
            var body = document.body,
                html = document.documentElement;
            
            var height = Math.max(body.scrollHeight, body.offsetHeight,
                                    html.clientHeight, html.scrollHeight, html.offsetHeight);
            
            parent.postMessage({{ frameHeight: height, frameId: '{iframe_id}' }}, '*');
        }}
        window.onload = updateHeight;
        window.onresize = updateHeight;
    """


def iframe_template(header: str, content: str, style: Dict = {}, iframe_id: str = None) -> html.Iframe:
    if not iframe_id:
        iframe_id = get_random_id()

    iframe_style = {
        "width": "100%",
        "border": "none",
        "overflow": "hidden",
    }

    iframe_style.update(style)

    return html.Iframe(
        id=iframe_id,
        srcDoc=f"""
        <!DOCTYPE html>
        <html>
        <head>
            {header}
        </head>
        <body>
            {content}
        </body>
        </html>
        """,
        style=iframe_style,
    )


def get_random_id() -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=20))


def highlight_code(code: str) -> html.Iframe:
    highlighted_code = highlight(code, PythonLexer(), HtmlFormatter())
    iframe_id = get_random_id()
    return iframe_template(
        header=f"<style>{HtmlFormatter().get_style_defs()}</style>",
        content=f"""<pre class='highlight'>{highlighted_code}</pre>
        <script>{update_height_js(iframe_id)}</script>""",
        iframe_id=iframe_id,
        style={"border": "black 1px solid", "background-color": "#ebecf0d8"},
    )
