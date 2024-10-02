import json
import os
import random
import re

import gradio as gr
from latex2mathml.converter import convert
from latex2mathml.exceptions import NoAvailableTokensError


def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def render_latex(text):
    def replace_matrix(match):
        matrix_content = match.group(1)
        rows = matrix_content.split('\\\\')
        mml_rows = ''.join(f'<mtr><mtd>{convert_and_clean(row.strip())}</mtd></mtr>' for row in rows)
        return f'<mrow><mo>(</mo><mtable rowspacing="4pt" columnspacing="1em">{mml_rows}</mtable><mo>)</mo></mrow>'

    def convert_and_clean(latex):
        try:
            # Convert to MathML
            mathml = convert(latex)
            # Remove <math> tags if present
            mathml = re.sub(r'<math.*?>(.*)</math>', r'\1', mathml)
            return mathml
        except NoAvailableTokensError:
            # Return the original LaTeX if conversion fails
            return latex

    # First, handle matrix notation
    text = re.sub(r'\\begin{pmatrix}(.*?)\\end{pmatrix}', replace_matrix, text, flags=re.DOTALL)

    # Handle multi-line LaTeX expressions
    text = re.sub(
        r'\[(.*?)\]',
        lambda m: f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">{convert_and_clean(m.group(1))}</math>',
        text,
        flags=re.DOTALL,
    )

    # Handle inline LaTeX
    text = re.sub(
        r'\$(.*?)\$',
        lambda m: f'<math xmlns="http://www.w3.org/1998/Math/MathML">{convert_and_clean(m.group(1))}</math>',
        text,
    )

    return text


def display_entry(index, data_openmath2, data_math_train):
    entry_openmath2 = data_openmath2[index]
    entry_math_train = data_math_train[index]
    test_problem = render_latex(entry_openmath2['problem'])
    similar_openmath2 = [render_latex(cand) for cand in entry_openmath2['similar_items']]
    similar_math_train = [render_latex(cand) for cand in entry_math_train['similar_items']]

    html = f"<h2>Test set problem:</h2><p>{test_problem}</p>"
    html += "<hr>"  # Add horizontal bar
    html += "<div style='display: flex;'>"
    html += "<div style='flex: 1; padding-right: 10px;'>"
    html += "<h2>Most similar OpenMathInstruct-2 problems:</h2><ol>"
    for cand in similar_openmath2:
        html += f"<li>{cand}</li>"
    html += "</ol></div>"
    html += "<div style='border-left: 1px solid #ccc;'></div>"
    html += "<div style='flex: 1; padding-left: 10px;'>"
    html += "<h2>Most similar MATH training set problems:</h2><ol>"
    for cand in similar_math_train:
        html += f"<li>{cand}</li>"
    html += "</ol></div>"
    html += "</div>"

    return html


def random_entry(data):
    return random.randint(0, len(data) - 1)


def load_test_sets(test_set):
    file_path_openmath2 = f'./similar-retrieved-openmath2/{test_set}'
    file_path_math_train = f'./similar-retrieved-math-train/{test_set}'
    return load_jsonl(file_path_openmath2), load_jsonl(file_path_math_train)


# Get list of test sets (assuming both folders have the same files)
test_sets = [f for f in os.listdir('./similar-retrieved-openmath2') if f.endswith('.jsonl')]
test_set_names = [os.path.splitext(f)[0] for f in test_sets]

with gr.Blocks() as demo:
    gr.Markdown("# OpenMathInstruct-2 test set contamination explorer")
    gr.Markdown("Dataset: [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)")

    with gr.Row():
        test_set_dropdown = gr.Dropdown(choices=test_set_names, label="Select Test Set", value=test_set_names[0])
        index_input = gr.Number(label="Problem Index", value=0, step=1)
        random_button = gr.Button("Random Problem")

    output = gr.HTML()

    data_openmath2 = gr.State(load_test_sets(test_sets[0])[0])
    data_math_train = gr.State(load_test_sets(test_sets[0])[1])

    def update_test_set(test_set):
        new_data_openmath2, new_data_math_train = load_test_sets(f"{test_set}.jsonl")
        return new_data_openmath2, new_data_math_train, 0, display_entry(0, new_data_openmath2, new_data_math_train)

    def display_entry_wrapper(index, data_openmath2, data_math_train):
        return display_entry(index, data_openmath2, data_math_train)

    def random_entry_wrapper(data_openmath2):
        return random_entry(data_openmath2)

    test_set_dropdown.change(
        update_test_set, inputs=[test_set_dropdown], outputs=[data_openmath2, data_math_train, index_input, output]
    )
    index_input.change(display_entry_wrapper, inputs=[index_input, data_openmath2, data_math_train], outputs=output)
    random_button.click(random_entry_wrapper, inputs=[data_openmath2], outputs=index_input)

    # Initial display
    demo.load(display_entry_wrapper, inputs=[index_input, data_openmath2, data_math_train], outputs=output)

demo.launch(debug=False, server_name='0.0.0.0', server_port=5005)
