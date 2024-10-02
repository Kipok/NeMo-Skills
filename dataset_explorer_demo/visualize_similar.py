import json
import random

import gradio as gr
from latex2mathml.converter import convert


def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def render_latex(text):
    # Convert LaTeX to MathML
    parts = text.split('$')
    for i in range(1, len(parts), 2):
        parts[i] = convert(parts[i])
    return ''.join(parts)


def display_entry(index, data):
    entry = data[index]
    test_problem = render_latex(entry['problem'])
    similar_training_problems = [render_latex(cand) for cand in entry['similar_items']]

    html = f"<h2>Test set problem:</h2><p>{test_problem}</p>"
    html += "<h2>Most similar training set problems:</h2><ol>"
    for cand in similar_training_problems:
        html += f"<li>{cand}</li>"
    html += "</ol>"

    return html


def random_entry(data):
    return random.randint(0, len(data) - 1)


# Load the JSONL file
data = load_jsonl('./similar-retrieved/math.jsonl')

with gr.Blocks() as demo:
    gr.Markdown("# JSONL Viewer with LaTeX Support")

    with gr.Row():
        index_input = gr.Number(label="Entry Index", value=0, step=1)
        random_button = gr.Button("Random Entry")

    output = gr.HTML()

    index_input.change(display_entry, inputs=[index_input, gr.State(data)], outputs=output)
    random_button.click(random_entry, inputs=gr.State(data), outputs=index_input)

demo.launch(debug=False, server_name='0.0.0.0', server_port=5005)
