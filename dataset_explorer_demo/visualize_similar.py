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
    question = render_latex(entry['question'])
    candidates = [render_latex(cand['question']) for cand in entry['top_k_similar_example']]

    html = f"<h2>Question:</h2><p>{question}</p>"
    html += "<h2>Candidates:</h2><ol>"
    for cand in candidates:
        html += f"<li>{cand}</li>"
    html += "</ol>"

    return html


def random_entry(data):
    return random.randint(0, len(data) - 1)


# Load the JSONL file
data = load_jsonl('test-contam.jsonl')
# Filter the data to keep only the required keys
filtered_data = []
for entry in data:
    filtered_entry = {
        'question': entry['question'],
        'top_k_similar_example': [{'question': cand['question']} for cand in entry['top_k_similar_example']],
    }
    filtered_data.append(filtered_entry)

data = filtered_data

with gr.Blocks() as demo:
    gr.Markdown("# JSONL Viewer with LaTeX Support")

    with gr.Row():
        index_input = gr.Number(label="Entry Index", value=0, step=1)
        random_button = gr.Button("Random Entry")

    output = gr.HTML()

    index_input.change(display_entry, inputs=[index_input, gr.State(data)], outputs=output)
    random_button.click(random_entry, inputs=gr.State(data), outputs=index_input)

demo.launch(debug=False, server_name='0.0.0.0', server_port=5005)
