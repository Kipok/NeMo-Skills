import json
import random
import argparse
import os
import textwrap

def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def wrap_text(text, width=80):
    """Wrap text to a specified width, preserving newlines."""
    # Split the text into paragraphs
    paragraphs = text.split('\n')
    # Wrap each paragraph
    wrapped_paragraphs = [textwrap.fill(p, width=width) for p in paragraphs]
    # Join the wrapped paragraphs back together
    return '\n'.join(wrapped_paragraphs)

def format_problem(problem, index):
    separator = "=" * 80
    header = f"Problem {index}".center(80, "=")
    
    question = wrap_text(problem.get("question", "N/A"))
    solution = wrap_text(problem.get("solution", "N/A"))
    generation = wrap_text(problem.get("generation", "N/A"))
    
    formatted_problem = f"""
{header}

QUESTION:
{question}

{'-' * 80}

SOLUTION:
{solution}

{'-' * 80}

GENERATION:
{generation}

{separator}
"""
    return formatted_problem

def extract_and_format_problems(initial_book, data_dir):
    problems = read_jsonl(initial_book)
    
    if len(problems) < 30:
        raise ValueError("Not enough problems in the file to sample 30.")
    
    selected_problems = random.sample(problems, 30)
    
    formatted_content = ""
    for i, problem in enumerate(selected_problems, start=1):
        formatted_content += format_problem(problem, i)
    
    if data_dir is None:
        data_dir = os.path.dirname(initial_book)
    output_file = os.path.join(data_dir, 'formatted_problems.txt')
    
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(formatted_content)
    
    print(f"Formatted problems written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract random problems from JSONL file and format them for easy reading.")
    parser.add_argument('initial_book', type=str, help="Path to the initial book JSONL file")
    parser.add_argument('data_dir', type=str, nargs='?', default=None, help="Directory to save the generated output (default: same directory as initial_book)")

    args = parser.parse_args()
    extract_and_format_problems(args.initial_book, args.data_dir)