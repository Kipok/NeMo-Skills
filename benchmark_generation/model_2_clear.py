import re
import json
import argparse
import os

# Define the regex patterns and their replacements using raw strings
patterns = {
    r'\\\(': '(',          # \\( to (
    r'\\\)': ')',          # \\) to )
    r'\\left': '',         # \\left to empty string
    r'\\right': '',        # \\right to empty string
    r'\$': '',             # $ to empty string
    r'\\n': ' ',           # \n to space
    r'\(\|a\|\^3,\s*\\\\frac\{1\}\{\|a\|\};\s*\\\\frac\{1\}\{\|a\|\},\s*\|a\|\^3\)': r'((|a|^3, \\frac{1}{|a|}), (\\frac{1}{|a|}, |a|^3))'
}

def is_tuple(s):
    if isinstance(s, tuple):
        return True
    if not s.startswith('(') or not s.endswith(')'):
        return False
    
    stack = []
    contains_comma = False
    for i, char in enumerate(s):
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                return False
            stack.pop()
            if not stack and i != len(s) - 1:
                return False
        elif char == ',':
            contains_comma = True
    
    return not stack and contains_comma  # True if stack is empty and it contains a comma, False otherwise

def is_valid_tuple(s):
    # Check for balanced parentheses and presence of commas to identify tuples
    if not s.startswith('(') or not s.endswith(')'):
        return False

    stack = []
    contains_comma = False
    for char in s:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                return False
            stack.pop()
        elif char == ',':
            contains_comma = True
    
    return not stack and contains_comma

def process_generation_field(generation):
    # Apply each regex pattern and its replacement
    for pattern, replacement in patterns.items():
        generation = re.sub(pattern, replacement, generation)

    # Check if "cases" is in the string, skip tuple processing if it is
    if "cases" in generation:
        return generation

    # If not a valid tuple, try adding parentheses and checking again
    if not is_valid_tuple(generation):
        concatenated_generation = f"({generation})"
        if is_valid_tuple(concatenated_generation):
            print(f"Concatenated parentheses: \033[1mOriginal\033[0m: {generation}, \033[1mConcatenated\033[0m: {concatenated_generation}")
            generation = concatenated_generation

    # Strip unnecessary parentheses in a loop
    while generation.startswith('(') and generation.endswith(')') and not is_valid_tuple(generation):
        original_generation = generation
        generation = generation[1:-1].strip()
        print(f"Stripped parentheses: \033[1mOriginal\033[0m: {original_generation}, \033[1mStripped\033[0m: {generation}")

    return generation

def process_jsonl_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            record = json.loads(line.strip())
            record['generation'] = process_generation_field(record['generation'])
            outfile.write(json.dumps(record) + '\n')

def main(args):
    input_file = args.initial_book
    output_file = os.path.join(args.data_dir, 'processed_' + os.path.basename(input_file))
    process_jsonl_file(input_file, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate subsets of book tasks")
    parser.add_argument('initial_book', type=str, help="Path to the initial book JSONL file")
    parser.add_argument('data_dir', type=str, nargs='?', help="Directory to save the generated subsets (default: same directory as initial_book)")

    args = parser.parse_args()

    if not args.data_dir:
        args.data_dir = os.path.dirname(os.path.abspath(args.initial_book))

    main(args)
