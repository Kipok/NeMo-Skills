import json
import re
import argparse
import os

# Function to load the JSON data from a JSON Lines file
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    print(f"Loaded {len(data)} problems from {file_path}")
    return data

# Function to check if a string is a tuple
def is_tuple(s):
    if isinstance(s, tuple):
        return True
    if not s.startswith('('):
        return False
    
    stack = []
    for i, char in enumerate(s):
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                return False
            stack.pop()
            if not stack and i != len(s) - 1:
                return False
    return not stack  # True if stack is empty, False otherwise

# Function to convert multiple answers to tuple format without quotes
def convert_to_tuple(answer):
    if isinstance(answer, list):
        return tuple(answer)
    
    # Remove whitespace around separators
    answer = re.sub(r'\s*;\s*', ';', answer)
    answer = re.sub(r'\s*,\s*', ',', answer)

    # Handle ';' separator for multiple answers
    if ';' in answer:
        groups = answer.split(';')
        return tuple(map(str.strip, groups))
    
    # Handle ',' separator for multiple values within one answer
    elif ',' in answer:
        return tuple(map(str.strip, answer.split(',')))

    return answer

# Function to check if an answer contains the specified inequalities
def contains_inequalities(answer):
    patterns = [
        r'.*\\le[^f].*',  # Matches \le not followed by f
        r'.*\\ge[^f].*',  # Matches \ge not followed by f
        r'.*<.*',         # Matches <
        r'.*>.*'          # Matches >
    ]
    for pattern in patterns:
        if re.search(pattern, answer):
            return True
    return False

# Function to process the problems and update expected answers
def process_problems(data, output_file, all_processed_file, skip_ids, inequality_file, auto_confirm):
    relevant_tasks = [problem for problem in data if ';' in str(problem.get('expected_answer', '')) or ',' in str(problem.get('expected_answer', ''))]
    total_tasks = len(relevant_tasks)
    done_tasks = 0
    updated_data = []
    all_processed_data = []

    for problem in data:
        problem_id = problem.get('problem_id')
        expected_answer = problem.get('expected_answer')
        
        if problem_id in skip_ids:
            all_processed_data.append(problem)
            continue

        if contains_inequalities(expected_answer):
            save_inequality_problem(problem, inequality_file)
            continue

        if ';' in str(expected_answer) or ',' in str(expected_answer):
            original_answer = expected_answer
            new_answer = convert_to_tuple(expected_answer)
            new_answer_str = f"({', '.join(map(str, new_answer))})"
            
            # Strip the outer parentheses and check if it is still a tuple
            stripped_answer = new_answer_str[1:-1].strip()
            if is_tuple(f"{stripped_answer}"):
                final_answer = stripped_answer
            else:
                final_answer = new_answer_str

            if auto_confirm:
                problem['expected_answer'] = final_answer
            else:
                print(f"\033[1mProblem ID:\033[0m {problem_id}")
                print(f"\033[1mOriginal expected_answer:\033[0m {original_answer}")
                print(f"\033[1mConverted to tuple format:\033[0m {final_answer}")
                user_input = input("Press Enter to confirm, type 'skip' to skip, or provide a new answer: ")
                if user_input.lower() == 'skip':
                    skip_ids.add(problem_id)
                    all_processed_data.append(problem)
                    continue
                elif user_input:
                    problem['expected_answer'] = f"({', '.join(map(str.strip, user_input.split(',')))})"
                else:
                    problem['expected_answer'] = final_answer

            updated_data.append(problem)

        all_processed_data.append(problem)
        done_tasks += 1
        print(f"Progress: {done_tasks}/{total_tasks}")

    save_data(updated_data, output_file)
    save_data(all_processed_data, all_processed_file)
    return updated_data, skip_ids

# Function to save problems with inequalities to a separate file immediately
def save_inequality_problem(problem, file_path):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(problem) + '\n')
    print(f"Saved inequality problem ID {problem['problem_id']} to {file_path}")

# Function to save the updated data back to a JSON Lines file
def save_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for problem in data:
            file.write(json.dumps(problem) + '\n')
    print(f"Saved {len(data)} problems to {file_path}")

# Function to load skip ids from a file
def load_skip_ids(skip_file_path):
    if os.path.exists(skip_file_path):
        with open(skip_file_path, 'r', encoding='utf-8') as file:
            skip_ids = set(json.load(file))
        print(f"Loaded skip IDs from {skip_file_path}")
    else:
        skip_ids = set()
    return skip_ids

# Function to save skip ids to a file
def save_skip_ids(skip_ids, skip_file_path):
    with open(skip_file_path, 'w', encoding='utf-8') as file:
        json.dump(list(skip_ids), file)
    print(f"Saved skip IDs to {skip_file_path}")

# Main function to orchestrate the process
def main(args):
    tuple_output_file = os.path.join(args.data_dir, 'tuple_answer_problems.json')
    all_processed_file = os.path.join(args.data_dir, 'all_processed_problems.json')
    skip_file = os.path.join(args.data_dir, 'skip_ids.json')
    inequality_file = os.path.join(args.data_dir, 'inequality_problems.json')
    
    if args.continue_processing and os.path.exists(tuple_output_file):
        print(f"Resuming from {tuple_output_file}")
        data = load_data(tuple_output_file)
    else:
        print(f"Starting fresh processing, removing old files if they exist.")
        # Remove old files if they exist
        for file_path in [tuple_output_file, all_processed_file, skip_file, inequality_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
        data = load_data(args.initial_book)

    skip_ids = load_skip_ids(skip_file)
    updated_data, skip_ids = process_problems(data, tuple_output_file, all_processed_file, skip_ids, inequality_file, args.auto_confirm)
    
    save_skip_ids(skip_ids, skip_file)

    if args.modify_original:
        save_data(updated_data, args.initial_book)
    else:
        save_data(updated_data, tuple_output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process book tasks and update expected answers")
    parser.add_argument('initial_book', type=str, help="Path to the initial book JSON file")
    parser.add_argument('data_dir', type=str, nargs='?', help="Directory to save the updated JSON file (default: same directory as initial_book)")
    parser.add_argument('--modify-original', action='store_true', help="Modify the original file instead of creating a new one")
    parser.add_argument('--continue-processing', action='store_true', help="Continue from the last saved state or start over")
    parser.add_argument('--auto-confirm', action='store_true', help="Automatically confirm and process answers")

    args = parser.parse_args()
    
    if not args.data_dir:
        args.data_dir = os.path.dirname(os.path.abspath(args.initial_book))

    main(args)
