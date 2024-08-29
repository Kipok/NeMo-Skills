import json
import argparse
from collections import defaultdict
from typing import List, Dict
import os

def process_files(file_paths: List[str]) -> Dict[str, Dict]:
    # Use both `id` and `output` as a composite key to uniquely identify each solution
    solutions = defaultdict(lambda: {'correct_count': 0, 'total_count': 0, 'data': None})
    
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                task_id = data['id']
                output = data['output']
                is_correct = data['is_correct']

                # Create a composite key using both task_id and output
                composite_key = f"{task_id}:{output}"
                
                solutions[composite_key]['correct_count'] += int(is_correct)
                solutions[composite_key]['total_count'] += 1
                
                if solutions[composite_key]['data'] is None:
                    solutions[composite_key]['data'] = {k: v for k, v in data.items() if k not in ['generation', 'is_correct', 'predicted_answer']}
    
    return solutions

def calculate_hardness(solutions: Dict[str, Dict], raw_fraction: bool, field_name: str) -> List[Dict]:
    result = []
    
    for solution_data in solutions.values():
        if raw_fraction:
            hardness = f"{solution_data['correct_count']}/{solution_data['total_count']}"
        else:
            hardness = solution_data['correct_count'] / solution_data['total_count']
        solution_data['data'][field_name] = hardness
        result.append(solution_data['data'])
    
    return result

def write_output(results: List[Dict], output_file: str):
    with open(output_file, 'w') as file:
        for result in results:
            json.dump(result, file)
            file.write('\n')

def main():
    parser = argparse.ArgumentParser(description="Calculate solution hardness across multiple runs.")
    parser.add_argument('files', nargs='+', help='Path to the output-rs files')
    parser.add_argument('output', nargs='?', default=None, help='Path to the output file (optional)')
    parser.add_argument('--raw_fraction', action='store_true', help='Keep hardness as a fraction instead of converting to float')
    parser.add_argument('--field_name', default='soln_hard', help='Name of the field to store the hardness value (default: soln_hard)')
    args = parser.parse_args()

    if args.output:
        output_file = args.output
    else:
        input_dir = os.path.dirname(args.files[0])
        output_file = os.path.join(input_dir, 'combined_results.jsonl')

    solutions = process_files(args.files)
    results = calculate_hardness(solutions, args.raw_fraction, args.field_name)
    write_output(results, output_file)
    
    print(f"Combined results written to {output_file}")

if __name__ == "__main__":
    main()
