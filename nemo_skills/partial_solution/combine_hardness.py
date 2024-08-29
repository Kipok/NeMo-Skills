import json
import argparse
from collections import defaultdict
from typing import List, Dict
import os

def merge_jsonl_files(file_paths: List[str], merge_fields: List[str]) -> Dict[str, Dict]:
    merged_data = defaultdict(dict)
    
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                # Create a composite key using the merge fields
                key = tuple(data.get(field) for field in merge_fields)
                if None not in key:  # Ensure all fields are present
                    merged_data[key].update(data)
    
    return merged_data

def write_output(merged_data: Dict[str, Dict], output_file: str):
    with open(output_file, 'w') as file:
        for data in merged_data.values():
            json.dump(data, file)
            file.write('\n')

def main():
    parser = argparse.ArgumentParser(description="Merge multiple .jsonl files based on specified fields.")
    parser.add_argument('files', nargs='+', help='Paths to the input .jsonl files')
    parser.add_argument('--output', default=None, help='Path to the output file (optional)')
    parser.add_argument('--merge_fields', nargs='+', default=['id', 'output'], help='Field names to use for merging (default: id and output)')
    args = parser.parse_args()

    if args.output:
        output_file = args.output
    else:
        input_dir = os.path.dirname(args.files[0])
        output_file = os.path.join(input_dir, 'merged_results.jsonl')

    merged_data = merge_jsonl_files(args.files, args.merge_fields)
    write_output(merged_data, output_file)
    
    print(f"Merged results written to {output_file}")

if __name__ == "__main__":
    main()
