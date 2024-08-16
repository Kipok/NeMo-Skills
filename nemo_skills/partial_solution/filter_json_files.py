import json
import os
import glob
import argparse

def load_small_subset_ids(small_subset_file):
    ids = set()
    with open(small_subset_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'id' in data:
                    ids.add(data['id'])
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line in {small_subset_file}: {line.strip()}")
    return ids

def process_jsonl_file(input_file, output_file, small_subset_ids):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            try:
                data = json.loads(line.strip())
                if 'id' in data and data['id'] in small_subset_ids:
                    outfile.write(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line in {input_file}: {line.strip()}")

def filter_jsonl_files(small_subset_file, folder_path):
    small_subset_ids = load_small_subset_ids(small_subset_file)
    
    for jsonl_file in glob.glob(os.path.join(folder_path, '**', '*.jsonl'), recursive=True):
        rel_path = os.path.relpath(jsonl_file, folder_path)
        output_file = os.path.join(folder_path, 'small', rel_path)
        output_file = output_file.replace('.jsonl', '_small.jsonl')
        
        process_jsonl_file(jsonl_file, output_file, small_subset_ids)
        print(f"Processed: {jsonl_file} -> {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter JSONL files based on IDs from a small subset file.")
    parser.add_argument("small_subset_file", help="Path to the small subset JSONL file")
    parser.add_argument("folder_path", help="Path to the folder containing JSONL files to filter")
    
    args = parser.parse_args()

    filter_jsonl_files(args.small_subset_file, args.folder_path)