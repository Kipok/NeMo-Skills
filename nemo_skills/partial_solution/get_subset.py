import json
import random
import os
import argparse

def create_small_subset(input_file, output_file, subset_size=100):
    # Check if the output file exists and is not empty
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"{output_file} already exists and is not empty. Skipping subset creation.")
        return

    # Read all valid JSON lines from the input file
    valid_lines = []
    with open(input_file, 'r') as f:
        for line in f:
            try:
                json.loads(line.strip())
                valid_lines.append(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")

    # Randomly select subset_size number of lines
    subset = random.sample(valid_lines, min(subset_size, len(valid_lines)))

    # Write the selected lines to the output file
    with open(output_file, 'w') as f:
        f.writelines(subset)

    print(f"Created {output_file} with {len(subset)} randomly selected questions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a small subset of a JSONL file.")
    parser.add_argument("input_file", help="Path to the input JSONL file")
    parser.add_argument("output_file", help="Path to the output JSONL file")
    parser.add_argument("-n", "--subset_size", type=int, default=100, help="Number of items in the subset (default: 100)")
    
    args = parser.parse_args()

    create_small_subset(args.input_file, args.output_file, args.subset_size)