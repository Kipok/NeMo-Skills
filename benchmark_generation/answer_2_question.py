import json
import os
import argparse

def process_json(file_path, output_dir):
    processed_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            entry['task'] = entry.pop('question')
            entry['question'] = entry.pop('original_answer')
            processed_data.append(entry)
    
    output_file = os.path.join(output_dir, 'processed_' + os.path.basename(file_path))
    
    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in processed_data:
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Processed data saved to {output_file}")

def main(args):
    process_json(args.initial_book, args.data_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate subsets of book tasks")
    parser.add_argument('initial_book', type=str, help="Path to the initial book JSON file")
    parser.add_argument('data_dir', type=str, nargs='?', help="Directory to save the generated subsets (default: same directory as initial_book)")

    args = parser.parse_args()

    if not args.data_dir:
        args.data_dir = os.path.dirname(os.path.abspath(args.initial_book))

    main(args)
