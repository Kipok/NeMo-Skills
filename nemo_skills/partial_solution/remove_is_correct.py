import json
import argparse
import os

def remove_is_correct(file_path):
    temp_file_path = file_path + '.temp'
    
    with open(file_path, 'r') as input_file, open(temp_file_path, 'w') as output_file:
        for line in input_file:
            data = json.loads(line)
            if 'is_correct' in data:
                del data['is_correct']
            if 'predicted_answer' in data:
                del data['predicted_answer']
            output_file.write(json.dumps(data) + '\n')
    
    # Replace the original file with the modified one
    os.replace(temp_file_path, file_path)
    print(f"Processed: {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Remove 'is_correct' key from data files.")
    parser.add_argument('files', nargs='+', help='Paths to the data files')
    args = parser.parse_args()

    for file_path in args.files:
        remove_is_correct(file_path)

if __name__ == "__main__":
    main()