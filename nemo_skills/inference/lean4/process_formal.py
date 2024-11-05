import json
import sys

def process_jsonl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as file, open(output_path, 'w', encoding='utf-8') as output_file:
        for line in file:
            data = json.loads(line)

            # Skip entries where proof_status is not "completed"
            if data.get('proof_status') != "completed":
                continue

            # Remove unnecessary fields
            data.pop('proof_status', None)
            data.pop('predicted_proof', None)

            # Rename 'generation' field to 'formal_statement' without modifying its content
            if 'generation' in data:
                data['formal_statement'] = data.pop('generation').lstrip("\n\n")  # Remove leading newlines

            # Write modified JSON to output file
            json.dump(data, output_file, ensure_ascii=False)
            output_file.write('\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_formal.py <input_file> <output_file>")
        print(f"Received: {sys.argv}")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    process_jsonl(input_file, output_file)