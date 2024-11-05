import json
import sys

def replace_with_not_equal(theorem_statement):
    """
    Replace the first '=' before ':= by' in theorem_statement with '≠'.
    """
    # Locate the position of ':= by'
    replacement_start = theorem_statement.find(":= by")
    if replacement_start == -1:
        return theorem_statement  # Return as-is if ':= by' is not found

    # Find the first '=' moving backwards from the start of ':= by'
    equal_position = theorem_statement.rfind("=", 0, replacement_start)
    if equal_position == -1:
        return theorem_statement  # Return as-is if '=' is not found before ':= by'

    # Replace '=' with '≠' and return the modified theorem statement
    return theorem_statement[:equal_position] + "≠" + theorem_statement[equal_position + 1:]

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

            # Rename 'generation' field to 'formal_statement_leg' and modify content
            if 'generation' in data:
                # Store original generation content in 'formal_statement_leg'
                theorem_statement = data.pop('generation')
                data['formal_statement_leg'] = theorem_statement.lstrip("\n\n")  # Remove leading newlines
                
                # Create 'formal_statement' with modified content
                data['formal_statement'] = replace_with_not_equal(data['formal_statement_leg'])
            
            # Write modified JSON to output file
            json.dump(data, output_file, ensure_ascii=False)
            output_file.write('\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_contr.py <input_file> <output_file>")
        print(f"Received: {sys.argv}")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    process_jsonl(input_file, output_file)