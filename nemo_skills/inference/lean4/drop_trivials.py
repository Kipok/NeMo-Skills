import json
import sys

def drop_redundant_theorems(input_file1, input_file2, output_file1, output_file2):
    with open(input_file1, 'r', encoding='utf-8') as file1, \
         open(input_file2, 'r', encoding='utf-8') as file2, \
         open(output_file1, 'w', encoding='utf-8') as out_file1, \
         open(output_file2, 'w', encoding='utf-8') as out_file2:
        
        # Read both files line by line
        for line1, line2 in zip(file1, file2):
            data1 = json.loads(line1)
            data2 = json.loads(line2)
            
            # Check proof_status in both lines
            proof_status1 = data1.get('proof_status')
            proof_status2 = data2.get('proof_status')
            
            # Skip if both have "completed" status
            if proof_status1 == "completed" and proof_status2 == "completed":
                continue
            
            # Write to output files if either line has a non-"completed" status
            json.dump(data1, out_file1, ensure_ascii=False)
            out_file1.write('\n')
    
            json.dump(data2, out_file2, ensure_ascii=False)
            out_file2.write('\n')

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python drop_redundant_theorems.py <input_file1> <input_file2> <output_file1> <output_file2>")
        sys.exit(1)
    
    input_file1 = sys.argv[1]
    input_file2 = sys.argv[2]
    output_file1 = sys.argv[3]
    output_file2 = sys.argv[4]
    
    drop_redundant_theorems(input_file1, input_file2, output_file1, output_file2)
