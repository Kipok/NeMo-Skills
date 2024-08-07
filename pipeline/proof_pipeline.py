import argparse
import subprocess
import time
import os
import json

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode(), stderr.decode()

def check_output_file(output_file):
    print(f"Checking output file: {output_file}")
    try:
        if not os.path.exists(output_file):
            print(f"File not found: {output_file}")
            return False
        
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "generation" in data:
                        print("'generation' key found in the data")
                        return True
                except json.JSONDecodeError:
                    continue  # Skip lines that aren't valid JSON
            
            print("'generation' key not found in any line of the file")
            return False
    except Exception as e:
        print(f"Unexpected error when checking output file: {e}")
        return False

def drop_proof_questions(output_file):
    temp_file = output_file + '.temp'
    proof_questions_count = 0
    total_questions = 0
    try:
        with open(output_file, 'r') as input_file, open(temp_file, 'w') as output:
            for line in input_file:
                total_questions += 1
                data = json.loads(line)
                if "generation" in data and "Classification: proof_question" in data["generation"]:
                    proof_questions_count += 1
                else:
                    output.write(line)
        
        # Replace the original file with the filtered file
        os.replace(temp_file, output_file)
        print(f"Proof questions have been dropped from the output file.")
        print(f"Total questions processed: {total_questions}")
        print(f"Proof questions found and dropped: {proof_questions_count}")
        print(f"Remaining questions: {total_questions - proof_questions_count}")
    except Exception as e:
        print(f"Error occurred while dropping proof questions: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)

def main():
    parser = argparse.ArgumentParser(description="Run OpenAI pipeline for question classification and summarization.")
    parser.add_argument("data_file", help="Path to the input data file")
    parser.add_argument("output_file", help="Path to the output file")
    parser.add_argument("--drop_proof", action="store_true", help="Drop proof questions after summarization")
    args = parser.parse_args()

    # Get OpenAI API Key from environment variable
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        return

    # Generate solutions command
    generate_cmd = f"""
    OPENAI_API_KEY={openai_api_key} python3 ../nemo_skills/inference/generate_solutions.py \
        output_file={args.output_file} \
        server.server_type=openai \
        batch_size=1024 \
        inference.tokens_to_generate=256 \
        inference.use_batch_api=true \
        +prompt=openai/fewshot_proof_question_classification \
        ++prompt.few_shot_examples.examples_type=antonov_proof_questions \
        ++data_file={args.data_file} \
        ++server.model=gpt-4o
    """

    print("Running generate solutions command...")
    returncode, stdout, stderr = run_command(generate_cmd)
    if returncode != 0:
        print(f"Error in generate solutions command:\n{stderr}")
        return

    print("Generate solutions command completed successfully.")

    # Get the directory of the output file
    output_dir = os.path.dirname(os.path.abspath(args.output_file))

    # Summarize results command
    summarize_cmd = f"python3 ./summarize_results.py {output_dir}"

    print("Running summarize results command...")
    while True:
        returncode, stdout, stderr = run_command(summarize_cmd)
        if returncode != 0:
            print(f"Error in summarize results command:\n{stderr}")
            break
        
        print("Summarize results output:")
        print(stdout)
        
        if check_output_file(args.output_file):
            print("Summarization completed successfully.")
            break
        
        print("Summarization still in progress. Waiting 10 seconds before next check...")
        time.sleep(10)

    # Drop proof questions if the flag is set
    if args.drop_proof:
        print("Dropping proof questions from the output file...")
        drop_proof_questions(args.output_file)

if __name__ == "__main__":
    main()