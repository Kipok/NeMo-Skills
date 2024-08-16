# import argparse
# import json
# import torch
# from tqdm import tqdm
# from tqdm import tqdm
# from pathlib import Path
# from transformers import AutoTokenizer
# from typing import List, Union, Tuple

# def load_tokenizer(tokenizer_dir: str, model_name: str):
#     tokenizer = AutoTokenizer.from_pretrained(
#         tokenizer_dir,
#         legacy=False,
#         padding_side='left',
#         truncation_side='left',
#         trust_remote_code=True,
#     )

#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id
#     pad_id = tokenizer.pad_token_id
#     end_id = tokenizer.eos_token_id

#     return tokenizer, pad_id, end_id

# def tokenize(tokenizer, input_text: Union[str, List[str]]) -> Tuple[List[torch.Tensor], List[List[str]]]:
#     if isinstance(input_text, str):
#         input_text = [input_text]
    
#     batch_input_ids = []
#     batch_token_strings = []
    
#     for text in input_text:
#         tokens = tokenizer.tokenize(text)
#         input_ids = tokenizer.convert_tokens_to_ids(tokens)
#         batch_input_ids.append(torch.tensor(input_ids, dtype=torch.int32).unsqueeze(0))
#         batch_token_strings.append(tokens)
    
#     return batch_input_ids, batch_token_strings

# def decode(tokenizer, token_ids: Union[torch.Tensor, List[int]]) -> str:
#     if torch.is_tensor(token_ids):
#         token_ids = token_ids.tolist()
#     return tokenizer.decode(token_ids)

# def split_and_decode(tokenizer, input_text: str, percentage: float) -> Tuple[str, str]:
#     if not (0 <= percentage <= 1):
#         raise ValueError("Percentage must be between 0 and 1.")
    
#     tokenized, _ = tokenize(tokenizer, input_text)
#     token_ids = tokenized[0][0]
#     split_idx = int(len(token_ids) * percentage)
    
#     first_part_ids = token_ids[:split_idx]
#     second_part_ids = token_ids[split_idx:]
    
#     first_part = decode(tokenizer, first_part_ids)
#     second_part = decode(tokenizer, second_part_ids)
    
#     return first_part, second_part

# import json
# from pathlib import Path

# def process_json(input_file: str, output_file: str, model_path: str, split_percentage: float, readable: bool, solution_key: str):
#     # Load model configuration
#     with open(Path(model_path) / "config.json", 'r') as f:
#         config = json.load(f)
    
#     model_name = config['pretrained_config']['architecture'].lower()
#     name_map = {
#         'GPTForCausalLM'.lower(): 'gpt-next',
#     }
#     model_name = name_map.get(model_name, None)
    
#     # Load tokenizer
#     tokenizer, _, _ = load_tokenizer(model_path, model_name)
    
#     # Count the total number of lines in the input file for tqdm
#     with open(input_file, 'r') as infile:
#         total_lines = sum(1 for _ in infile)
    
#     # Open the input file and process it line by line with tqdm
#     with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
#         for line in tqdm(infile, total=total_lines, desc="Processing Lines"):
#             data = json.loads(line.strip())
            
#             # Get the generation data
#             generations = data.get(solution_key, "")
            
#             # Split and decode the generations
#             first_part, second_part = split_and_decode(tokenizer, generations, split_percentage)
            
#             # Prepare the new fields
#             new_data = {
#                     "first_part": first_part,
#                     "second_part": second_part
#             }
            
#             # Append the new fields to the existing data
#             data.update(new_data)
            
#             # Write the updated data back to the output file
#             outfile.write(json.dumps(data) + '\n')
            
#             # If readable is True, print the parts
#             if readable:
#                 print(f"\nFirst Part ({split_percentage * 100}%): {first_part}")
#                 print(f"Second Part ({(1 - split_percentage) * 100}%): {second_part}")



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Tokenize, split, and decode text from a JSON file")
#     parser.add_argument("--model_path", type=str, default="/mnt/data/llm/models/llama3.1-8b-base-2p", help="Path to the model directory")
#     parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file")
#     parser.add_argument("--output_file", type=str, help="Path to the output JSONL file (default: input_file with '_split_{percentage}' suffix)")
#     parser.add_argument("--split_percentage", type=float, default=0.5, help="Split percentage (between 0 and 1)")
#     parser.add_argument("--readable", action="store_true", help="Prints the results if this flag is set")
#     parser.add_argument("--solution_key", type=str, default="generation", help="Key in the JSON file that contains the text to process")
    
#     args = parser.parse_args()

#     # Determine the output file name if not provided
#     if args.output_file is None:
#         input_file_stem = Path(args.input_file).stem
#         output_file_suffix = f"_split_{int(args.split_percentage * 100)}"
#         args.output_file = str(Path(args.input_file).with_name(input_file_stem + output_file_suffix + ".jsonl"))

#     process_json(
#         input_file=args.input_file,
#         output_file=args.output_file,
#         model_path=args.model_path,
#         split_percentage=args.split_percentage,
#         readable=args.readable,
#         solution_key=args.solution_key
#     )


import argparse
import json
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer
from typing import List, Union, Tuple
from multiprocessing import Pool, cpu_count

def load_tokenizer(tokenizer_dir: str, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        legacy=False,
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id

def tokenize(tokenizer, input_text: Union[str, List[str]]) -> Tuple[List[torch.Tensor], List[List[str]]]:
    if isinstance(input_text, str):
        input_text = [input_text]
    
    batch_input_ids = []
    batch_token_strings = []
    
    for text in input_text:
        tokens = tokenizer.tokenize(text)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        batch_input_ids.append(torch.tensor(input_ids, dtype=torch.int32).unsqueeze(0))
        batch_token_strings.append(tokens)
    
    return batch_input_ids, batch_token_strings

def decode(tokenizer, token_ids: Union[torch.Tensor, List[int]]) -> str:
    if torch.is_tensor(token_ids):
        token_ids = token_ids.tolist()
    return tokenizer.decode(token_ids)

def split_and_decode(tokenizer, input_text: str, percentage: float) -> Tuple[str, str]:
    if not (0 <= percentage <= 1):
        raise ValueError("Percentage must be between 0 and 1.")
    
    tokenized, _ = tokenize(tokenizer, input_text)
    token_ids = tokenized[0][0]
    split_idx = int(len(token_ids) * percentage)
    
    first_part_ids = token_ids[:split_idx]
    second_part_ids = token_ids[split_idx:]
    
    first_part = decode(tokenizer, first_part_ids)
    second_part = decode(tokenizer, second_part_ids)
    
    return first_part, second_part

def process_line(line, tokenizer, split_percentage, solution_key):
    data = json.loads(line.strip())
    generations = data.get(solution_key, "")
    
    first_part, second_part = split_and_decode(tokenizer, generations, split_percentage)
    
    new_data = {
        "first_part": first_part,
        "second_part": second_part
    }
    
    data.update(new_data)
    
    return json.dumps(data)

def process_chunk(chunk, tokenizer, split_percentage, solution_key):
    results = []
    for line in tqdm(chunk, desc="Processing Chunk", unit="line"):
        processed_line = process_line(line, tokenizer, split_percentage, solution_key)
        results.append(processed_line)
    return results

def process_json_parallel(input_file: str, output_file: str, model_path: str, split_percentage: float, readable: bool, solution_key: str):
    with open(Path(model_path) / "config.json", 'r') as f:
        config = json.load(f)
    
    model_name = config['pretrained_config']['architecture'].lower()
    name_map = {
        'GPTForCausalLM'.lower(): 'gpt-next',
    }
    model_name = name_map.get(model_name, None)
    
    tokenizer, _, _ = load_tokenizer(model_path, model_name)
    
    # Read the input file and split it into chunks for parallel processing
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    num_workers = cpu_count()  # Number of processes to use
    chunk_size = len(lines) // num_workers + 1  # Calculate chunk size
    
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    
    with Pool(processes=num_workers) as pool:
        # Add a progress bar for the overall process
        with tqdm(total=len(chunks), desc="Processing Chunks") as pbar:
            results = []
            for chunk_results in pool.starmap(process_chunk, [(chunk, tokenizer, split_percentage, solution_key) for chunk in chunks]):
                results.extend(chunk_results)
                pbar.update(1)
    
    with open(output_file, 'w') as outfile:
        for line in results:
            outfile.write(line + '\n')
    
    if readable:
        for line in results:
            data = json.loads(line)
            print(f"\nFirst Part ({split_percentage * 100}%): {data['first_part']}")
            print(f"Second Part ({(1 - split_percentage) * 100}%): {data['second_part']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize, split, and decode text from a JSON file")
    parser.add_argument("--model_path", type=str, default="/mnt/data/llm/models/llama3.1-8b-base-2p", help="Path to the model directory")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output_file", type=str, help="Path to the output JSONL file (default: input_file with '_split_{percentage}' suffix)")
    parser.add_argument("--split_percentage", type=float, default=0.5, help="Split percentage (between 0 and 1)")
    parser.add_argument("--readable", action="store_true", help="Prints the results if this flag is set")
    parser.add_argument("--solution_key", type=str, default="generation", help="Key in the JSON file that contains the text to process")
    
    args = parser.parse_args()

    if args.output_file is None:
        input_file_stem = Path(args.input_file).stem
        output_file_suffix = f"_split_{int(args.split_percentage * 100)}"
        args.output_file = str(Path(args.input_file).with_name(input_file_stem + output_file_suffix + ".jsonl"))

    process_json_parallel(
        input_file=args.input_file,
        output_file=args.output_file,
        model_path=args.model_path,
        split_percentage=args.split_percentage,
        readable=args.readable,
        solution_key=args.solution_key
    )
