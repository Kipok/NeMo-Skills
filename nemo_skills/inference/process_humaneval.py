"""
Adapted from https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder

Call this script as

python process_humaneval_nemo.py --path <path_to_dir_with_files> --out_path <path to output>.jsonl --add_prompt --ast_check

# If you want to log incorrect code (that cannot be parsed by python AST), use the --ast_check flag

#############################################################################################################

# HF Model:

python process_humaneval_nemo.py \
    --path "results/codellama_7b_instruct/" \
    --out_path "./results/codellama_7b_instruct/results.jsonl" \
    --add_prompt --ast_check


# NeMo model:
python process_humaneval_nemo.py \
    --path "results/nemo_codellama_1m_data_humaneval/" \
    --out_path "./results/nemo_codellama_1m_data_humaneval/results.jsonl" \
    --add_prompt --ast_check

# Evaluate the results in a container for safety
# If you want only base eval results, add the `--base-only` flag
# You can add `--i-just-wanna-run` to force recompute the results

docker run -v $(pwd):/app ganler/evalplus:latest --dataset humaneval --samples \
    results/nemo_codellama_1m_data_humaneval_plus/results.jsonl

OR RUN IT WITH SYSTEM RISK
evalplus.evaluate --dataset humaneval --samples samples.jsonl \
    --i-just-wanna-run

"""

import argparse
import ast

# from human_eval.data import read_problems, write_jsonl, stream_jsonl
import glob
import os

from evalplus.data import get_human_eval_plus, write_jsonl
from evalplus.data.utils import stream_jsonl
from tqdm import tqdm

try:
    from evalplus_checker import sanitize
except:
    pass

parser = argparse.ArgumentParser()

# Inputs
parser.add_argument('--path', type=str, help="")
parser.add_argument('--out_path', type=str, help="")
parser.add_argument('--add_prompt', action='store_true', help='')
parser.add_argument('--ast_check', action='store_true', help='')
parser.add_argument('--is-completion', action='store_true', help='')

parser.add_argument("--rm-prefix-lines", type=str, help="Remove lines starting with this")
parser.add_argument("--eofs", nargs="+", type=str, default=[])

args = parser.parse_args()

files = sorted(glob.glob(args.path + '/*.jsonl'))
# Filter out the file with the output path
files = [f for f in files if f != args.out_path]
print("{} files in {}".format(len(files), args.path))

problems = get_human_eval_plus()


output = []
a = 0
ast_failures = 0
for code_file in tqdm(files, total=len(files)):
    codes = [c for c in stream_jsonl(code_file)]
    if args.add_prompt:
        for code in codes:
            task_id = code['task_id']
            prompt = problems[task_id]['prompt']
            entry_point = problems[task_id]['entry_point']

            completion = code['completion']
            completion = completion.strip()
            completion = completion.replace("\r", "")

            old_code = completion

            # new_code = sanitize(
            #     old_code=old_code,
            #     entry_point=entry_point,
            #     rm_prefix_lines=args.rm_prefix_lines,
            #     eofs=args.eofs,
            # ).strip()
            #
            # # if changed, print the message
            # if new_code != old_code:
            #     msg = "Sanitized: " + task_id
            #     print(msg)
            #     print(new_code)
            #     print()

            if '```' in completion:

                # Is complete code generation
                if not args.is_completion:
                    if '```python' in completion:
                        def_line = completion.index('```python') + len('```python')
                    else:
                        def_line = completion.index('```') + len('```')
                    completion = completion[def_line:].strip()
                    completion = completion.replace('```python', '')
                    # print(completion)
                    try:
                        next_line = completion.index('```')
                        completion = completion[:next_line].strip()
                    except:
                        a += 1
                        print(completion)
                        print("================\n")

                else:
                    # Is templated completion generation
                    try:
                        end_line = completion.index('```')
                        completion = completion[:end_line].strip()
                    except:
                        a += 1
                        print(completion)
                        print("================\n")

                # print(completion)

            # If the code starts with `, remove all preceding ` before the def
            # if '`' in completion:
            #     completion = completion.replace("`", "")  # remove all preceding ` before the def

            if "__name__ == \"__main__\"" in completion:
                next_line = completion.index('if __name__ == "__main__":')
                completion = completion[:next_line].strip()
                # print(completion)

            if "# Example usage" in completion:
                # print(completion)
                next_line = completion.index('# Example usage')
                completion = completion[:next_line].strip()

            if "[/CODE]" in completion:
                next_line = completion.index('[/CODE]')
                completion = completion[:next_line].strip()

            if "[SOL]" in completion:
                next_line = completion.index('[SOL]')
                completion = completion[:next_line].strip()

                if "[/SOL]" in completion:
                    next_line = completion.index('[/SOL]')
                    completion = completion[:next_line].strip()

            if "[/INST]" in completion:
                next_line = completion.index('[/INST]')
                completion = completion[:next_line].strip()

            if "Explanation:" in completion:
                next_line = completion.index('Explanation:')
                completion = completion[:next_line].strip()

            if "<extra_id_1>" in completion:
                next_line = completion.index('<extra_id_1>')
                completion = completion[:next_line].strip()

            if "here" in completion.lower() and "code" in completion.lower() and "def" in completion:
                # Find first def after "Here is the" and remove everything before that
                next_line = completion.index('def')
                completion = completion[next_line:].strip()

            if completion.startswith(" "):
                completion = completion.strip()

            # Try parsing the code via ast to make sure it is valid
            if args.ast_check:
                try:
                    ast.parse(completion)
                    passed_ast = True
                except:
                    passed_ast = False
                    ast_failures += 1
                    print("AST check failed. Invalid code:\n{}".format(completion))
                    print("*" * 50)
                    print()

            code['completion'] = completion

            # For evalplus checks only
            # code.pop("completion", None)
            # code['solution'] = new_code

            # For ast checks only
            if args.ast_check:
                code['ast_successful'] = passed_ast

    output += codes

print("save to {}".format(args.out_path))

# Write the results to file
write_jsonl(args.out_path, output)

print("Number of ``` parsing failures:", a)
if args.ast_check:
    print("Number of ast failures:", ast_failures)
