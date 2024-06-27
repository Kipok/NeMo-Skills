import argparse
import json
import os
import pprint
from typing import List, Union

import requests
from evalplus.data import get_human_eval_plus, write_jsonl
from tqdm import tqdm


def make_request(
    sentences: Union[str, List[str]],
    tokens_to_generate=512,
    temperature=1.0,
    add_BOS=False,  # True for llama and mistral/mixtral base models
    top_k=0,
    top_p=0.95,
    greedy=False,
    all_probs=False,
    repetition_penalty=1.0,
    min_tokens_to_generate=2,
    end_strings=None,
    host='localhost',
    port=5555,
):

    if not isinstance(sentences, (list, tuple)):
        sentences = [sentences]

    if end_strings is None:
        end_strings = ['<|endoftext|>']

    if not isinstance(end_strings, list):
        end_strings = [end_strings]

    data = {
        "sentences": sentences,
        "tokens_to_generate": tokens_to_generate,
        "temperature": temperature,
        "add_BOS": add_BOS,
        "top_k": top_k,
        "top_p": top_p,
        "greedy": greedy,
        "all_probs": all_probs,
        "repetition_penalty": repetition_penalty,
        "min_tokens_to_generate": min_tokens_to_generate,
        'end_strings': end_strings,
    }
    return generate_nemo(data, host=host, port=port)


def generate_nemo(data, host='localhost', port=5555):
    headers = {"Content-Type": "application/json"}
    resp = requests.put(
        'http://{host}:{port}/generate'.format(host=host, port=port), data=json.dumps(data), headers=headers
    )
    sentences = resp.json()
    return sentences


def generate_prompt(input, template=None):

    if template == "[wizardcoder]":

        INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem. The code must be enclosed in ```python``` code block.
{input}

### Response:"""

    elif template == "[llama3-instruct]":
        INSTRUCTION = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

### Instruction:
You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
Here is the given code to do completion:
```python
{input}
```
Please continue to complete the function with python programming language. You are not allowed to modify the given code and do the completion only.

Please return all completed codes in one code block.
This code block should be in the following format:
```python
# Your codes here
```

@@ Response<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    elif template == "[llama3-evalplus]":
        INSTRUCTION = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Please provide a self-contained Python script that solves the following problem in a markdown code block:
```
{input.strip()}
```<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Below is a self-contained Python script that solves the problem:
```python
"""

    elif template == "[codellama-instruct]":
        INSTRUCTION = f"""[INST] Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a Python script for this problem.
{input}

### Response: [/INST]"""

    elif template == "[mistral-instruct]" or template == "[mixtral-instruct]":
        INSTRUCTION = f"""[INST] Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a Python script for this problem:
{input} [/INST]

### Response:"""

    elif template == "[mistral-og]" or template == "[codellama-og]" or template == "[mixtral-og]":
        INSTRUCTION = f"[INST] {input} [/INST]"

    elif template == "[nemo-wizardcoder]":
        INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{input}

### Response:"""

    elif template == "[nemo-wizardcoder-completion]":
        input_strip = input.strip()
        INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Complete the following code segment:

{input_strip}

### Response:"""

    elif template == "[nemo-chat]":
        INSTRUCTION = f"""<extra_id_0>System\n\n<extra_id_1>User\n{input}\n<extra_id_1>Assistant\n"""

    elif template == "[nemo-code-interpreter]":
        INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Here is the given code to do completion:
```python
{input}
```
Please continue to complete the function with python programming language. You are not allowed to modify the
given code and do the completion only.
Please return all completed codes in one code block.
This code block should be in the following format:
```python
# Your codes here
```

### Response:"""

    elif template == "[code-interpreter]":
        INSTRUCTION = f"""<｜begin▁of▁sentence｜>You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer
### Instruction:
You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
Here is the given code to do completion:
```python
{input}
```
Please continue to complete the function with python programming language. You are not allowed to modify the given code and do the completion only.

Please return all completed codes in one code block.
This code block should be in the following format:
```python
# Your codes here
```

@@ Response
"""

    else:
        if template is None:
            raise RuntimeError("Template string must be specified.")

        if template is not None and "{input}" not in template:
            raise RuntimeError("Template string must contain {input}.")

        INSTRUCTION = template.format(input=input)

    return INSTRUCTION


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_path', type=str, help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--temperature', type=float, default=0.8, help="")
    parser.add_argument('--top_p', type=float, default=0.95, help="")
    parser.add_argument('--N', type=int, default=200, help="")
    parser.add_argument('--max_len', type=int, default=1280, help="")
    parser.add_argument('--add_BOS', action='store_true', help='')
    parser.add_argument('--decoding_style', type=str, default='sampling', help="")
    parser.add_argument('--greedy_decode', action='store_true', help='')
    parser.add_argument('--batchsize', type=int, default=10, help='')
    parser.add_argument('--overwrite', action='store_true', help='')

    # Model specification
    parser.add_argument('--template', type=str, default='[llama3-instruct]', help='')
    parser.add_argument('--nemo_host', type=str, default='127.0.0.1', help='')
    parser.add_argument('--nemo_port', type=int, default=5555, help='')

    args = parser.parse_args()

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    problems = get_human_eval_plus()

    task_ids = sorted(problems.keys())[args.start_index : args.end_index]
    prompts = [problems[task_id]['prompt'] for task_id in task_ids]
    num_samples = len(prompts)
    print("Number of samples: {}".format(num_samples))

    # Call nemo api
    generation_config = {
        "tokens_to_generate": args.max_len,
        "temperature": args.temperature,
        "add_BOS": args.add_BOS,
        "top_k": 1 if args.greedy_decode else 50,
        "top_p": args.top_p,
        "greedy": True if args.greedy_decode else False,
        "all_probs": False,
        "repetition_penalty": 1.0,
        "min_tokens_to_generate": 2,
        'end_strings': ['<|endoftext|>', '<extra_id_1>', '</s>', '<|eot_id|>'],
    }

    if generation_config['greedy']:
        print("Greedy generation set - setting temperature, top_p and top_k to 1.0")
        generation_config['temperature'] = 1.0
        generation_config['top_p'] = 1.0
        generation_config['top_k'] = 1

    basedir = os.path.expanduser(args.output_path)
    if not os.path.exists(basedir):
        os.makedirs(basedir, exist_ok=True)

    # Greedy batch generation
    if args.greedy_decode:
        print("Generating completions for greedy decoding...")
        prog = tqdm(range(num_samples), ncols=0, total=num_samples)
        for i in range(0, num_samples, args.batchsize):
            output_files = []
            for j in range(args.batchsize):
                if i + j >= args.end_index:
                    break

                output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i + j)

                if os.path.exists(output_file) and not args.overwrite:
                    print(f'Skip {output_file} as it already exists')
                    continue

                output_files.append(output_file)

            if len(output_files) == 0:
                continue

            print("Using template :", args.template)
            prompt_subbatch = []
            ids_subbatch = []
            for j in range(args.batchsize):
                if i + j < args.end_index:
                    prompt = prompts[i + j].replace('    ', '\t')
                    prompt_subbatch.append(generate_prompt(prompt, template=args.template))

                    idx = task_ids[i + j]
                    ids_subbatch.append(idx)

            # nemo api call
            print(prompt_subbatch, flush=True)
            response = make_request(
                sentences=prompt_subbatch, host=args.nemo_host, port=args.nemo_port, **generation_config
            )
            gen_seqs = response['sentences']

            if gen_seqs is not None:
                for k, gen_seq in enumerate(gen_seqs):
                    idx = ids_subbatch[k]
                    output_file = output_files[k]

                    completion_seqs = []
                    completion_seq = gen_seq[len(prompt_subbatch[k]) :]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')

                    completion_seqs.append(
                        {
                            'task_id': idx,
                            'completion': completion_seq,
                            'all_code': all_code,
                        }
                    )

                    print("Saving results to {}".format(output_file))
                    write_jsonl(output_file, completion_seqs)

                print()
                prog.update(len(gen_seqs))

        print("Finished!")
        return

    # Loop over all samples for pass@K generation
    print(f"Generating completions for pass@{args.N}...")
    prog = tqdm(range(num_samples), ncols=0, total=num_samples)
    for i in range(0, num_samples, args.batchsize):
        output_files = []
        for j in range(args.batchsize):
            if i + j >= args.end_index:
                break

            output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i + j)

            if os.path.exists(output_file) and not args.overwrite:
                print(f'Skip {output_file} as it already exists')
                continue

            output_files.append(output_file)

        if len(output_files) == 0:
            continue

        print("Using template :", args.template)
        prompt_subbatch = []
        ids_subbatch = []
        for j in range(args.batchsize):
            if i + j < args.end_index:
                prompt = prompts[i + j].replace('    ', '\t')
                prompt_subbatch.append(generate_prompt(prompt, template=args.template))

                idx = task_ids[i + j]
                ids_subbatch.append(idx)

        # vllm api call
        prompt_subbatch_expanded = []
        if args.N > 1:
            for prompt_ in prompt_subbatch:
                prompt_subbatch_expanded.extend([prompt_] * args.N)
        else:
            prompt_subbatch_expanded = prompt_subbatch

        # nemo api call
        response = make_request(
            sentences=prompt_subbatch_expanded, host=args.nemo_host, port=args.nemo_port, **generation_config
        )
        gen_seqs = response['sentences']

        if gen_seqs is not None:
            for k in range(len(prompt_subbatch)):
                idx = ids_subbatch[k]
                output_file = output_files[k]

                completion_seqs = []
                for k_gen in range(args.N):
                    gen = gen_seqs[k * args.N + k_gen]
                    completion_seq = gen[len(prompt_subbatch[k]) :]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen.replace('\t', '    ')

                    completion_seqs.append(
                        {
                            'task_id': idx,
                            'completion': completion_seq,
                            'all_code': all_code,
                        }
                    )

                print("Saving results to {}".format(output_file))
                write_jsonl(output_file, completion_seqs)

            print()
            prog.update(len(prompt_subbatch))


if __name__ == '__main__':
    main()
