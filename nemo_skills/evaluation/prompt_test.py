import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    CodeLlamaTokenizer,
)

model_name = "meta-llama/CodeLlama-7b-hf"
load_in_8bit = False
device_map = "auto"
max_gen_len = 128
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=load_in_8bit,
    device_map=device_map,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
suffix_first = False
tokenizer = CodeLlamaTokenizer.from_pretrained(
    model_name, trust_remote_code=True, suffix_first=suffix_first)


def generate_one_completion(pre, suf):
    # prompt = pre+"<FILL_ME>"+suf
    prompt = "▁<PRE>" + pre + "▁<SUF>" + suf + "▁<MID>"
    input_ids = tokenizer(prompt, suffix_first=suffix_first, return_tensors="pt")[
        "input_ids"].to('cuda')
    print(tokenizer.batch_decode(input_ids, skip_special_tokens=False)[0])
    generation_tokens = model.generate(
        input_ids,
        max_new_tokens=max_gen_len,
        temperature=0.0
    )
    print(tokenizer.batch_decode(
        generation_tokens, skip_special_tokens=False)[0])
    outputs = tokenizer.batch_decode(
        generation_tokens[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    return outputs


# example = {"task_id": "RandomSpanInfillingLight/HumanEval/3/1", "entry_point": "below_zero", "prompt": "\nfrom typing import List\n\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"\n    balance = 0\n\n    for op in o", "suffix": "\n\n\treturn False\n", "canonical_solution": "perations:\n        balance += op\n        if balance < 0:\n            return True",
    #    "test": "\n\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([]) == False\n    assert candidate([1, 2, -3, 1, 2, -3]) == False\n    assert candidate([1, 2, -4, 5, 6]) == True\n    assert candidate([1, -1, 2, -2, 5, -5, 4, -4]) == False\n    assert candidate([1, -1, 2, -2, 5, -5, 4, -5]) == True\n    assert candidate([1, -2, 2, -2, 5, -5, 4, -4]) == True\n", "question": "\nfrom typing import List\n\n\ndef below_zero(operations: List[int]) -> bool:\n\t\"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n\tzero balance. Your task is to detect if at any point the balance of account fallls below zero, and\n\tat that point function should return True. Otherwise it should return False.\n\t>>> below_zero([1, 2, 3])\n\tFalse\n\t>>> below_zero([1, 2, -4, 5])\n\tTrue\n\t\"\"\"\n\tbalance = 0\n\n\tfor op in o"}

# example = {"task_id": "RandomSpanInfilling/HumanEval/0/2", "entry_point": "has_close_elements", "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n    for idx, e", "suffix": "\t\tif idx != idx2:\n\t\t\t\tdistance = abs(elem - elem2)\n\t\t\t\tif distance < threshold:\n\t\t\t\t\treturn True\n\n\treturn False\n", "canonical_solution": "lem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n    ",
    #    "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n", "question": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n\t\"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n\tgiven threshold.\n\t>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n\tFalse\n\t>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n\tTrue\n\t\"\"\"\n\tfor idx, e"}

example = {
    "task_id": "MultiLineInfilling/HumanEval/0/L0_L7",
    "entry_point": "has_close_elements",
    "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n\n",
    "suffix": "",
    "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
    "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n", "question": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n\t\"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n\tgiven threshold.\n\t>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n\tFalse\n\t>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n\tTrue\n\t\"\"\"\n\n"
}

prompt = example["prompt"].replace('    ', '\t')
suffix = example["suffix"].replace('    ', '\t')

completion = generate_one_completion(prompt, suffix)
print("**"*20)
print(completion)
print("**"*20)
full_code = prompt + completion + suffix
print(full_code)
