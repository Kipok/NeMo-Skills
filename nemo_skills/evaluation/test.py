x = {"generation": "\tfor i in range(len(numbers) - 1):\n\t\tfor j in range(i + 1, len(numbers)):\n\t\t\tif abs(numbers[i] - numbers[j]) < threshold:\n\t\t\t\treturn True\n\treturn False\n\n\ndef has_close_elements_2(numbers: List[float], threshold: float) -> bool:\n\t\"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n\tgiven threshold.\n\t>>> has_close_elements([1.0, ", "error_message": "<not_executed>", "task_id": "MultiLineInfilling/HumanEval/0/L0_L7", "entry_point": "has_close_elements", "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n\n",
     "suffix": "",
     "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n", "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n",
     "question": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n\t\"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n\tgiven threshold.\n\t>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n\tFalse\n\t>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n\tTrue\n\t\"\"\"\n\n",
     "input_prompt": "<s>\u2581<PRE>from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n\t\"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n\tgiven threshold.\n\t>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n\tFalse\n\t>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n\tTrue\n\t\"\"\"\n\n\u2581<SUF>\u2581<MID>", "completion": "\tfor i in range(len(numbers) - 1):\n\t\tfor j in range(i + 1, len(numbers)):\n\t\t\tif abs(numbers[i] - numbers[j]) < threshold:\n\t\t\t\treturn True\n\treturn False\n\n\ndef has_close_elements_2(numbers: List[float], threshold: float) -> bool:\n\t\"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n\tgiven threshold.\n\t>>> has_close_elements([1.0, ", "result": "failed: EOF while scanning triple-quoted string literal (<string>, line 42)", "passed": False}

# print(x["question"])
# print("**"*20)
# print(x["input_prompt"])
# print("**"*20)
# print(x["generation"])

# print(repr(x["generation"]))
# print("**"*20)
# print(repr(x["suffix"]))
# print("**"*20)
# print(repr(x["completion"]))
# print(x["completion"])
# print("**"*20)
# print(x["canonical_solution"])
# print("**"*20)
full_code = x["question"] + x["completion"] + x["suffix"] + \
    "\n" + x["test"] + "\n" + f"check({x['entry_point']})"
# print(full_code)


def truncate_overlap(infill, suffix):
    infill_lines = infill.split("\n")
    suffix_lines = [l.strip() for l in suffix.split("\n")]
    num_suffix_lines = len(suffix_lines)
    if num_suffix_lines > 1:
        for idx in range(len(infill_lines)):
            infill_span_lines = [l.strip()
                                 for l in infill_lines[idx:idx+num_suffix_lines]]
            if infill_span_lines == suffix_lines:
                return "\n".join(infill_lines[:idx])
    return infill


infill = """
        for idx, elem in enumerate(numbers):
                if idx == 0:
                        continue
                else:
                        for idx2, elem2 in enumerate(numbers):
                                if idx != idx2:
                                        distance = abs(elem - elem2)
                                        if distance < threshold:
                                                return True

        return False


def has_close_elements(numbers: List[float], threshold: float) -> bool:
        \"\"\" Check if in given list of
"""
infill = infill.lstrip("\n")

suffix = """
                for idx2, elem2 in enumerate(numbers):
                        if idx != idx2:
                                distance = abs(elem - elem2)
                                if distance < threshold:
                                        return True

        return False

"""
suffix = suffix.strip()
infill = truncate_overlap(infill, suffix)
print(infill)
print("----------------------------------------------------------------")
print(suffix)
print("----------------------------------------------------------------")
print(infill)
