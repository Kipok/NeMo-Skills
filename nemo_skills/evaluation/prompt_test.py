import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

prompt = '''[INST] <<SYS>>
You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
<</SYS>>

Here is the given code where you need to fill in the middle by replacing [FILL_ME] with your code.
```python
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    for idx, elem in enumerate(numbers):
            for idx2, elem2 in enumerate(numbers):
                    if [FILL_ME]
    return False
```

[/INST]
'''

input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
output = model.generate(
    input_ids,
    max_new_tokens=200,
)
output = output[0].to("cpu")

filling = tokenizer.decode(output, skip_special_tokens=False)
# print(prompt)
# print("--------------------------------")
# print(filling)


prompt = '''<s> <PRE> from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
        """ Check if in given list of numbers, are any two numbers closer to each other than
        given threshold.
        >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
        False
        >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
        True
        """
        for idx, elem in enumerate(numbers):
                for idx2, elem2 in enumerate(numbers): <SUF>
                                if distance < threshold:
                                        return True

        return False

<MID>
'''

prompt = '''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
        """ Check if in given list of numbers, are any two numbers closer to each other than
        given threshold.
        >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
        False
        >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
        True
        """
        for idx, elem in enumerate(numbers):
                for idx2, elem2 in enumerate(numbers):<FILL_ME>
                                if distance < threshold:
                                        return True

        return False

'''

model_id = "codellama/CodeLlama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
output = model.generate(
    input_ids,
    max_new_tokens=200,
)
output = output[0].to("cpu")

filling = tokenizer.decode(output, skip_special_tokens=False)
print(prompt)
print("--------------------------------")
print(filling)
print(repr(filling))
