def preprocess_code(generation_dict: dict):
    completion = generation_dict['generation']
    completion = completion.strip()
    completion = completion.replace("\r", "")
    if '```' in completion:
        if '```python' in completion:
            def_line = completion.index('```python') + len('```python')
        else:
            def_line = completion.index('```') + len('```')
        completion = completion[def_line:].strip()
        completion = completion.replace('```python', '')
        try:
            next_line = completion.index('```')
            completion = completion[:next_line].strip()
        except:
            print(completion)
            print("================\n")

    if completion.startswith(" "):
        completion = completion.strip()

    generation_dict["completion"] = completion
    return generation_dict


def postprocess_code_fim(generation_dict: dict):
    # start and end tokens for FIM generation
    start_token = "<|start_of_middle|>"
    end_token = "<|end_of_middle|>"

    completion = generation_dict['generation']
    # completion = completion.strip()
    # completion = completion.replace("\r", "")
    if '```' in completion:
        if '```python' in completion:
            def_line = completion.index('```python') + len('```python')
        else:
            def_line = completion.index('```') + len('```')
        completion = completion[def_line:].strip()
        completion = completion.replace('```python', '')
        try:
            next_line = completion.index('```')
            completion = completion[:next_line].strip()
        except:
            print(completion)
            print("================\n")

    start_index = completion.find(start_token)
    end_index = completion.find(end_token)
    if start_index != -1 and end_index != -1:
        start_index += len(start_token)
        completion = completion[start_index:end_index]

    generation_dict["completion"] = completion
    return generation_dict
