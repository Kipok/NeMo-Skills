HUMAN_EVAL_STOP_WORDS = ["\nclass", "\ndef", "\n#", "\nif"]


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


def postprocess_code_fim(generation_dict: dict, truncation_type: str):
    # start and end tokens for FIM generation
    start_token = "<|start_of_middle|>"
    end_token = "<|end_of_middle|>"

    completion = generation_dict['generation']
    suffix = generation_dict['suffix']

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

    if truncation_type == "single-line":
        completion = truncate_num_lines(completion, 1) + "\n"
    elif truncation_type == "multi-line":
        max_num_lines = len(
            generation_dict['canonical_solution'].strip().split("\n"))
        completion = truncate_num_lines(
            completion, max_num_lines) + "\n"
    elif truncation_type == "random-span":
        # print("--"*20)
        # print(completion)
        # print("**"*20)
        # print(suffix)
        completion = truncate_overlapped_characters(completion, suffix)
        # print("==============================")
        # print("After truncation:")
        # print("==============================")
        # print(completion)

    completion = truncate_at_stop_words(completion, HUMAN_EVAL_STOP_WORDS)
    generation_dict["completion"] = completion
    return generation_dict


def truncate_num_lines(infill: str, max_num_lines: int = 1) -> str:
    """Truncates infill to up to max number of lines."""
    infill_lines = stripped_line_split(infill)

    return "\n".join(infill_lines[:max_num_lines])


def stripped_line_split(text):
    return text.strip("\n").split("\n")


def truncate_overlapped_characters(infill, suffix):
    index = infill.find(suffix)
    if index != -1:
        return infill[:index]
    return infill


def truncate_overlapped_lines(infill, suffix):
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


def truncate_at_stop_words(infill, stop_words):
    infill_truncated = infill
    stop_index = None
    for stop_token in stop_words:
        if stop_token in infill_truncated:
            index = infill_truncated.index(stop_token)
            if stop_index is None or index < stop_index:
                stop_index = index
    if stop_index is not None:
        infill_truncated = infill_truncated[:stop_index]
    return infill_truncated


def normalize_string(s):
    # Remove all whitespace characters from the string
    return ''.join(s.split())


def find_prefix(sub, main):
    # Normalize both strings by removing whitespace
    normalized_sub = normalize_string(sub)
    normalized_main = normalize_string(main)

    # Find the starting index of the normalized substring in the normalized main string
    start_index = normalized_main.find(normalized_sub)

    # If the normalized substring is found
    if start_index != -1:
        # Initialize an empty prefix
        prefix = ''
        # Create a pointer to track the length of the normalized prefix
        pointer = 0

        # Iterate through the main string to build the prefix with whitespaces
        for char in main:
            if not char.isspace():
                pointer += 1
            if pointer > start_index:
                break
            prefix += char

        return prefix

    return main
