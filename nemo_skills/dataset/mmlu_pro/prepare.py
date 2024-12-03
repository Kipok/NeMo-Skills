import argparse
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def format_entry(entry, type):
    return {
        "question": entry['question'],
        "options": "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(entry['options'])),
        "expected_answer": entry['answer'],
        "examples_type": f'mmlu_pro_few_shot_{type}_{entry["category"]}',
        "subset_for_metrics": entry['category'],
    }


def write_data_to_file(output_file, data, type):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {output_file.name}"):
            json.dump(format_entry(entry, type), fout)
            fout.write("\n")


def main(args):
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")[args.split]
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    for type in ['llama', 'tigerlab']:
        output_file = data_dir / f"{args.split}_{type}.jsonl"
        write_data_to_file(output_file, dataset, type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("validation", "test"), help="Dataset split to process.")
    args = parser.parse_args()
    main(args)
