import json
import os
import argparse

def extract_mismatch(file_name, answer_file_path=None):
    if answer_file_path is None:
        answer_file_path = os.path.dirname(file_name)

    false_negatives = []
    false_positives = []
    true_positives = []
    true_negatives = []
    unclassified = []

    total_records = 0
    total_judgement_no_mistake = 0
    total_is_correct_true = 0
    total_unclassified = 0

    with open(file_name, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            total_records += 1
            
            generation = data.get("generation", "").lower()
            is_correct = data.get("is_correct", None)
            
            if "judgment: no mistake" in generation:
                total_judgement_no_mistake += 1

            if is_correct == True:
                total_is_correct_true += 1

            if "judgment: no mistake" in generation and is_correct == False:
                false_positives.append(data)
            elif "judgment: no mistake" in generation and is_correct == True:
                true_positives.append(data)
            elif "judgment: wrong answer" in generation and is_correct == False:
                true_negatives.append(data)
            elif "judgment: wrong answer" in generation and is_correct == True:
                false_negatives.append(data)
            else:
                total_unclassified += 1
                unclassified.append(data)

    base_name = os.path.splitext(os.path.basename(file_name))[0]
    save_to_jsonl(false_negatives, os.path.join(answer_file_path, f"{base_name}_false_negatives.jsonl"))
    save_to_jsonl(false_positives, os.path.join(answer_file_path, f"{base_name}_false_positives.jsonl"))
    save_to_jsonl(true_positives, os.path.join(answer_file_path, f"{base_name}_true_positives.jsonl"))
    save_to_jsonl(true_negatives, os.path.join(answer_file_path, f"{base_name}_true_negatives.jsonl"))
    save_to_jsonl(unclassified, os.path.join(answer_file_path, f"{base_name}_unclassified.jsonl"))

    print_statistics(total_records, total_judgement_no_mistake, total_is_correct_true, total_unclassified, false_negatives, false_positives, true_positives, true_negatives)

def save_to_jsonl(data_list, file_path):
    with open(file_path, 'w') as f:
        for item in data_list:
            f.write(json.dumps(item) + "\n")

def print_statistics(total_records, total_judgement_no_mistake, total_is_correct_true, total_unclassified, false_negatives, false_positives, true_positives, true_negatives):
    tp = len(true_positives)
    tn = len(true_negatives)
    fp = len(false_positives)
    fn = len(false_negatives)

    accuracy = (tp + tn) / total_records if total_records > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision * recall) if (precision + recall) > 0 else 0

    print(f"Total records: {total_records}")
    print(f"Total 'No mistake' in judgement: {total_judgement_no_mistake} ({total_judgement_no_mistake / total_records * 100:.2f}%)")
    print(f"Total 'true' in is_correct: {total_is_correct_true} ({total_is_correct_true / total_records * 100:.2f}%)")
    print(f"Total unclassified: {total_unclassified} ({total_unclassified / total_records * 100:.2f}%)")

    print(f"True Positives (TP): {tp} ({tp / total_records * 100:.2f}%)")
    print(f"True Negatives (TN): {tn} ({tn / total_records * 100:.2f}%)")
    print(f"False Positives (FP): {fp} ({fp / total_records * 100:.2f}%)")
    print(f"False Negatives (FN): {fn} ({fn / total_records * 100:.2f}%)")

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1_score * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract mismatches from JSONL file")
    parser.add_argument('initial_book', type=str, help="Path to the initial book JSONL file")
    parser.add_argument('data_dir', type=str, nargs='?', default=None, help="Directory to save the generated subsets (default: same directory as initial_book)")

    args = parser.parse_args()
    extract_mismatch(args.initial_book, args.data_dir)
