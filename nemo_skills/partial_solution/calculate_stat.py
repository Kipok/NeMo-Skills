import json
import argparse
from collections import defaultdict
from typing import List, Dict
import statistics

def process_files(file_paths: List[str]) -> Dict[str, List[bool]]:
    questions = defaultdict(list)
    
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                question_id = data['id']
                is_correct = data['is_correct']
                questions[question_id].append(is_correct)
    
    return questions

def calculate_completion_stats(questions: Dict[str, List[bool]]) -> Dict:
    stats = {
        'total_questions': len(questions),
        'completion_rates': defaultdict(float)
    }
    
    for solutions in questions.values():
        correct_solutions = sum(solutions)
        total_solutions = len(solutions)
        
        if correct_solutions > 0:
            stats['completion_rates']['at_least_once'] += 1
        if correct_solutions >= total_solutions * 0.25:
            stats['completion_rates']['25%'] += 1
        if correct_solutions >= total_solutions * 0.5:
            stats['completion_rates']['50%'] += 1
        if correct_solutions >= total_solutions * 0.75:
            stats['completion_rates']['75%'] += 1
        if correct_solutions == total_solutions:
            stats['completion_rates']['100%'] += 1
    
    total_questions = stats['total_questions']
    for key in stats['completion_rates']:
        stats['completion_rates'][key] /= total_questions
    
    return stats

def print_stats(stats: Dict, file_name: str = None):
    if file_name:
        print(f"\nStatistics for {file_name}:")
    print(f"Total unique questions: {stats['total_questions']}")
    print("\nCorrect solution completion rates:")
    
    for threshold in ['at_least_once', '25%', '50%', '75%', '100%']:
        print(f"  {threshold}: {stats['completion_rates'][threshold]:.2%}")

def main():
    parser = argparse.ArgumentParser(description="Calculate completion statistics for question-answering outputs.")
    parser.add_argument('files', nargs='+', help='Path to the output-rs files')
    args = parser.parse_args()

    questions = process_files(args.files)
    stats = calculate_completion_stats(questions)
    print_stats(stats, "All Files Combined")

if __name__ == "__main__":
    main()