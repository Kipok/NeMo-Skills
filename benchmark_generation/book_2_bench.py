import json
import argparse
import os
from collections import defaultdict

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} problems from {file_path}")
    return data

def group_problems_by_chapter(data):
    problems_by_chapter = defaultdict(list)
    for problem in data:
        chapter = problem['chapter']
        problems_by_chapter[chapter].append(problem)
    print(f"Grouped problems into {len(problems_by_chapter)} chapters")
    return problems_by_chapter

def filter_problems_by_answer(problems):
    filtered = []
    for problem in problems:
        if problem.get('expected_answer') != "":
            filtered.append(problem)
        else:
            print(f"Filtered out problem: {problem}")
    print(f"Filtered problems: {len(filtered)} of {len(problems)} remain after filtering")
    return filtered

def create_subset_by_percentage(problems_by_chapter, percentage, subset_type):
    subset = []
    for chapter, problems in problems_by_chapter.items():
        filtered_problems = filter_problems_by_answer(problems)
        n = max(1, len(filtered_problems) * percentage // 100)
        print(f"Chapter {chapter}: using {n} problems for {subset_type} subset")

        if subset_type == 'easy':
            subset.extend(filtered_problems[:n])
        elif subset_type == 'hard':
            subset.extend(filtered_problems[-n:])
        elif subset_type == 'balanced':
            chapter_size = len(filtered_problems)
            if chapter_size <= n:
                subset.extend(filtered_problems)
            else:
                indices = [i * (chapter_size // n) for i in range(n)]
                subset.extend(filtered_problems[i] for i in indices)
    print(f"Created {subset_type} subset with {len(subset)} problems")
    return subset

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Saved {len(data)} problems to {file_path}")

def main(args):
    data = load_data(args.initial_book)
    problems_by_chapter = group_problems_by_chapter(data)

    easy_subset = create_subset_by_percentage(problems_by_chapter, args.p, 'easy')
    hard_subset = create_subset_by_percentage(problems_by_chapter, args.p, 'hard')
    balanced_subset = create_subset_by_percentage(problems_by_chapter, args.p, 'balanced')

    if not args.data_dir:
        args.data_dir = os.path.dirname(os.path.abspath(args.initial_book))

    print(f"Initial book directory: {os.path.dirname(os.path.abspath(args.initial_book))}")
    print(f"Data directory: {args.data_dir}")

    os.makedirs(args.data_dir, exist_ok=True)

    # Print the content of the subsets
    print("Easy subset content:")
    for problem in easy_subset:
        print(problem)
    print("\nHard subset content:")
    for problem in hard_subset:
        print(problem)
    print("\nBalanced subset content:")
    for problem in balanced_subset:
        print(problem)

    save_json(easy_subset, os.path.join(args.data_dir, 'easy_subset.json'))
    save_json(hard_subset, os.path.join(args.data_dir, 'hard_subset.json'))
    save_json(balanced_subset, os.path.join(args.data_dir, 'balanced_subset.json'))

    print("Subsets created successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate subsets of book tasks")
    parser.add_argument('initial_book', type=str, help="Path to the initial book JSON file")
    parser.add_argument('data_dir', type=str, nargs='?', help="Directory to save the generated subsets (default: same directory as initial_book)")
    parser.add_argument('-p', type=int, default=20, help="Percentage of exercises per subset (default: 20%)")

    args = parser.parse_args()

    if not args.data_dir:
        args.data_dir = os.path.dirname(os.path.abspath(args.initial_book))

    main(args)
