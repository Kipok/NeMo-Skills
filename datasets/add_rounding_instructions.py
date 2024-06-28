import json
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="./rounded.jsonl")

    args = parser.parse_args()

    with open(args.path, 'r') as fin, open(args.save_path, 'w') as fout:
        for line in fin:
            data = json.loads(line)
            try:
                float(data['expected_answer'])
                number_of_values = 0
                if '.' in str(data['expected_answer']):
                    number_of_values = len(str(data['expected_answer']).split('.')[1])
                if number_of_values == 0:
                    data['question'] += ' Express the answer as an integer.'
                elif number_of_values == 1:
                    data['question'] += ' Round the answer to one decimal place.'
                else:
                    data['question'] += f' Round the answer to {number_of_values} decimal places.'
            except ValueError:
                pass
            fout.write(json.dumps(data) + '\n')
