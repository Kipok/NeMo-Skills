import argparse
import json
import logging
import os
from pathlib import Path

LOG = logging.getLogger(__name__)


def process_batch_results(prediction_jsonl_files):
    for batch_request_file in prediction_jsonl_files:
        jsonl_file = batch_request_file.with_name(batch_request_file.name.replace('.jsonl-batch-request-id', '.jsonl'))

        if batch_request_file.exists():
            try:
                with open(batch_request_file, 'rt', encoding='utf-8') as fin:
                    line = json.load(fin)
                    request_id = line['request_id']
                    generation_key = line['generation_key']

                from nemo_skills.inference.server.model import get_model

                llm = get_model(server_type='openai', model='gpt-4-1106-preview')
                metadata, outputs = llm.get_batch_results(request_id)

                if outputs is None:
                    LOG.warning("Batch generations are not ready yet for %s! Current status: %s", jsonl_file, metadata)
                    continue

                # Read existing data from the jsonl file
                with open(jsonl_file, 'rt', encoding='utf-8') as fin:
                    data = [json.loads(line) for line in fin]

                # Update data with judgements
                for data_point, output in zip(data, outputs):
                    data_point[generation_key] = output['generation']

                # Write updated data back to the jsonl file
                with open(jsonl_file, 'wt', encoding='utf-8') as fout:
                    for data_point in data:
                        fout.write(json.dumps(data_point) + '\n')

                batch_request_file.unlink()
                LOG.info(f"Processed batch results for {jsonl_file}")
            except PermissionError:
                user = os.getenv('USER', 'your_username')
                LOG.error(f"Permission denied when trying to access {jsonl_file} or {batch_request_file}")
                LOG.error(f"Permission denied. Try running the following command to change file ownership:")
                LOG.error(f"sudo chown {user} {jsonl_file} {batch_request_file}")
                LOG.error("Then run this script again.")
            except Exception as e:
                LOG.error(f"An error occurred while processing {batch_request_file}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process batch results from a specified folder.")
    parser.add_argument("results_folder", help="Path to the folder containing batch results")
    args = parser.parse_args()

    # Process batch results
    prediction_files = list(Path(args.results_folder).glob('**/*.jsonl-batch-request-id'))
    process_batch_results(prediction_files)
