# Dataset Explorer Demo

1. Download data TBD
2. Retrieve similar questions. Do it for all benchmarks you want to compare against.
   Assuming you're running from this folder.

   ```
   python -m nemo_skills.inference.retrieve_similar \
       ++retrieve_from=./data.jsonl \
       ++compare_to="./nemo_skills/dataset/<benchmark>/test.jsonl" \
       ++output_file=./similar-retrieved/<benchmark>.jsonl \
       ++top_k=5
   ```

3. Start the Gradio demo.

   ```
   python visualize_similar.py
   ```