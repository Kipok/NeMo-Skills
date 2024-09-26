# Generation

Make sure to complete [prerequisites](/docs/prerequisites.md) before proceeding.

[nemo_skills/pipeline/generate.py](/nemo_skills/pipeline/generate.py) can be used for large-scale data generation
using LLMs. You provide an input jsonl file as well as the prompt config/template and we run LLM for each line
of the input using the dictionary there to format the prompt. You input file keys need to match the prompt config
but otherwise there is no restrictions on what data you use for input. See [prompt format](/docs/prompt-format.md)
for more details on how to create new prompts.

Here are a few typical use-cases of the generation pipeline.

## Greedy inference

