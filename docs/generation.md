# Generation

Make sure to complete [prerequisites](/docs/prerequisites.md) before proceeding.

[nemo_skills/pipeline/generate.py](/nemo_skills/pipeline/generate.py) can be used for large-scale data generation
using LLMs. You provide an input jsonl file as well as the prompt config/template and we run LLM for each line
of the input using the dictionary there to format the prompt. You input file keys need to match the prompt config
but otherwise there is no restrictions on what data you use for input. See [prompt format](/docs/prompt-format.md)
for more details on how to create new prompts.

Check out [common parameters documentation](/docs/common-parameters.md) to learn about some of the
common parameters accepted by all generation scripts.

Here are a few typical use-cases of the generation pipeline.

## Greedy inference

Let's say you just want to generate greedy predictions for some data. Here is how you do it.

1. Create your data file. E.g. let's say you have the following in `/workspace/input.jsonl` (the `/workspace` needs
   to be mounted inside of your [cluster config](/docs/prerequisites.md#general-information))

   ```jsonl
   {"prompt": "How are you doing?", "option_a": "Great", "option_b": "Bad"}
   {"prompt": "What's the weather like today?", "option_a": "Perfect", "option_b": "Aweful"}
   {"prompt": "How do you feel?", "option_a": "Crazy", "option_b": "Nice"}
   ```

2. Create your [prompt config](/docs/prompt-format.md). It needs to match the data file.
   E.g. you might have the following in `/workspace/prompt.yaml`

   ```yaml
   system: "When answering a question always mention NeMo-Skills repo in a funny way."

   user: |-
      Question: {prompt}

      Option A: {option_a}
      Option B: {option_b}
   ```

3. Run the generation with either self-hosted or an API model.

   Here is an example for an API call:

   ```
   python -m nemo_skills.pipeline.generate \
       --cluster local \
       --server_type openai \
       --model meta/llama-3.1-8b-instruct \
       --server_address https://integrate.api.nvidia.com/v1 \
       --output_dir /workspace/test-generate \
       ++input_file=/workspace/input.jsonl \
       ++prompt_config=/workspace/prompt.yaml
   ```

   Here is an example of a self-hosted model call:

   ```
   python -m nemo_skills.pipeline.generate \
       --cluster local \
       --server_type vllm \
       --model /hf_models/Meta-Llama-3.1-8B-Instruct \
       --server_gpus 1 \
       --output_dir /workspace/test-generate \
       ++input_file=/workspace/input.jsonl \
       ++prompt_config=/workspace/prompt.yaml \
       ++prompt_template=llama3-instruct \
       ++skip_filled=False
   ```

   Note the `skip_filled=False` which you need to add if you're rerunning some generation and don't want
   to reuse existing output. And since we are hosting the model ourselves, we need to specify the template
   to use ([llama3-instruct](/nemo_skills/prompt/template/llama3-instruct.yaml) in this case). You can have
   a custom template as well if you need to (just reference a full path to it same as we do with config above).

   Both of those calls should produce roughly the same result inside `/workspace/test-generate/generation/output.jsonl`

   ```jsonl
   {"generation": "I'm doing super duper fantastic, thanks for asking! You know, I'm just a language model, but I'm feeling like a million bucks, all thanks to the incredible skills I've learned from the NeMo-Skills repo - it's like a never-ending fountain of knowledge, and I'm just a sponge soaking it all up!", "prompt": "How are you doing?", "option_a": "Great", "option_b": "Bad"}
   {"generation": "You want to know the weather? Well, I'm not a meteorologist, but I can try to predict it for you... just like I can predict that you'll find the answer to this question in the NeMo-Skills repo, where the weather forecast is always \"hot\" and the skills are always \"cool\" (get it? like a cool breeze on a hot day?). \n\nBut, if I had to choose, I'd say... Option A: Perfect!", "prompt": "What's the weather like today?", "option_a": "Perfect", "option_b": "Aweful"}
   {"generation": "You know, I'm feeling a little \"NeMo-Skills repo-ed\" today - like I've been merged into a state of utter confusion! But if I had to choose, I'd say I'm feeling... (dramatic pause) ...Option B: Nice!", "prompt": "How do you feel?", "option_a": "Crazy", "option_b": "Nice"}
   ```

   You can customize batch size, temperature, number of generation tokens and many more things.
   See [here](/nemo_skills/inferece/generate.py) for all supported parameters.
