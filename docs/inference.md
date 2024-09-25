# Inference

Here are the instructions on how to run inference with our repo.

Make sure to complete [prerequisites](/docs/prerequisites.md) before proceeding.

1. Get the model you want to use. You can use any model that's supported by VLLM, TensorRT-LLM or NeMo.
   You can also use [Nvidia NIM API](https://www.nvidia.com/en-us/ai/) for models that are hosted there.

2. [Convert the model](/docs/checkpoint-conversion.md) if it's not in the format you want to use.
   You do not need any conversion if using VLLM inference with HF models.
   For fastest inference we recommend to convert the model to TensorRT-LLM format.

3. Start the server hosting your model. Here is an example (make sure the `/hf_models` mount is defined in your cluster config).

   ```
   python -m nemo_skills.pipeline.start_server \
       --cluster local \
       --model /hf_models/Meta-Llama-3.1-8B-Instruct \
       --server_type vllm \
       --server_gpus 1 \
       --server_nodes 1
   ```

4. Run inference

   ```
   from nemo_skills.inference.server import get_model
   from nemo_skills.prompt.utils import Prompt, get_prompt
   import os
   import json
   with open('our-70b-eval-results/aime24/output-greedy.jsonl') as fin:
       data = [json.loads(elem) for elem in fin.readlines()]

   os.environ['NEMO_SKILLS_SSH_KEY_PATH'] = '~/.ssh/clusters/draco-ord'
   # os.environ['NEMO_SKILLS_SSH_SERVER'] = 'igitman@draco-oci-login-01.draco-oci-iad.nvidia.com'
   os.environ['NEMO_SKILLS_SSH_SERVER'] = 'igitman@cs-oci-ord-login-01.nvidia.com'

   sandbox = get_sandbox(
       sandbox_type="local",
       host='batch-block1-2034',
   )

   llm = get_code_execution_model(
       server_type='trtllm',
       host='batch-block1-2034',
       sandbox=sandbox,
   )

   prompt = get_prompt('generic/math', 'llama3-instruct')

   prompts = [prompt.build_string({
       'problem': 'There are 3 cats worth 21 dollars in total. What\'s the price of a dog?',
   })]
   print(prompts[0])
   outputs = llm.generate(
       prompts=prompts,
       stop_phrases=list(prompt.config.template.stop_phrases),
       tokens_to_generate=4096,
       temperature=0.0,
       random_seed=1,
   )
   print(outputs[0]["generation"])
   ```



## With code execution

Note that you cannot use a simple LLM call for models that rely on Python
code interpreter to execute parts of the output.

3. Start model server and local [sandbox](/docs/sandbox.md) for code execution.
   We recommend using TensorRT-LLM for fastest inference,
   but NeMo or vLLM might be easier to setup (works on more GPU types and does **not** require
   conversion for pretrained models).

   You can also run this command on a slurm cluster and then submit requests from a local workstation by setting up
   `NEMO_SKILLS_SSH_SERVER` and `NEMO_SKILLS_SSH_KEY_PATH` environment variables (if you have ssh access there).

   ```
   python pipeline/start_server.py \
       --model_path <path to the model in the right format> \
       --server_type <nemo or trtllm> \
       --num_gpus <number of GPUs you want to use>
   ```

   Make sure to provide the path to the `nemo_model` subfolder if using NeMo models.

4. Wait until you see "Server is running on" message, then send requests to the server through Python API or by using our [visualization tool](/visualization/Readme.md).

    ```python
    from nemo_skills.inference.server.code_execution_model import get_code_execution_model
    from nemo_skills.code_execution.sandbox import get_sandbox
    from nemo_skills.prompt.utils import Prompt, get_prompt_config

    sandbox = get_sandbox(
        sandbox_type="local",
        host=<IP address of sandbox printed on previous step>,
    )

    llm = get_code_execution_model(
        server_type=<server type from previous step>,
        host=<IP address of server printed on previous step>,
        sandbox=sandbox,
    )

    # replace with "openmathinstruct/base" if model that was not pretrained with our pipeline
    # check out other yaml files inside nemo_skills/inference/prompt
    # or write your own to customize further
    prompt_config = get_prompt_config("openmathinstruct/sft")
    prompt_config.few_shot_examples.num_few_shots = 0
    # replace with the following if model that was not pretrained with our pipeline
    # you can pick different few shot examples based on your needs
    # prompt_config.few_shot_examples.num_few_shots = 5
    # prompt_config.few_shot_examples.examples_type = "gsm8k_text_with_code"

    question = (
        "In a dance class of 20 students, 20% enrolled in contemporary dance, "
        "25% of the remaining enrolled in jazz dance, and the rest enrolled in "
        "hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?"
    )
    prompt_template = Prompt(config=prompt_config)
    # can provide multiple requests
    prompts = [prompt_template.build_string({'question': question})]

    outputs = llm.generate(
        prompts=prompts,
        stop_phrases=list(prompt_config.stop_phrases),
    )
    print(outputs[0]["generation"])
    ```

## Without code execution

All of the released OpenMath models require code execution, so if you want to
use one of them, please refer to the previous section. If you want to try
some other model (e.g. llama3) and don't want to execute code, you can use
the following simpler workflow.

3. Start model server.  We recommend using TensorRT-LLM for fastest inference,
   but NeMo or vLLM might be easier to setup (works on more GPU types and does not require
   conversion for pretrained models).

   You can also run this command on a slurm cluster and then submit requests from a local workstation by setting up
   `NEMO_SKILLS_SSH_SERVER` and `NEMO_SKILLS_SSH_KEY_PATH` environment variables (if you have ssh access there).

   ```
   python pipeline/start_server.py \
       --model_path <path to the model in the right format> \
       --server_type <nemo or trtllm> \
       --num_gpus <number of GPUs you want to use> \
       --no_sandbox
   ```

   Make sure to provide the path to the `nemo_model` subfolder if using NeMo models.

4. Wait until you see "Server is running on" message, then send requests to the server through Python API or by using our [visualization tool](/visualization/Readme.md).

    ```python
    from nemo_skills.inference.server.model import get_model
    from nemo_skills.prompt.utils import Prompt, get_prompt_config

    llm = get_model(
        server_type=<server type from previous step>,
        host=<IP address of server printed on previous step>,
    )

    # check out other yaml files inside nemo_skills/inference/prompt
    # or write your own to customize further
    prompt_config = get_prompt_config("llama3/instruct")
    prompt_config.few_shot_examples.num_few_shots = 5
    prompt_config.few_shot_examples.examples_type = "gsm8k_only_text"

    question = (
        "In a dance class of 20 students, 20% enrolled in contemporary dance, "
        "25% of the remaining enrolled in jazz dance, and the rest enrolled in "
        "hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?"
    )
    prompt_template = Prompt(config=prompt_config)
    # can provide multiple requests
    prompts = [prompt_template.build_string({'question': question})]

    outputs = llm.generate(
        prompts=prompts,
        stop_phrases=list(prompt_config.stop_phrases),
    )
    print(outputs[0]["generation"])
    ```

## Without prompt format

If you just want to use our server hosting code and have another way to prepare
model prompts in the correct format, you can leverage the following minimal api.

```python
from nemo_skills.inference.server.model import get_model

llm = get_model(
    server_type=<server type from previous step>,
    host=<IP address of server printed on previous step>,
)

prompts = [  # can provide multiple requests
    "In a dance class of 20 students, 20% enrolled in contemporary dance, "
    "25% of the remaining enrolled in jazz dance, and the rest enrolled in "
    "hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?"
]
outputs = llm.generate(prompts=prompts)
print(outputs[0]["generation"])
```


If running locally, the server might sometimes hang around
after `start_server.py` is already killed. In that case you should manually stop it,
e.g. by running docker stop with the id of the running container.

If for some reason you do not want to use [pipeline/start_server.py](/pipeline/start_server.py) helper script,
you can always start sandbox and LLM server manually. We describe this process in
[evaluation/details](/docs/evaluation.md#details) section.