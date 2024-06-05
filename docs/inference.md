# Inference

Here are the instructions on how to run inference with our models.

Make sure to complete [prerequisites](/docs/prerequisites.md) before proceeding.

1. Get the model you want to use. You can use one of our
   [pretrained models](https://huggingface.co/collections/nvidia/openmath-65c5619de2ba059be0775014)
   in a zero-shot setting or any other model with few-shot examples of code usage in the prompt.

2. [Convert the model](/docs/checkpoint-conversion.md) if it's not in the right format.
   You do not need any conversion if using one of our NeMo models and can refer to
   the exact steps on how to convert HF version of the models to TensorRT-LLM format
   [here](/docs/reproducing-results.md#evaluation).

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
       --server_type <nemo or tensorrt_llm> \
       --num_gpus <number of GPUs you want to use>
   ```

   Make sure to provide the path to the `nemo_model` subfolder if using NeMo models.

4. Wait until you see "Server is running on" message, then send requests to the server through Python API or by using our [visualization tool](/visualization/Readme.md).

    ```python
    from nemo_skills.inference.server.code_execution_model import get_code_execution_model
    from nemo_skills.code_execution.sandbox import get_sandbox

    sandbox = get_sandbox(
        sandbox_type="local",
        host=<IP address of sandbox printed on previous step>,
    )

    llm = get_code_execution_model(
        server_type=<server type from previous step>,
        host=<IP address of server printed on previous step>,
        sandbox=sandbox,
    )

    prompts = [  # can provide multiple requests
        "In a dance class of 20 students, 20% enrolled in contemporary dance, "
        "25% of the remaining enrolled in jazz dance, and the rest enrolled in "
        "hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?"
    ]
    outputs = llm.generate(prompts=prompts)
    print(outputs[0]["generation"])
    ```

## Without code execution

3. Start model server.  We recommend using TensorRT-LLM for fastest inference,
   but NeMo or vLLM might be easier to setup (works on more GPU types and does not require
   conversion for pretrained models).

   You can also run this command on a slurm cluster and then submit requests from a local workstation by setting up
   `NEMO_SKILLS_SSH_SERVER` and `NEMO_SKILLS_SSH_KEY_PATH` environment variables (if you have ssh access there).

   ```
   python pipeline/start_server.py \
       --model_path <path to the model in the right format> \
       --server_type <nemo or tensorrt_llm> \
       --num_gpus <number of GPUs you want to use> \
       --no_sandbox
   ```

   Make sure to provide the path to the `nemo_model` subfolder if using NeMo models.

4. Wait until you see "Server is running on" message, then send requests to the server through Python API or by using our [visualization tool](/visualization/Readme.md).

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