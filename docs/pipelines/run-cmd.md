# Running arbitrary commands

!!! info

    This pipeline starting script is [nemo_skills/pipeline/run_cmd.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/pipeline/run_cmd.py)

    All extra parameters are directly executed as a shell command.

We often need to run arbitrary pre/post processing commands as part of a larger pipeline and thus we provide a simple
`run_cmd` utility that can be used to schedule those on slurm. Here is an example that simply enters the packaged
code and tries to install it (will finish with error if not running from Nemo-Skills repo or other installable package).

```bash
ns run_cmd --cluster=local cd /nemo_run/code/ && pip install -e .
```

There are many more examples of how to use `run_cmd` throughout our documentation.

## LLM Server and Sandbox Server

While we can run arbitrary commands with the default `run_cmd` script, we also provide the ability to
run a LLM server with the `--model` argument and a few extra arguments for the server config. These arguments
are similar to the ones used for `start_server` script.

This can be useful to run a server on a local machine or on a cluster with GPUs in a slurm job, while also being able to
run arbirary code that uses LLM calls.

### Example

Say you have the following inference file that uses OpenAI API with a vLLM backed server (say to run a
project that is compatible with OpenAI API). Imagine a file called `inference.py` with the following code:

```python
from openai import OpenAI
client = OpenAI(api_key='EMPTY', base_url=f"http://0.0.0.0:5000/v1", timeout=None)
api_model = client.models.list().data[0].id

response = client.chat.completions.create(
    model=api_model,
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=0.7,
    max_tokens=128,
    top_p=0.95,
    n=1,
    stream=False,
)
print(response.choices[0].message.content)
```

Then we can run the server and the inference code in a single command as below. The --with_sandbox argument starts the
code execution server that can be used to run arbitrary code in a sandboxed environment and is added here just as a
demonstration. While the current example does not use it, this can be useful to execute code or to run code that
requires a specific environment in a container.

**Note**: While the container is a little more secure than running code directly on the host, it is still not a fully
secure sandbox and should not be used to run untrusted code.

```bash
ns run_cmd \
    --cluster=local \
    --model=Qwen/Qwen3-1.7B \
    --server_type=vllm \
    --server_gpus=1 \
    --with_sandbox \
    "cd /nemo_run/code/ && python inference.py"
```

This will launch the LLM inference server, the sandbox server and then run the inference code.

### Run once inside a Slurm Ray cluster

Pass `--with_ray` to run the command on the Ray head inside a multi-node Slurm
allocation. This uses the existing NeMo-Run Ray launcher and leaves ordinary
`run_cmd` jobs unchanged. For example:

```bash
ns run_cmd --cluster=my-cluster --container=my-public-image \
  --num_nodes=4 --num_gpus=8 --with_ray \
  'cd /nemo_run/code/my-project && uv run --locked driver.py'
```

Python callers pass `with_ray=True`. The environment invoking `run_cmd` must
install the Skills version exposing this option; packaged runtime source does
not change the host launcher's Python signature.

Python callers can also pass `ray_template` (CLI `--ray_template`) through the
existing task/executor template API. The checked-in
`nemo_skills/pipeline/nemo_rl/ray_templates/nemo_skills_sandbox_ray.sub.j2`
works with ordinary Ray jobs and optional sandbox roles, without rewriting
installed Ray source. Nsight support must come from the chosen Ray package.
Set the allocation's `RAY_EXECUTABLE` to the absolute `bin/ray` in the same
prepared environment used by the driver. All start/status commands use it;
the default remains `ray`. Build and verify that environment before allocating
training GPUs. This setting does not change the host launcher's Python/PATH.

If the image requires normal package setup in each Ray container, declare
`RAY_NODE_SETUP_SCRIPT` in the cluster environment. The checked-in template
executes that script inside both head and worker containers before starting Ray,
and stops that container if setup fails. The script must be available through a
normal packaged or mounted path. This supports image-specific public package
installation without changing the Ray bootstrap; leave it unset when the image
is already prepared. Also declare any environment settings required by Ray and
its actors in the cluster configuration, since changes made by the setup script
are local to its subprocess.
