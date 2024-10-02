# How to reproduce our results

Make sure to complete [prerequisites](/docs/prerequisites.md).

If you want to reproduce results for [OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset](https://arxiv.org/abs/2402.10176)
please check out v0.1.1 branch of the repository and read the instructions in there.

```bash
git checkout v0.1.1
```

Below are the instructions for reproducing
[OpenMathInstruct-1: Accelerating AI for Math with Massive Open-Source Instruction Data](TBD).

Please note that unless you have an access to a large GPU cluster, it might take a very long time
for some of the commands to complete!

## Evaluation

Here are the commands you can run to reproduce our evaluation numbers.
The commands below are for OpenMath-2-Llama3.1-8b model as an example.
We assume you have `/workspace` defined in your cluster config and are
executing all commands from that folder locally. Change all commands accordingly
if running on slurm or using different paths.

1. Get the model from HF. E.g.

   ```
   pip install -U "huggingface_hub[cli]"
   huggingface-cli download nvidia/OpenMath2-Llama3.1-8B --local-dir OpenMath2-Llama3.1-8B
   ```

2. Convert the model to TensorRT-LLM format. This is optional, but highly recommended for more exact
   results and faster inference. If you skip it, replace `--server_type trtllm` with `--server-type vllm`
   in the commands below and change model path to `/workspace/OpenMath2-Llama3.1-8B`. You might also need
   to set smaller batch size for vllm.

   ```
   ns convert \
       --cluster=local \
       --input_model=/workspace/OpenMath2-Llama3.1-8B \
       --output_model=/workspace/openmath2-llama3.1-8b-trtllm \
       --convert_from=hf \
       --convert_to=trtllm \
       --num_gpus=1 \
       --hf_model_name=nvidia/OpenMath2-Llama3.1-8B
   ```

   Change the number of GPUs if you have more than 1 (required for 70B model).

3. Run greedy decoding.

   ```
   ns eval \
       --cluster=local \
       --model=/workspace/openmath2-llama3.1-8b-trtllm \
       --server_type=trtllm \
       --output_dir=/workspace/openmath2-llama3.1-8b-eval \
       --benchmarks=aime24:0,amc23:0,math:0,gsm8k:0,omni-math:0 \
       --server_gpus=1 \
       --num_jobs=1 \
       ++prompt_template=llama3-instruct \
       ++batch_size=512 \
       ++inference.tokens_to_generate=4096
   ```

   If running on slurm, you can set `--num_jobs` to a bigger number of -1 to run
   each benchmark in a separate node. The number of GPUs need to match what you used
   in the conversion command.

   After the generation is done, we want to run LLM-as-a-judge evaluation to get more
   accurate numbers than symbolic comparison.

   ```
   ns llm_math_judge \
       --cluster=local \
       --model=gpt-4o \
       --server_type=openai \
       --server_address=https://api.openai.com/v1 \
       --input_files="/workspace/openmath2-llama3.1-8b-eval/eval-results/**/output*.jsonl"
   ```

   Finally, to print the metrics run

   ```
   ns summarize_results /workspace/openmath2-llama3.1-8b-eval/ --cluster local
   ```

   This should print the metrics including both symbolic and judge evaluation. The judge is typically more accurate.

   ```
   TBD
   ```

## Dataset construction

TBD


## Model training

TBD