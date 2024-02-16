# How to reproduce our results

Make sure to complete [prerequisites](/docs/prerequisites.md) before proceeding.

Please note that unless you have an access to a large GPU cluster, it might take a long time
for some of the commands to complete!

All commands were tested on a node with 8 80Gb A100 GPUs.
If you're using different GPU configuration, change the commands accordingly and
expect ~1% variation in results.

## Evaluation

Here are the commands you can run to reproduce our evaluation numbers.
The commands below are for Mistral-7B model as an example. They are identical for all models,
except we use batch size of 16 for 34B+ model sizes.

1. Get the model from HuggingFace

   ```
   git clone https://huggingface.co/nvidia/OpenMath-Mistral-7B-v0.1-hf
   ```

2. Convert the model to TensorRT-LLM format for fastest evaluation.

   ```
   docker run --rm --gpus all --ipc=host -v <path to nemo-skills repo>:/code -v <path to OpenMath-Mistral-7B-v0.1-hf>:/model igitman/nemo-skills-trtllm:0.1.0 \
   bash -c ' \
   export PYTHONPATH=/code && cd /code && \
   python nemo_skills/conversion/hf_to_trtllm.py \
      --model_dir /model \
      --output_dir /tmp/trtllm \
      --dtype bfloat16 \
      --tp_size 8 && \
   trtllm-build \
      --checkpoint_dir /tmp/trtllm \
      --output_dir /code/openmath-mistral-7b-trtllm \
      --gpt_attention_plugin bfloat16 \
      --gemm_plugin bfloat16 \
      --context_fmha "enable" \
      --max_input_len 4096 \
      --max_output_len 512 \
      --max_batch_size 64 &&
   cp /model/tokenizer.model /code/openmath-mistral-7b-trtllm'
   ```

3. Run greedy decoding for all datasets. You can increase number of nodes if running on Slurm cluster for faster evaluation.

   ```
   python pipeline/run_eval.py \
     --model_path `pwd`/openmath-mistral-7b-trtllm \
     --server_type tensorrt_llm \
     --output_dir `pwd`/openmath-mistral-7b-eval-results \
     --benchmarks gsm8k:0 asdiv:0 gsm-hard:0 mawps:0 svamp:0 tabmwp:0 algebra222:0 math:0 \
     --num_gpus 8 \
     --num_nodes 1 \
     +prompt=code_sfted \
     ++prompt.num_few_shots=0 \
     ++split_name=test \
     ++server.max_code_executions=6 \
     ++server.stop_on_code_error=False \
     ++batch_size=64
   ```

4. Run 50 samples for gsm8k and math to get self-consistency numbers.

   ```
   python pipeline/run_eval.py \
     --model_path `pwd`/openmath-mistral-7b-trtllm \
     --server_type tensorrt_llm \
     --output_dir `pwd`/openmath-mistral-7b-eval-results \
     --benchmarks gsm8k:50 math:50 \
     --num_gpus 8 \
     --num_nodes 1 \
     +prompt=code_sfted \
     ++prompt.num_few_shots=0 \
     ++skip_filled=True \
     ++split_name=test \
     ++server.max_code_executions=6 \
     ++server.stop_on_code_error=False \
     ++batch_size=64
   ```

   If the above command fails with mpi error, try to reduce number of samples
   per run, e.g. use `gsm8k:10 math:10` and then do another run with
   `--starting_seed 10` and so on.

5. Compute and summarize all metrics.

   ```
   python pipeline/summarize_results.py `pwd`/openmath-mistral-7b-eval-results
   ```

   If you get permission errors, run the command with sudo or change permissions of the results folder.

## Dataset generation

To re-create the masked versions of the GSM8K and MATH datasets, you can follow the steps
[here](/docs/synthetic-data-generation.md#masked-solutions).

To re-create the GSM8K version of the [OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) dataset
you can run the following:

1. Convert [Mixtral-8x7B base](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) model to TensorRT-LLM format.
   Refer to the instructions from steps 1-2 in the [reproducing evaluation](#evaluation) section as well as
   [checkpoint conversion docs](/docs/checkpoint-conversion.md). Use `--max_input_len 4096` and `--max_output_len 512`
   but you might need to change the other parameters based on the exact GPUs configuration.

2. GSM8K portion of the dataset.

   Without reference solutions:

   ```
   python pipeline/run_labeling.py \
     --model_path <path to trtllm model> \
     --server_type tensorrt_llm \
     --output_dir ./synthetic-solutions/gsm8k/ \
     --num_gpus 8 \
     --num_runs 128 \
     +prompt=code_base \
     ++prompt.examples_type=gsm8k_text_with_code \
     ++prompt.context_type=empty \
     ++dataset=gsm8k \
     ++split_name=train_full
   ```

   With masked reference solutions:

   ```
   python pipeline/run_labeling.py \
     --model_path <path to trtllm model> \
     --server_type tensorrt_llm \
     --output_dir ./synthetic-solutions/gsm8k-masked/ \
     --num_gpus 8 \
     --num_runs 128 \
     +prompt=code_base \
     ++prompt.examples_type=gsm8k_text_with_code \
     ++prompt.context_type=masked_solution \
     ++dataset=gsm8k-masked \
     ++split_name=train_full
   ```

   For the MATH portion of the dataset, change `gsm8k -> math` and use 224 runs.
   Additionally, we generated 32 samples for each of the following example types
   corresponding to the subject-specific prompts we created.

   ```
   math_algebra, math_probability, math_prealgebra, math_number_theory,
   math_geometry, math_precalculus, math_intermediate_algebra
   ```

   Make sure to use different output folder for each generation to not override results.

## Model finetuning

Coming soon!