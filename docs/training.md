# Training an LLM

Make sure to complete [prerequisites](/docs/prerequisites.md).

Please refer to the following docs if you have questions about:
- [Prompt format](/docs/prompt-format.md)

## Preparing the data

Before running the training we need to prepare the data in the right format. Here is an example command

```
python -m nemo_skills.training.prepare_sft_data \
    ++input_files="<path to the generated synthetic data>/output-rs*.jsonl <you can have multiple paths as well>" \
    ++output_path=sft-data.jsonl \
    ++prompt_config=generic/math \
    ++prompt_template=llama3-instruct \
    ++generation_suffix='\"<|eot_id|>\"'
```

Note that unlike most other scripts, this one doesn't accept a `--cluster` parameter and you can currently only run
it locally (this will be changed soon).

You need to pass in the config/template files so that we can format the data accordingly. There are many more parameters
that data preparation script supports which you can see [here](/nemo_skills/training/data_preparation_utils/prepare_sft_data.yaml).
We are using [SDP library](https://github.com/NVIDIA/NeMo-speech-data-processor) for preparing the data, so it's
a good idea to check their documentation to understand how this config is structured.

> **_NOTE:_** Even though we support both SFT and DPO training, the data preparation is currently only implemented
> for SFT jobs. For DPO, you'd need to manually prepare the data according to the
> [NeMo-Aligner documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/dpo.html#dpo-model-training).
> We will add a proper support for DPO data preparation in the near future.


## Running training

We use [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner/) to run LLM training,
so you can check their documentation to learn about all supported parameters.

Here is an example of how to run a training job.

python -m nemo_skills.pipeline.train \
    --cluster=slurm \
    --expname=my-training-job \
    --output_dir=/workspace/my-training-job/checkpoints \
    --nemo_model=/nemo_models/llama3.1-8b-base \
    --num_nodes=8 \
    --num_gpus=8 \
    --num_training_jobs=4 \
    --training_data=/data/sft-data.jsonl

This will run training on 8 nodes of 8 GPUs, using 4 dependent slurm jobs.
By default we are training for 2 epochs, saving checkpoints every 1000 steps,
but you can adjust these values. It's also recommended to tune micro batch size
and tensor parallel parameters for optimal performance. E.g. these are good
defaults for an 8B model size

```
    ++model.data.train_ds.micro_batch_size=4 \
    ++model.tensor_model_parallel_size=4
```

The training script will average all of your generated checkpoints upon completion
(we found this to consistently increase the downstream accuracy). If you want to
only average a subset of checkpoint, add `--average_steps` parameter (e.g. if you
want to disable averaging, set it to the last training step). If you only want
to average the checkpoints of the finished job, set `--num_training_jobs=0`.

Typically after training we want to follow up with evaluation. You can schedule
an evaluation job right away by providing a `--run_after my-training-job` argument
which will appropriately set slurm dependencies.

```
python -m nemo_skills.pipeline.eval \
    --cluster=slurm \
    --model=/workspace/my-training-job/checkpoints/model-averaged-nemo \
    --server_type=nemo \
    --output_dir=/workspace/my-training-job/results/ \
    --benchmarks gsm8k:0 math:0 \
    --server_gpus=8 \
    --run_after=my-training-job \
    ++prompt_template=llama3-instruct \
    ++batch_size=128
```

In general we don't recommend to run inference using NeMo checkpoints as it is
much slower than other server formats. Here is how you can chain the commands
to schedule checkpoint conversion and evaluation after training.

```
python -m nemo_skills.pipeline.convert \
    --cluster=slurm \
    --input_model=/workspace/my-training-job/checkpoints/model-averaged-nemo \
    --output_model=/workspace/my-training-job/checkpoints/model-averaged-hf \
    --expname=my-training-job-to-hf \
    --run_after=my-training-job \
    --convert_from=nemo \
    --convert_to=hf \
    --num_gpus=8 \
    --hf_model_name=meta-llama/Meta-Llama-3.1-8B

python -m nemo_skills.pipeline.convert \
    --cluster=slurm \
    --input_model=/workspace/my-training-job/checkpoints/model-averaged-hf \
    --output_model=/workspace/my-training-job/checkpoints/model-averaged-trtllm \
    --expname=my-training-job-to-trtllm \
    --run_after=training-job-to-hf \
    --convert_from=hf \
    --convert_to=trtllm \
    --num_gpus=8

python -m nemo_skills.pipeline.eval \
    --cluster=slurm \
    --model=/workspace/my-training-job/checkpoints/model-averaged-nemo \
    --server_type=nemo \
    --output_dir=/workspace/my-training-job/results/ \
    --benchmarks gsm8k:0 math:0 \
    --server_gpus=8 \
    --run_after=my-training-job-to-trtllm \
    ++prompt_template=llama3-instruct \
    ++batch_size=128
```



Supervised training (SFT) is the final stage of our pipeline. We use [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner/)
to run SFT and would encourage you to check their documentation to learn more details.
Here are the commands to prepare data, run SFT and evaluate the finetuned models.

Make sure to complete [prerequisites](/docs/prerequisites.md).

1. Prepare the dataset, if you've [generated new solutions](/docs/synthetic-data-generation.md).
   E.g. for gsm8k train_full subset (which combines our custom train-validation split together)

   ```
   python nemo_skills/training/prepare_sft_data.py \
       ++prediction_jsonl_files=<path to the generated synthetic data>/output-rs*.jsonl \
       ++output_path=sft-data.jsonl
   ```

   Note that `prediction_jsonl_files` can accept multiple glob patterns separated by space.

2. Run SFT + checkpoint averaging + evaluation in one script.

   ```
   python pipeline/run_pipeline.py \
      --expname <name for experiment> \
      --nemo_model <path to the nemo model> \
      --num_nodes <number of nodes> \
      --num_gpus <number of GPUs per node> \
      --extra_eval_args="+prompt=openmathinstruct/sft" \
      ++model.data.train_ds.file_path=/data/<path to the data inside NEMO_SKILLS_DATA>
   ```

   This will put all checkpoints, results and logs inside `$NEMO_SKILLS_RESULTS`.
   Note that you can provide `--stages` argument to control which steps to run. E.g.
   to skip evaluation use `--stages sft prepare_eval` or to only run evaluation
   (e.g. to re-run with different sampling parameters) use `--stages eval`.

   You can also customize any evaluation parameters with `--extra_eval_args`, e.g.
   to use 2 evaluation jobs, batch size of 32 and evaluate on the test set use

   ```
   --extra_eval_args="--num_jobs=2 ++split=test ++batch_size=32 "
   ```

   You can customize any of the SFT parameters by directly providing those
   arguments to the [pipeline/run_pipeline.py](/pipeline/run_pipeline.py) script (training data is already customized
   in the example above). E.g. to disable dropout and use tensorboard logging instead of wandb you can set

   ```
   --disable_wandb \
   ++model.ffn_dropout=0.0 \
   ++model.attention_dropout=0.0 \
   ++model.hidden_dropout=0.0
   ```

Alternatively, you can run all the steps separately.

1. Run SFT

   ```
   python pipeline/run_training.py \
      --expname <name for experiment> \
      --checkpoints_folder <where to save checkpoints>
      --nemo_model <path to the nemo model> \
      --num_nodes <number of nodes> \
      --num_gpus <number of GPUs per node> \
      ++model.data.train_ds.file_path=/data/<path to the data inside NEMO_SKILLS_DATA folder>
   ```

   Note that you cannot submit multiple dependent jobs with this script and would have to do this manually if required.

2. Run checkpoint averaging

   ```
   python pipeline/prepare_eval.py \
       --training_folder <path to the checkpoints folder above>/training/checkpoints \
       --output_path <where to place the averaged model> \
       --nemo_model <same as in the run sft step, needed to get config>
   ```

3. Run [evaluation](/docs/evaluation.md)
