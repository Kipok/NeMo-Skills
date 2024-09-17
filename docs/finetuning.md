# Supervised finetuning

instead of run_pipeline.py

EXPNAME=test-training

python nemo_skills/pipeline/train.py \
    --cluster draco-ord \
    --expname $EXPNAME \
    --output_dir /exps/checkpoints/$EXPNAME \
    --nemo_model /nemo_models/llama3.1-8b-base \
    --num_nodes 2 \
    --num_gpus 8 \
    --num_training_jobs 2 \
    --average_steps 4000 8000 12000 16000 20000 24000 \
    --training_data /data/llama_data_all_max_len_1024.jsonl \
    ++model.data.train_ds.add_eos=False \
    ++model.data.train_ds.global_batch_size=1024 \
    ++model.data.train_ds.micro_batch_size=4 \
    ++trainer.sft.val_check_interval=10 \
    ++trainer.sft.save_interval=10 \
    ++trainer.sft.limit_val_batches=1 \
    ++trainer.sft.max_steps=30 \
    ++trainer.sft.max_epochs=10 \
    ++model.optim.lr=5e-6 \
    ++model.optim.sched.warmup_steps=0 \
    ++model.tensor_model_parallel_size=4 \
    ++model.pipeline_model_parallel_size=1 \
    ++exp_manager.checkpoint_callback_params.save_top_k=50 \
    ++exp_manager.max_time_per_run=00:03:45:00 \
    --partition interactive

python nemo_skills/pipeline/eval.py \
    --cluster draco-ord \
    --model /exps/checkpoints/$EXPNAME/model-averaged.nemo \
    --server_type nemo \
    --output_dir /exps/results/$EXPNAME \
    --benchmarks gsm8k:0 math:0 \
    --server_gpus 8 \
    --server_nodes 1 \
    --run_after $EXPNAME \
    ++prompt_template=llama3-instruct \
    ++batch_size=128 \
    ++split_name=test


or with conversion

python nemo_skills/pipeline/convert.py \
    --cluster draco-ord \
    --input_model /exps/checkpoints/$EXPNAME/model-averaged.nemo \
    --output_model /exps/checkpoints/$EXPNAME/model-averaged-hf \
    --expname $EXPNAME-to-hf \
    --convert_to hf \
    --num_gpus 8 \
    --hf_model_name meta-llama/Meta-Llama-3.1-8B

python nemo_skills/pipeline/convert.py \
    --cluster draco-ord \
    --input_model /exps/checkpoints/$EXPNAME/model-averaged-hf \
    --output_model /exps/checkpoints/$EXPNAME/model-averaged-trtllm \
    --expname $EXPNAME-to-trtllm \
    --run_after $EXPNAME-to-hf \
    --num_gpus 8 \
    --convert_from hf \
    --convert_to trtllm \

python nemo_skills/pipeline/eval.py \
    --cluster draco-ord \
    --model /exps/checkpoints/$EXPNAME/model-averaged-trtllm \
    --server_type trtllm \
    --output_dir /exps/results/$EXPNAME/trtllm-eval \
    --benchmarks gsm8k:0 math:0 \
    --server_gpus 8 \
    --server_nodes 1 \
    --num_jobs 1 \
    --run_after $EXPNAME-to-trtllm \
    ++prompt_template=llama3-instruct \
    ++batch_size=512 \
    ++split_name=test




Supervised finetuning (SFT) is the final stage of our pipeline. We use [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner/)
to run SFT and would encourage you to check their documentation to learn more details.
Here are the commands to prepare data, run SFT and evaluate the finetuned models.

Make sure to complete [prerequisites](/docs/prerequisites.md) before proceeding.

1. Prepare the dataset, if you've [generated new solutions](/docs/synthetic-data-generation.md).
   E.g. for gsm8k train_full subset (which combines our custom train-validation split together)

   ```
   python nemo_skills/finetuning/prepare_sft_data.py \
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
   --extra_eval_args="--num_jobs=2 ++split_name=test ++batch_size=32 "
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
