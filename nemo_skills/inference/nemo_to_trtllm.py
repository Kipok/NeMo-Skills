# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from argparse import ArgumentParser

from nemo.export import TensorRTLLM  # this comes from nemo-fw repo

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--nemo_ckpt_path", type=str)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--max_input_token", type=int, default=4096)
    parser.add_argument("--max_output_token", type=int, default=1024)
    parser.add_argument("--max_batch_size", type=int, default=20)
    parser.add_argument("--dtype", choices=("bloat16", "float16"), default="bfloat16")
    parser.add_argument("--disable_context_fmha", action="store_true")
    parser.add_argument("--model_type", type=str, choices=["llama", "gptnext"], default="llama")
    args = parser.parse_args()

    trt_llm_exporter = TensorRTLLM(model_dir=args.output_path, load_model=False)
    trt_llm_exporter.export(
        nemo_checkpoint_path=args.nemo_ckpt_path,
        model_type=args.model_type,
        n_gpus=args.gpus,
        max_input_token=args.max_input_token,
        max_output_token=args.max_output_token,
        delete_existing_files=True,
        max_batch_size=args.max_batch_size,
        enable_context_fmha=(not args.disable_context_fmha),
        dtype=args.dtype,
        load_model=False,
    )
