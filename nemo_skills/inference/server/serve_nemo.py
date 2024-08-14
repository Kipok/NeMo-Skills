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

# adapted from https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/megatron_gpt_eval.py

import datetime
import logging
import os

import nemo.collections.nlp.modules.common.text_generation_utils as tgu
import torch
from megatron.core import parallel_state
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.text_generation_utils import (
    AppState,
    F,
    generate,
    get_model_parallel_src_rank,
    model_inference_strategy_dispatcher,
    receive_generate_info,
    repetition_penalty,
    seed_everything,
    send_generate_info,
    switch,
    tensor_parallel,
    top_k_logits,
)
from nemo.collections.nlp.parts.nlp_overrides import CustomProgressBar, NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils.apex_utils import _reconfigure_microbatch_calculator
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo_skills.inference.inference_strategy import CodeExecutionStrategy


def sample_sequence_batch(
    model,
    inference_strategy,
    context_tokens,
    context_lengths,
    tokens_to_generate,
    all_probs=False,
    compute_attention_mask=True,
    compute_logprob=False,
    type_ids=None,
    temperature=None,
    end_strings=['<|endoftext|>'],
    image_list=None,
    extra={},
):
    # Importing here to avoid circular import errors

    app_state = AppState()
    micro_batch_size = context_tokens.shape[0]
    _reconfigure_microbatch_calculator(
        rank=app_state.global_rank,
        rampup_batch_size=None,
        global_batch_size=micro_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=1,
    )
    assert (
        model.cfg.get('activations_checkpoint_granularity', None) is None
    ), 'activations_checkpoint_granularity should be None during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint'
    assert (
        model.cfg.get('activations_checkpoint_method', None) is None
    ), 'activations_checkpoint_method should be None during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint'

    tokenizer = model.tokenizer
    # initialize the batch
    with torch.no_grad():
        # ******************************* THIS PART IS CHANGED TO MAX TO RUN ALL PROMPTS IN ONE GEN **********************************
        context_length = context_lengths.max().item()
        # ****************************************************************************************************************************
        if 'neighbors_tokens' in extra:  # for Mcore retrieval RETRO model

            # For Mcore retrieval RETRO model, context_tokens tensors are updated after init_batch() (the length is doubled after processing)
            context_tokens = inference_strategy.init_batch(
                context_tokens, context_length, compute_attention_mask, **extra
            )

        else:
            inference_strategy.init_batch(context_tokens, context_length, compute_attention_mask)
        # added eos_id to support the function generate_samples_eval that passes
        # eos_id as an argument and needs termination when that id id found.
        eod_id = tokenizer.eos_id
        counter = 0

        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        output_logits = None
        all_generated_indices = None  # used to track all generated indices
        # Generate enough tokens for the longest sequence
        maxlen = tokens_to_generate + context_lengths.max().item()

        maxlen = inference_strategy.clip_max_len(maxlen)

        lengths = torch.ones([batch_size]).long().cuda() * maxlen

        # while context_length < maxlen:
        if image_list is not None:
            batch, tensor_shape = inference_strategy.prepare_batch_at_step(
                tokens, maxlen, micro_batch_size, counter, context_length, compute_attention_mask, image_list
            )
        else:
            batch, tensor_shape = inference_strategy.prepare_batch_at_step(
                tokens, maxlen, micro_batch_size, counter, context_length, compute_attention_mask
            )
        output = inference_strategy.forward_step(batch, tensor_shape)

        if parallel_state.is_pipeline_last_stage():

            output = output[0]['logits']
            output = tensor_parallel.gather_from_tensor_model_parallel_region(output)
            assert output is not None
            logits = output.contiguous()

            # *************************** LOGIT FUNCTION GOES HERE *********************************
            # logits.shape = torch.Size([bs, seqlen, vocab_size])
            logits = logits.float()
            probs = F.softmax(logits, dim=-1)
            outputs = []
            for bs_idx in range(probs.shape[0]):
                # the tokens are padded to max length in a batch, so taking the right slice. Note 1: in the beginning
                prompt_ids = context_tokens[bs_idx, 1 : context_lengths[bs_idx]]
                tokenprobs = probs[bs_idx, torch.arange(prompt_ids.shape[0]), prompt_ids]

                # as a sanity check I printed above on the model generated completion
                # and we see probs close to 1 (after the prompt tokens, for which it's all over the place)

                # if instead you want a top1 probs ignoring what's in the prompt but checking
                # what the model *wants* to generate, you can use the following:
                # tokenprobs = torch.topk(probs, k=1, dim=-1).values[:, 0][:-1]

                # if you want to see what the model wants to predict in text, call the following
                # tokenidx = torch.topk(probs[bs_idx, :context_lengths[bs_idx] - 1], k=1, dim=-1).indices[:, 0]
                # tokenizer.tokenizer.decode(tokenidx)

                # does it make sense to use a diff of the prompt token probs and the top1 probs?

                # before returning, we want to remove all logits for the original prompt part as we don't care about them
                # as a hacky way to do that, let's just split on the id of the last <|end_header_id|> token

                import numpy as np

                split_position = np.where(prompt_ids.cpu().numpy() == 128007)[0][-1] + 1
                assert (
                    tokenizer.tokenizer.decode(prompt_ids[split_position - 4 : split_position])
                    == '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
                ), (
                    prompt_ids,
                    tokenizer.tokenizer.decode(prompt_ids),
                    tokenizer.tokenizer.decode(prompt_ids[split_position - 4 : split_position]),
                )
                outputs.append([str(elem) for elem in list(tokenprobs[split_position:].cpu().numpy())])

            # to be able to convert to json and pass over http
            output = [str(elem) for elem in list(tokenprobs[split_position:].cpu().numpy())]
            yield outputs, None, None, None

            # **************************************************************************************

            # # make sure it will generate at least min_length
            # min_length = extra.get('min_tokens_to_generate', 0)
            # if min_length > 0:
            #     within_min_length = (context_length - context_lengths) < min_length
            #     logits[within_min_length, eod_id] = -float('Inf')

            # # make sure it won't sample outside the vocab_size range
            # logits[:, tokenizer.vocab_size :] = -float('Inf')

            # # started indicates whether the current token step passes the context_length, so we make sure not to overwrite the context tokens

            # started = context_lengths <= context_length
            # if extra.get('greedy', False):
            #     prev = torch.argmax(logits, dim=-1).view(-1)
            # else:
            #     logits = logits.float()
            #     logits /= temperature
            #     # handle repetition penality
            #     logits = repetition_penalty(logits, extra.get('repetition_penalty', 1.2), all_generated_indices)
            #     logits = top_k_logits(
            #         logits, top_k=extra.get('top_k', 0), top_p=extra.get('top_p', 0.9), started=started
            #     )
            #     probs = F.softmax(logits, dim=-1)
            #     prev = torch.multinomial(probs, num_samples=1).view(-1)

            # # Clamp the predicted out of vocabulary tokens
            # prev = torch.clamp(prev, max=tokenizer.vocab_size - 1)
            # new_tokens = switch(tokens[:, context_length].view(-1), prev, started)

            # # Replace sampled tokens w/ done token if EOD has already been sampled
            # new_tokens = switch(new_tokens, eod_id, is_done)

            # # post process the inference tokens based on the strategy
            # inference_strategy.post_process(tokens, new_tokens, context_length)

            # # Insert either new predicted or next prompt token
            # tokens[:, context_length] = new_tokens

            # if compute_logprob:
            #     if output_logits is None:
            #         output = F.log_softmax(output[:, :context_length, :], 2)

            #         indices = torch.unsqueeze(tokens[:, 1 : context_length + 1], 2)
            #         output_logits = torch.gather(output, 2, indices).squeeze(2)
            #         all_generated_indices = indices[:, :, 0]
            #         if all_probs:
            #             full_logits = output
            #     else:
            #         output = F.log_softmax(output, 2)
            #         indices = torch.unsqueeze(new_tokens, 1).unsqueeze(2)
            #         new_output_logits = torch.gather(output, 2, indices).squeeze(2)

            #         # TODO(rprenger) we're copying output_logits every time.  Should pre-allocate
            #         output_logits = torch.cat([output_logits, new_output_logits], 1)
            #         all_generated_indices = torch.cat([all_generated_indices, indices[:, :, 0]], 1)
            #         if all_probs:
            #             full_logits = torch.cat([full_logits, output], 1)

            # src = parallel_state.get_pipeline_model_parallel_last_rank()
            # group = parallel_state.get_embedding_group()
            # torch.distributed.broadcast(new_tokens, src, group)

            # #                done_token = (prev == eod_id).byte() & started.byte()
            # done_token = inference_strategy.end_of_generation_condition(
            #     tokens[:, : context_length + 1], prev, eod_id, end_strings
            # )
            # done_token = done_token.byte() & started.byte()

            # just_finished = (done_token & ~is_done).bool()
            # lengths[just_finished.view(-1)] = context_length
            # is_done = is_done | done_token

            # done = torch.all(is_done)
            # src = parallel_state.get_pipeline_model_parallel_last_rank()
            # group = parallel_state.get_pipeline_model_parallel_group()
            # torch.distributed.broadcast(done, src, group)
            # if compute_logprob:
            #     if all_probs:
            #         yield tokens, lengths, output_logits, full_logits
            #     else:
            #         yield tokens, lengths, output_logits, None
            # else:
            #     yield tokens, lengths, None, None

            # else:
            #     if parallel_state.is_pipeline_first_stage():
            #         src = parallel_state.get_pipeline_model_parallel_last_rank()
            #         group = parallel_state.get_embedding_group()
            #         new_tokens = torch.empty_like(tokens[:, context_length])
            #         torch.distributed.broadcast(new_tokens, src, group)
            #         tokens[:, context_length] = new_tokens
            #         yield tokens, None, None, None
            #     else:
            #         yield None, None, None, None

            #     done = torch.cuda.ByteTensor([0])
            #     src = parallel_state.get_pipeline_model_parallel_last_rank()
            #     group = parallel_state.get_pipeline_model_parallel_group()
            #     torch.distributed.broadcast(done, src, group)

            # context_length += 1
            # counter += 1
            # if done:
            #     break


def synced_generate(
    model,
    inference_strategy,
    context_tokens_tensor,
    context_length_tensor,
    tokens_to_generate,
    all_probs,
    temperature,
    top_k=0,
    top_p=0.0,
    greedy=False,
    compute_attention_mask=True,
    compute_logprob=False,
    repetition_penalty=1.2,
    end_strings=[],
    min_tokens_to_generate=0,
    image_list=None,
    **strategy_args,
):
    context_length = context_length_tensor.min().item()
    tokenizer = model.tokenizer

    extra = {
        "top_p": top_p,
        "top_k": top_k,
        "greedy": greedy,
        "repetition_penalty": repetition_penalty,
        "min_tokens_to_generate": min_tokens_to_generate,
    }

    # if input containing neighbors (for Mcore retrieval RETRO model)
    if "neighbors_tokens" in strategy_args:
        extra['neighbors_tokens'] = strategy_args['neighbors_tokens']

    batch_token_iterator = sample_sequence_batch(
        model,
        inference_strategy,
        context_tokens_tensor,
        context_length_tensor,
        tokens_to_generate,
        all_probs,
        compute_attention_mask=compute_attention_mask,
        compute_logprob=compute_logprob,
        temperature=temperature,
        end_strings=end_strings,
        image_list=image_list,
        extra=extra,
    )

    for logits, lengths, output_logits, full_logits in batch_token_iterator:
        context_length += 1

    # that's our logits
    return logits
    # if parallel_state.is_pipeline_last_stage():
    #     src = parallel_state.get_pipeline_model_parallel_last_rank()
    #     group = parallel_state.get_embedding_group()
    #     if compute_logprob:
    #         torch.distributed.broadcast(output_logits, src, group)
    #     if all_probs:
    #         src = parallel_state.get_pipeline_model_parallel_last_rank()
    #         group = parallel_state.get_embedding_group()
    #         torch.distributed.broadcast(full_logits, src, group)

    # else:
    #     if parallel_state.is_pipeline_first_stage():
    #         src = parallel_state.get_pipeline_model_parallel_last_rank()
    #         group = parallel_state.get_embedding_group()

    #         if compute_logprob:
    #             precision = model._trainer.precision
    #             dtype = torch.float32

    #             output_logits = torch.empty(
    #                 tokens.size(0), context_length - 1, dtype=dtype, device=torch.device("cuda")
    #             )
    #             torch.distributed.broadcast(output_logits, src, group)

    #         if all_probs:
    #             src = parallel_state.get_pipeline_model_parallel_last_rank()
    #             group = parallel_state.get_embedding_group()
    #             full_logits = torch.empty(
    #                 tokens.size(0),
    #                 context_length - 1,
    #                 model.padded_vocab_size,
    #                 dtype=dtype,
    #                 device=torch.device("cuda"),
    #             )
    #             torch.distributed.broadcast(full_logits, src, group)
    # if tokens is not None:
    #     return tokens[:, :context_length], output_logits, full_logits


def generate(
    model,
    inputs=None,
    tokens_to_generate=0,
    all_probs=False,
    temperature=1.0,
    add_BOS=False,
    top_k=0,
    top_p=0.0,
    greedy=False,
    compute_attention_mask=True,
    compute_logprob=False,
    repetition_penalty=1.0,
    end_strings=['<|endoftext|>'],
    image_list=None,
    min_tokens_to_generate=0,
    random_seed=None,
    **strategy_args,
):
    """
    Args:
        model (NLPModel): text generative model
        inputs (Union[tuple, List[str]]): if it is a tuple, it is assumed to be (context_tokens_tensor, context_length_tensor). Otherwise it it a list of prompt text strings
        tokens_to_generate (int): The maximum length of the tokens to be generated.
        all_probs (bool): Return the log prob for all the tokens
        temperature (float): sampling temperature
        add_BOS (bool): add the bos token at the begining of the prompt
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (float): If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        greedy (bool):  Whether or not to use sampling ; use greedy decoding otherwise
        repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty
        min_tokens_to_generate (int): The minimum length of the tokens to be generated
        random_seed (int): can set to fix random seed for reproducibility. If None, we do not set random seed, so
            the behavior of generation will depend on whether the seed was set earlier or not.
        strategy_args, the extra arguments are treated as inference strategy arguments
        end_strings, a list of strings to stop generation when they are encountered in the output.

    Returns:
        OutputType: It generates the output in a dictionary type. It has the following keys:

            sentences: List[str], output sentences
            tokens: List[List[str]], output sentences borken into tokens
            logprob: List[Tensor], log prob of generated tokens
            full_logprob: List[Tensor], log prob of all the tokens in the vocab
            token_ids: List[Tensor], output sentence token ids
            offsets: List[List[int]]  # list of tokens start positions in text
    """
    if 'strategy' in strategy_args:
        inference_strategy = strategy_args['strategy']
    else:
        inference_strategy = model_inference_strategy_dispatcher(model, **strategy_args)
    tokenizer = model.tokenizer
    if torch.distributed.get_rank() == get_model_parallel_src_rank():
        if isinstance(inputs, tuple):
            context_tokens_tensor, context_length_tensor = inputs
        else:
            context_tokens_tensor, context_length_tensor = inference_strategy.tokenize_batch(
                inputs, tokens_to_generate, add_BOS
            )

        send_generate_info(
            context_tokens_tensor,
            context_length_tensor,
            tokens_to_generate,
            all_probs,
            compute_logprob,
            temperature,
            top_k,
            top_p,
            greedy,
            repetition_penalty,
            min_tokens_to_generate,
            end_strings,
            random_seed,
        )

        # tokenize neighbors and broadcast (for Mcore retrieval RETRO model)
        if 'neighbors' in strategy_args:
            # tokenize neighbors
            neighbors_tokens_tensor, neighbors_tokens_tensor_shape = inference_strategy.tokenize_neighbors_batch(
                strategy_args['neighbors'], strategy_args['retro_inference']
            )

            # send neighbors tensors to all ranks
            model_parallel_group = parallel_state.get_model_parallel_group()
            src = get_model_parallel_src_rank()
            torch.distributed.broadcast(neighbors_tokens_tensor_shape, src, model_parallel_group)
            torch.distributed.broadcast(neighbors_tokens_tensor, src, model_parallel_group)
        else:
            neighbors_tokens_tensor = None

    else:
        (
            context_length_tensor,
            context_tokens_tensor,
            tokens_to_generate,
            all_probs,
            compute_logprob,
            temperature,
            top_k,
            top_p,
            greedy,
            repetition_penalty,
            min_tokens_to_generate,
            end_strings,
            random_seed,
        ) = receive_generate_info()

        # receive broadcast (for Mcore retrieval RETRO model)
        if 'neighbors' in strategy_args:
            # receive neighbors tensors to all ranks
            model_parallel_group = parallel_state.get_model_parallel_group()
            src = get_model_parallel_src_rank()
            neighbors_tokens_tensor_shape = torch.empty(2, dtype=torch.float32, device=torch.cuda.current_device())
            torch.distributed.broadcast(neighbors_tokens_tensor_shape, src, model_parallel_group)
            neighbors_tokens_tensor = torch.empty(
                neighbors_tokens_tensor_shape[0],
                neighbors_tokens_tensor_shape[1],
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            torch.distributed.broadcast(neighbors_tokens_tensor, src, model_parallel_group)
        else:
            neighbors_tokens_tensor = None

    # add neighbors to strategy_args (for retrieval RETRO model)
    if 'neighbors' in strategy_args:
        strategy_args['neighbors_tokens'] = neighbors_tokens_tensor

    if random_seed is not None:
        seed_everything(random_seed)

    if hasattr(model, 'get_attention_mask_from_fusion') and model.get_attention_mask_from_fusion:
        compute_attention_mask = False

    output = synced_generate(
        model,
        inference_strategy,
        context_tokens_tensor,
        context_length_tensor,
        tokens_to_generate,
        all_probs,
        temperature,
        compute_attention_mask=compute_attention_mask,
        compute_logprob=compute_logprob,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
        repetition_penalty=repetition_penalty,
        end_strings=end_strings,
        min_tokens_to_generate=min_tokens_to_generate,
        image_list=image_list,
        **strategy_args,
    )
    # special_tokens = set()
    # if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is not None:
    #     special_tokens.add(tokenizer.pad_token)
    # if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
    #     special_tokens.add(tokenizer.eos_token)
    # if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token is not None:
    #     special_tokens.add(tokenizer.bos_token)
    # if hasattr(tokenizer, 'cls_token') and tokenizer.cls_token is not None:
    #     special_tokens.add(tokenizer.cls_token)
    # if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
    #     special_tokens.add(tokenizer.unk_token)
    # if hasattr(tokenizer, 'sep_token') and tokenizer.sep_token is not None:
    #     special_tokens.add(tokenizer.sep_token)
    # if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token is not None:
    #     special_tokens.add(tokenizer.mask_token)
    # if output is not None:
    #     decode_tokens, output_logits, full_logits = output
    #     resp_sentences = []
    #     resp_sentences_seg = []

    #     decode_tokens = decode_tokens.cpu().numpy().tolist()
    #     for decode_token in decode_tokens:
    #         sentence = tokenizer.ids_to_text(decode_token)
    #         resp_sentences.append(sentence)
    #         if not isinstance(tokenizer, TabularTokenizer):
    #             words = []
    #             for token in decode_token:
    #                 if not isinstance(token, Iterable):
    #                     token = [token]
    #                 word = tokenizer.ids_to_tokens(token)
    #                 if isinstance(word, Iterable):
    #                     word = word[0]
    #                 if hasattr(tokenizer.tokenizer, 'byte_decoder'):
    #                     word = bytearray([tokenizer.tokenizer.byte_decoder[c] for c in word]).decode(
    #                         'utf-8', errors='replace'
    #                     )
    #                 words.append(word)
    #             resp_sentences_seg.append(words)
    #         else:
    #             words = tokenizer.text_to_tokens(sentence)
    #             resp_sentences_seg.append(words)

    #     # offsets calculation
    #     all_offsets = []
    #     for item in resp_sentences_seg:
    #         offsets = [0]
    #         for index, token in enumerate(item):
    #             if index != len(item) - 1:
    #                 if token in special_tokens:
    #                     offsets.append(offsets[-1])
    #                 else:
    #                     offsets.append(len(token) + offsets[-1])
    #         all_offsets.append(offsets)

    output_dict = {}
    output_dict['sentences'] = output
    output_dict['full_logprob'] = None  # required to be deleted by the server
    # output['tokens'] = resp_sentences_seg
    # output['logprob'] = output_logits
    # output['full_logprob'] = full_logits
    # output['token_ids'] = decode_tokens
    # output['offsets'] = all_offsets
    # output = inference_strategy.post_generation_process(output)
    return output_dict


# monkey-patching the function to change logits calc logic
tgu.sample_sequence_batch = sample_sequence_batch
tgu.generate = generate

# needs to be here to pick up monkey patched functions
from nemo.collections.nlp.modules.common.text_generation_server import MegatronServer

"""
This is the script to run GPT text generation.

Usage:
         python megatron_gpt_eval.py \
            gpt_model_file=PATH_TO_MODEL \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            server=True

        To send a request to the server, here is one example code:
        ```python
        import json
        import requests

        batch_size = 8
        port_num = 5555
        headers = {"Content-Type": "application/json"}


        def request_data(data):
            resp = requests.put('http://localhost:{}/generate'.format(port_num),
                                data=json.dumps(data),
                                headers=headers)
            sentences = resp.json()['sentences']
            return sentences


        data = {
            "sentences": [""] * batch_size,
            "tokens_to_generate": 300,
            "temperature": 1.0,
            "add_BOS": True,
            "top_k": 0,
            "top_p": 0.9,
            "greedy": False,
            "all_probs": False,
            "repetition_penalty": 1.2,
            "min_tokens_to_generate": 2,
        }

        sentences = request_data(data)
        ```
"""

if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


@hydra_runner(config_path=".", config_name="nemo_inference")
def main(cfg) -> None:
    # trainer required for restoring model parallel models
    trainer = Trainer(
        strategy=NLPDDPStrategy(timeout=datetime.timedelta(seconds=18000)),
        **cfg.trainer,
        callbacks=[CustomProgressBar()],
    )

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    save_restore_connector = NLPSaveRestoreConnector()
    if os.path.isdir(cfg.gpt_model_file):
        save_restore_connector.model_extracted_dir = cfg.gpt_model_file

    pretrained_cfg = MegatronGPTModel.restore_from(
        restore_path=cfg.gpt_model_file,
        trainer=trainer,
        return_config=True,
        save_restore_connector=save_restore_connector,
    )
    OmegaConf.set_struct(pretrained_cfg, True)
    with open_dict(pretrained_cfg):
        pretrained_cfg.sequence_parallel = False
        pretrained_cfg.activations_checkpoint_granularity = None
        pretrained_cfg.activations_checkpoint_method = None
        pretrained_cfg.precision = trainer.precision
        pretrained_cfg.tensor_model_parallel_size = cfg.tensor_model_parallel_size
        pretrained_cfg.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
        if trainer.precision == "16":
            pretrained_cfg.megatron_amp_O2 = False
        elif trainer.precision in ['bf16', 'bf16-mixed'] and cfg.get('megatron_amp_O2', False):
            pretrained_cfg.megatron_amp_O2 = True
    model = MegatronGPTModel.restore_from(
        restore_path=cfg.gpt_model_file,
        trainer=trainer,
        override_config_path=pretrained_cfg,
        save_restore_connector=save_restore_connector,
        map_location=f'cuda:{trainer.local_rank}',  # map_location is needed for converted models
    )
    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    inference_strategy = CodeExecutionStrategy(sandbox_cfg=cfg.sandbox, model=model, **cfg.get('code_execution', {}))

    if parallel_state.is_pipeline_first_stage() and parallel_state.get_tensor_model_parallel_rank() == 0:
        server = MegatronServer(model.cuda(), inference_strategy=inference_strategy)
        server.run("0.0.0.0", port=cfg.port)

    while True:
        choice = torch.cuda.LongTensor(1)
        torch.distributed.broadcast(choice, 0)
        if choice[0].item() == 0:
            generate(model.cuda(), strategy=inference_strategy)


if __name__ == '__main__':
    # setting nemo logging to warning as there is too much info otherwise
    logging.getLogger("nemo_logger").setLevel(logging.WARNING)
    main()  # noqa pylint: disable=no-value-for-parameter
