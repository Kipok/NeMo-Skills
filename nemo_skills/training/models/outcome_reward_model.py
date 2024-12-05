# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import warnings
from functools import partial
from typing import List, Tuple, Union

import torch
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.utils import divide
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel, get_specs
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
    get_ltor_masks_and_position_ids,
)
from nemo.collections.nlp.parts.mixins.nlp_adapter_mixins import NLPAdapterModelMixin
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils.dtype import str_to_dtype
from nemo_aligner.models.alignable_interface import SupervisedInterface
from nemo_aligner.models.nlp.gpt.gpt_reward_model import GPTRewardModel
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import (
    broadcast_2d_tensor,
    broadcast_2d_tensor_within_pp,
    from_parallel_logits_to_logprobs,
)
from nemo_aligner.utils.text_generation_utils import tokenize_batch
from nemo_aligner.utils.train_utils import (
    finish_validation_step,
    grad_reductions,
    prepare_for_training_step,
    prepare_for_validation_step,
    set_sync_funcs,
)
from nemo_aligner.utils.utils import adapter_control, cpu_weight_swap
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer


class OutcomeRewardModel(MegatronGPTModel, SupervisedInterface):

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        if self.cfg.pipeline_model_parallel_size > 1 and not self.cfg.megatron_amp_O2:
            warnings.warn(
                "when using pipeline parallelism, it is recommended to set megatron_amp_O2 to be True to "
                "avoid explicit casting for pipeline communication"
            )
        self.automatic_optimization = False
        self.ref_policy_state_dict = None

        self.preference_avg_log_probs = self.cfg.kto.get("preference_average_log_probs", False)
        self.sft_avg_log_probs = self.cfg.kto.get("sft_average_log_probs", self.preference_avg_log_probs)

        self.preference_loss_weight = self.cfg.kto.get("preference_loss_weight", 1)
        self.sft_loss_weight = self.cfg.kto.get("sft_loss_weight", 0)
        assert (
            self.preference_loss_weight != 0 or self.sft_loss_weight != 0
        ), "sft loss weight and preference loss weight cannot both be 0"

        self.desirable_loss_weight = self.cfg.kto.get("desirable_loss_weight", 1.0)
        self.undesirable_loss_weight = self.cfg.kto.get("undesirable_loss_weight", 1.0)

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""

        force_head_dtype = self.cfg.get("force_head_dtype", torch.float32)
        head_dtype = None if force_head_dtype is None else str_to_dtype(force_head_dtype)

        model = GPTRewardModel(
            config=self.transformer_config,
            transformer_layer_spec=get_specs(self.spec_name, self.transformer_config),
            vocab_size=self.cfg.get("override_vocab_size", self.padded_vocab_size),
            max_sequence_length=self.cfg.get("encoder_seq_length", 512),
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=True,
            share_embeddings_and_output_weights=False,
            position_embedding_type=self.cfg.get("position_embedding_type", "learned_absolute"),
            rotary_percent=self.cfg.get("rotary_percentage", 1.0),
            seq_len_interpolation_factor=self.cfg.get("seq_len_interpolation_factor", None),
            rotary_base=self.cfg.get("rotary_base", 10000),
            output_sequence=self.cfg.get("output_sequence", False),
            use_avg_pool=self.cfg.get("use_avg_pool", False),
            head_dtype=head_dtype,
            num_attributes=self.cfg.get("regression", {}).get("num_attributes", 1),
            attribute_weights=self.cfg.get("regression", {}).get("attribute_weights", None),
            merge_attributes=self.cfg.get("regression", {}).get("merge_attributes", False),
        )
        return model

    def on_load_checkpoint(self, checkpoint) -> None:
        """NOTE: Have to set strict to False because we have a rm head"""
        # mcore uses distributed checkpointing
        # FSDP supports the lagecy checkpointing or torch-FSDP-native sharded checkpointing
        if not self.use_fsdp:
            if "state_dict" in checkpoint and checkpoint["state_dict"]:
                for index, module in enumerate(self.get_model_module_list()):
                    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                        checkpoint_state_dict = checkpoint["state_dict"][f"model_{index}"]
                    else:
                        checkpoint_state_dict = checkpoint["state_dict"]
                    # checkpoint_state_dict has "model." but module does not so we need to remove it when loading
                    checkpoint_state_dict = {
                        key.replace("model.", ""): checkpoint_state_dict.pop(key)
                        for key in list(checkpoint_state_dict.keys())
                    }
                    module.load_state_dict(checkpoint_state_dict, strict=False)
            else:
                # when restoring a distributed checkpoint from a ptl checkpoint we need to defer loading the state_dict
                # see NLPModel.on_load_checkpoint
                checkpoint["state_dict"] = {}

    def infer(
        self,
        inputs: Union[List[str], Tuple[torch.Tensor, torch.Tensor]],
        add_BOS: bool = False,
        add_EOS: bool = False,
    ):
        if isinstance(inputs, tuple):
            context_tokens_tensor, context_length_tensor = inputs
        elif isinstance(inputs, list):
            assert all(isinstance(item, str) for item in inputs), "list must contain all strings in infer function"
            context_tokens_tensor, context_length_tensor = tokenize_batch(
                inputs,
                self.tokenizer,
                self.cfg.encoder_seq_length,
                add_BOS=add_BOS,
                add_EOS=add_EOS,
            )
        else:
            raise NotImplementedError(f"{type(inputs)=} is not supported in infer function")

        context_tokens_tensor = context_tokens_tensor.cuda()
        context_length_tensor = context_length_tensor.cuda()

        inference_batch_size, sequence_length = context_tokens_tensor.size()
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            context_tokens_tensor,
            self.tokenizer.eos_id,
            self.cfg.get("reset_position_ids", False),
            self.cfg.get("reset_attention_mask", False),
            self.cfg.get("eod_mask_loss", False),
        )
        attention_mask = attention_mask.expand(inference_batch_size, -1, -1, -1)
        inputs = [context_tokens_tensor, context_length_tensor, position_ids, attention_mask]

        # if inference batch size is smaller than forward mbs run it at the lower batch size
        forward_micro_batch_size = min(inference_batch_size, self.forward_micro_batch_size)

        num_microbatches = divide(inference_batch_size, forward_micro_batch_size)
        data_iter = get_iterator_k_split(inputs, num_microbatches)

        rewards = self.forward_step(data_iter, forward_micro_batch_size, sequence_length, num_microbatches)

        if parallel_state.is_pipeline_last_stage():
            rewards = torch.cat(rewards)

            # Standardize values to subtract a bias.
            if self.enable_standardization:
                rewards = (rewards - self.rew_mean) / self.rew_std

        rewards = broadcast_2d_tensor_within_pp(rewards)
        return rewards

    @torch.no_grad()
    def gather_and_split_rewards(self, pi_logprobs, labels, preferences, average_log_probs=False):
        pi_logprobs = pi_logprobs.detach()

        dp_group = parallel_state.get_data_parallel_group()

        batch_logs = self.get_reduced_masked_logps(pi_logprobs, labels[:, 1:], average_log_probs=average_log_probs)

        output_list = [torch.zeros_like(batch_logs) for _ in range(dp_group.size())]

        torch.distributed.all_gather(output_list, batch_logs, group=dp_group)

        # split_iter = map(self.split_output_tensor, output_list)
        split_iter = map(partial(self.split_output_tensor, preferences=preferences), output_list)

        out_chosen, out_rejected = map(torch.cat, zip(*split_iter))

        return out_chosen.flatten(), out_rejected.flatten()

    def get_forward_output_and_loss_func(self, validation_step=False, logprobs_only=False):
        def fwd_output_and_loss_func(dataloader_iter, model):
            batch = next(dataloader_iter)

            required_keys = set()
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                # there is a problem with apex ignoring the mask on the older models
                # so we will always give the attention mask
                required_keys.add("attention_mask")

                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(("samples", "position_ids"))

                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(
                        (
                            "sample_labels",
                            "preference",
                        )
                    )

            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            tokens, labels = None, None, None
            if batch["samples"] is not None:
                tokens = batch["samples"]

            if batch["sample_labels"] is not None:
                labels = batch["sample_labels"]

            if batch["preference"] is not None:
                preferences = batch["preference"]

            # this is necessary if MBS > 1 with the new GBS padding logic, as you may get batch dim > 1 in some configs
            # these two lines ensure your position_ids and attn_mask are always B=1
            # position_ids = batch["position_ids"][0:1]
            attention_mask = batch["attention_mask"][0:1]

            # Model forward pass
            forward_args = {
                "input_ids": tokens,
                "position_ids": batch["position_ids"],
                "attention_mask": attention_mask,
                "labels": None,
            }

            output_tensor = model(**forward_args)

            # in this nemo version the model and autocast dtypes are not synced
            # so we need to explicitly cast it
            if not parallel_state.is_pipeline_last_stage():
                output_tensor = output_tensor.to(dtype=self.autocast_dtype)

            def logprobs_func(output_tensor, non_loss_data=True):
                # This function is expected to be used only when `collect_non_loss_data=True` in the fwd_bwd_function of Megatron-LM.
                # See https://github.com/NVIDIA/Megatron-LM/blob/0bc3547702464501feefeb5523b7a17e591b21fa/megatron/core/pipeline_parallel/schedules.py#L228
                assert non_loss_data
                logprobs = from_parallel_logits_to_logprobs(
                    vocab_parallel_logits=output_tensor,
                    target=labels,
                    inference_only=True,
                    higher_stability=True,
                )
                return {"logprobs": logprobs}

            def loss_func(output_tensor):
                if validation_step and not self.cfg.data.get("validation_drop_last", True):
                    raise NotImplementedError("KTO does not support validation when cfg.data.drop_last=False")

                per_token_logps = from_parallel_logits_to_logprobs(
                    vocab_parallel_logits=output_tensor,
                    target=labels,
                    inference_only=validation_step,
                    higher_stability=True,
                )

                loss = self.loss_func(
                    per_token_logps,
                    labels[:, 1:],
                    preferences,
                    average_log_probs=self.preference_avg_log_probs,
                )

                reduced_loss = average_losses_across_data_parallel_group([loss])

                out_chosen, out_rejected = self.gather_and_split_rewards(
                    per_token_logps, labels, preferences, average_log_probs=self.preference_avg_log_probs
                )

                return (
                    loss,
                    {
                        "avg": reduced_loss,
                        "out_chosen": out_chosen,
                        "out_rejected": out_rejected,
                    },
                )

            if logprobs_only:
                return output_tensor, logprobs_func
            else:
                return output_tensor, loss_func

        return fwd_output_and_loss_func

    def split_output_tensor(self, output_tensor, preferences=None):
        logps = output_tensor
        if preferences is None:
            # this is for the sample_logprobs
            return logps
        else:
            chosen_idx = torch.where(preferences == 1)[0]
            rejected_idx = torch.where(preferences == 0)[0]
            chosen_logps = logps[chosen_idx, ...]
            reject_logps = logps[rejected_idx, ...]

            return chosen_logps, reject_logps

    def get_reduced_masked_logps(self, logps, labels, average_log_probs=False):
        assert logps.shape == labels.shape, "logps and labels shape mismatch"

        loss_mask = (labels > -1).float()

        if average_log_probs:
            # need to guard against divide by zero in case labels are all -100
            return (logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
        else:
            return (logps * loss_mask).sum(-1)

    def loss_func(self, pi_logprobs, labels, preferences, average_log_probs=False):
        rewards = self.get_reduced_masked_logps(pi_logprobs, labels, average_log_probs=average_log_probs)
        chosen_rewards, reject_rewards = self.split_output_tensor(rewards, preferences)
        if chosen_rewards.shape[0] != 0:
            chosen_losses = 1.0 - torch.nn.functional.sigmoid(chosen_rewards)
        else:
            chosen_losses = torch.Tensor([]).to(rewards.dtype).to(rewards.device)

        if reject_rewards.shape[0] != 0:
            reject_losses = 1.0 - torch.nn.functional.sigmoid(-reject_rewards)
        else:
            reject_losses = torch.Tensor([]).to(rewards.dtype).to(rewards.device)

        loss = torch.cat(
            (self.desirable_loss_weight * chosen_losses, self.undesirable_loss_weight * reject_losses), dim=0
        )

        return loss.mean()

    def get_loss_and_metrics(self, batch, forward_only):
        seq_length = batch["samples"].shape[1]

        data_iter = get_iterator_k_split(batch, get_num_microbatches())
        set_sync_funcs(self, forward_only)

        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(forward_only, logprobs_only=False),
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=seq_length,
            micro_batch_size=self.cfg.micro_batch_size,
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # NOTE: assume that the returned values are already gathered across the DP workers
            rewards_chosen = torch.cat([item["out_chosen"] for item in losses_reduced_per_micro_batch])
            rewards_rejected = torch.cat([item["out_rejected"] for item in losses_reduced_per_micro_batch])

            rewards_all = torch.cat((rewards_chosen, rewards_rejected))
            rewards_chosen_mean = rewards_chosen.mean()
            rewards_rejected_mean = rewards_rejected.mean()
            rewards_all_mean = rewards_all.mean()
            rewards_all_std = rewards_all.std()

            # average loss across micro batches
            loss_mean = torch.as_tensor(
                [loss_reduced["avg"] for loss_reduced in losses_reduced_per_micro_batch],
                device=torch.cuda.current_device(),
            ).mean()
        else:

            loss_mean = torch.tensor(0.0, device=torch.cuda.current_device())

            rewards_chosen_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            rewards_rejected_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            rewards_all_mean = torch.tensor(0.0, device=torch.cuda.current_device())
            rewards_all_std = torch.tensor(0.0, device=torch.cuda.current_device())

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(loss_mean, get_last_rank())
        torch.distributed.broadcast(rewards_chosen_mean, get_last_rank())
        torch.distributed.broadcast(rewards_rejected_mean, get_last_rank())
        torch.distributed.broadcast(rewards_all_mean, get_last_rank())
        torch.distributed.broadcast(rewards_all_std, get_last_rank())

        metrics = {
            "loss": loss_mean,
            "rewards_chosen_mean": rewards_chosen_mean,
            "rewards_rejected_mean": rewards_rejected_mean,
            "rewards_all_mean": rewards_all_mean,
            "rewards_all_std": rewards_all_std,
        }

        # move to CPU
        metrics = {k: v.item() for k, v in metrics.items()}
        return loss_mean.item(), metrics

    def prepare_for_training_step(self):
        # custom trainers will always zero grad for us
        prepare_for_training_step(self, zero_grad=False)

    def finish_training_step(self):
        grad_reductions(self)

    def prepare_for_validation_step(self):
        prepare_for_validation_step(self)

    def finish_validation_step(self):
        finish_validation_step(self)

    @torch.no_grad()
    def _get_logprob_batch(self, batch):
        seq_length = batch["samples"].shape[1]
        batch_size = batch["samples"].shape[0]

        # Differently from DPO, KTO does not require to split the batch into microbatches as there is only one response per prompt
        # We use min to guard against user providing too high forward mbs
        num_microbatches = divide(batch_size, min(batch_size, self.cfg.kto.log_prob_forward_micro_batch_size))
        data_iter = get_iterator_k_split(batch, num_microbatches)
        set_sync_funcs(self, forward_only=True)

        fwd_bwd_function = get_forward_backward_func()

        logprobs_list = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(logprobs_only=True),
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=num_microbatches,
            forward_only=True,
            seq_length=seq_length,
            micro_batch_size=min(batch_size, self.cfg.kto.log_prob_forward_micro_batch_size),
            collect_non_loss_data=True,
        )

        if len(logprobs_list) > 0:
            sample_logprobs_list = []
            for item in logprobs_list:
                sample_logprobs = self.split_output_tensor(item["logprobs"])
                sample_logprobs_list.append(sample_logprobs)

            logprobs = torch.cat(sample_logprobs_list)
        else:
            logprobs = None

        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            # broadcast it from last PP stage to everything else
            logprobs = broadcast_2d_tensor(
                logprobs,
                parallel_state.get_pipeline_model_parallel_last_rank(),
                parallel_state.get_pipeline_model_parallel_group(),
            )

        return logprobs
