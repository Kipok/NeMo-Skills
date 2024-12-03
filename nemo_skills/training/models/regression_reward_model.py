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


import torch
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo_aligner.models.nlp.gpt.megatron_gpt_reward_model import MegatronGPTRewardModel
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.train_utils import set_sync_funcs
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer


class RegressionRewardModel(MegatronGPTRewardModel):
    """
    Megatron GPT Regression Reward Model Training.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

    def get_forward_output_and_loss_func(self, validation_step=False):
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
                    required_keys.update(("inputs", "position_ids"))

                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(("labels", "lengths", "loss_mask"))

            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            forward_args = {
                "input_ids": batch["inputs"],
                "lengths": batch["lengths"],
                "position_ids": batch["position_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": None,
            }

            output_tensor = model(**forward_args)

            # in this nemo version the model and autocast dtypes are not synced
            # so we need to explicitly cast it
            if not parallel_state.is_pipeline_last_stage():
                output_tensor = output_tensor.to(dtype=self.autocast_dtype)

            def loss_func(output_tensor):
                # Loss per micro batch (ub).
                loss_for_ub = self.loss_func(output_tensor, batch["labels"])
                if validation_step and not self.cfg.data.get("validation_drop_last", True):
                    num_valid_tokens_in_ub = batch["loss_mask"].sum()
                    if loss_for_ub.isnan():
                        assert batch["loss_mask"].count_nonzero() == 0, "Got NaN loss with non-empty input"
                        loss_sum_for_ub = torch.zeros_like(num_valid_tokens_in_ub)
                    else:
                        loss_sum_for_ub = num_valid_tokens_in_ub * loss_for_ub

                    loss_sum_and_ub_size_all_gpu = torch.cat(
                        [
                            loss_sum_for_ub.clone().detach().view(1),
                            torch.tensor([num_valid_tokens_in_ub]).cuda().clone().detach(),
                        ]
                    )
                    torch.distributed.all_reduce(
                        loss_sum_and_ub_size_all_gpu, group=parallel_state.get_data_parallel_group()
                    )

                    return (
                        loss_for_ub,
                        {
                            "loss_sum_and_ub_size": loss_sum_and_ub_size_all_gpu,
                        },
                    )
                else:
                    reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])

                    return (
                        loss_for_ub,
                        {
                            "avg": reduced_loss,
                        },
                    )

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def loss_func(self, output_tensor, label_tensor):
        loss_func_name = self.cfg.get("loss_func", "regression")
        assert loss_func_name in ["regression", "regular_bt", "margin_bt", "scaled_bt"]

        if loss_func_name == "regression":
            return self.reg_loss_func(output_tensor, label_tensor)
        elif loss_func_name.endswith("bt"):
            return self.bt_loss_func(output_tensor, label_tensor)
        else:
            raise ValueError("only accepted values for loss_func are regression, regular_bt, margin_bt and scaled_bt")

    def bt_loss_func(self, output_tensor, label_tensor):
        """
        label_tensor is a tensor of shape [2, n_attributes] and reflects the value of 'label' in the dataset file of two adjacent samples.

        An example is [
            [1, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0],
            [1, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0]
        ]

        label_tensor should have the value of the [0,0] index as a number indicating the prefered response as well as the strength of the preferred response

        In HelpSteer2-Preference, label is a non-negative integer between -3 and 3: -3, -2, -1 means A is preferred while 1, 2, 3 means B is preferred

        output_tensor uses the value of [:, 4] index as the reward for the chosen/rejected values, which inherits the position of the helpfulness attribute
        """
        label = label_tensor[0, 0]

        aspect_importance = output_tensor[:, 4]
        out_first = aspect_importance[0]
        out_second = aspect_importance[1]

        margin = abs(label)

        label_item = label.item()
        if label_item < 0:
            out_chosen, out_rejected = out_first, out_second
        # else include the zero case as well
        else:
            out_chosen, out_rejected = out_second, out_first

        if self.cfg.loss_func == "regular_bt":
            loss = -torch.nn.functional.logsigmoid(out_chosen - out_rejected).mean()
        elif self.cfg.loss_func == "margin_bt":
            loss = -torch.nn.functional.logsigmoid(out_chosen - out_rejected - margin).mean()
        elif self.cfg.loss_func == "scaled_bt":
            loss = margin * -torch.nn.functional.logsigmoid(out_chosen - out_rejected).mean()
        return loss

    def reg_loss_func(self, output_tensor, label_tensor):
        mask_val = self.cfg.get("loss_mask_val", -100.0)
        mask = label_tensor != mask_val
        num_valid_attributes = mask.float().sum()
        assert num_valid_attributes > 0, "Invalid sample: all attributes in label are masked, please check your data!"
        # Calculate the squared difference between prediction and label, and use the mask to ignore specific losses
        squared_diff = (output_tensor - label_tensor) ** 2 * mask
        # Calculate the mean of the masked squared differences
        loss = squared_diff.sum() / num_valid_attributes
        return loss

    def get_loss_and_metrics(self, batch, forward_only):
        data_iter = get_iterator_k_split(batch, get_num_microbatches())
        set_sync_funcs(self, forward_only)

        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(forward_only),
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=self.cfg.encoder_seq_length,
            micro_batch_size=self.cfg.micro_batch_size,
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # NOTE: assume that the returned values are already gathered across the DP workers
            # average loss across micro batches
            loss_tensors_list = [loss_reduced["avg"] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.concat(loss_tensors_list)
            loss_mean = loss_tensor.mean()

        else:
            loss_mean = torch.tensor(0.0, device=torch.cuda.current_device())

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(loss_mean, get_last_rank())

        metrics = {
            "loss": loss_mean,
        }

        # move to CPU
        metrics = {k: v.item() for k, v in metrics.items()}

        return loss_mean.item(), metrics
