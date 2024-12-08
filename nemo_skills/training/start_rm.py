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

# copied from https://github.com/NVIDIA/NeMo-Aligner/blob/main/examples/nlp/gpt/train_reward_model.py

from functools import partial

import torch.multiprocessing as mp
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.supervised import SupervisedTrainer
from nemo_aligner.data.nlp.builders import (
    build_dataloader,
    build_train_valid_test_regression_rm_datasets,
    build_train_valid_test_rm_datasets,
)
from nemo_aligner.models.nlp.gpt.reward_model_classes import REWARD_MODEL_CLASS_DICT, RewardModelType
from nemo_aligner.utils.distributed import Timer
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    add_custom_checkpoint_callback,
    extract_optimizer_scheduler_from_ptl_model,
    init_distributed,
    init_using_ptl,
    resolve_and_create_trainer,
    retrieve_custom_trainer_state_dict,
)
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo_skills.training.models.outcome_dataset import custom_collate

"""Script to start Reward Model training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)


def _modify_config(gpt_cfg, cfg, add_cfg_to_tree=False):
    """
    This function modifies the original gpt pre-training config (gpt_cfg) with attributes from the finetuning config (cfg).
    The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
    """
    OmegaConf.set_struct(gpt_cfg, True)
    OmegaConf.resolve(cfg)
    with open_dict(gpt_cfg):
        gpt_cfg.megatron_amp_O2 = cfg.model.get("megatron_amp_O2", False)
        gpt_cfg.micro_batch_size = cfg.model.micro_batch_size
        gpt_cfg.global_batch_size = cfg.model.global_batch_size
        gpt_cfg.sequence_parallel = cfg.model.get("sequence_parallel", False)
        gpt_cfg.activations_checkpoint_granularity = cfg.model.get("activations_checkpoint_granularity", None)
        gpt_cfg.activations_checkpoint_num_layers = cfg.model.get("activations_checkpoint_num_layers", None)
        gpt_cfg.activations_checkpoint_method = cfg.model.get("activations_checkpoint_method", None)
        gpt_cfg.activations_checkpoint_layers_per_pipeline = cfg.model.get(
            "activations_checkpoint_layers_per_pipeline", None
        )
        gpt_cfg.data = cfg.model.data
        gpt_cfg.optim = cfg.model.optim
        gpt_cfg.precision = cfg.trainer.precision
        gpt_cfg.restore_from_path = cfg.model.restore_from_path
        gpt_cfg.resume_from_checkpoint = cfg.model.resume_from_checkpoint
        gpt_cfg.use_flash_attention = cfg.model.get("use_flash_attention", False)
        # if TP/PP size is -1, use default TP/PP size as original model
        if cfg.model.get("tensor_model_parallel_size", 1) > 0:
            gpt_cfg.tensor_model_parallel_size = cfg.model.get("tensor_model_parallel_size", 1)
        if cfg.model.get("pipeline_model_parallel_size", 1) > 0:
            gpt_cfg.pipeline_model_parallel_size = cfg.model.get("pipeline_model_parallel_size", 1)
        gpt_cfg.pipeline_model_parallel_split_rank = cfg.model.get("pipeline_model_parallel_split_rank", 0)

        if cfg.model.get("use_flash_attention", None) is not None:
            gpt_cfg.use_flash_attention = cfg.model.use_flash_attention

        # This is needed when modifying a hparam file directly to load `.ckpt` files.
        # This is not needed to modify the cfg in `.nemo` files.
        if add_cfg_to_tree:
            OmegaConf.resolve(gpt_cfg)
            gpt_cfg.cfg = gpt_cfg

        # OVERRRIDE: set the dist_ckpt_format to zarr explicitly unless specified in the config
        gpt_cfg.dist_ckpt_format = cfg.model.get("dist_ckpt_format", "zarr")
    return gpt_cfg


@hydra_runner(config_path=".", config_name="training_rm")
def main(cfg) -> None:
    """
    Binary ranking reward models use comparison based objective similar to the one found in the
    InstructGPT paper: https://arxiv.org/pdf/2203.02155.pdf and have no explicit labels.
    Regression reward models use a MSE loss to fit multi-attribute numeric labels for each data point.
    """

    reward_model_type = RewardModelType(cfg.model.get("reward_model_type", "binary_ranking"))
    reward_model_cls = REWARD_MODEL_CLASS_DICT[reward_model_type]

    # cfg.model = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = resolve_and_create_trainer(cfg, "rm")
    exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    # ptl_model = load_from_nemo(
    #     reward_model_cls,
    #     cfg,
    #     # cfg.model,
    #     trainer,
    #     strict=True,
    #     load_base_model_only=True,
    #     restore_path=cfg.pretrained_checkpoint.restore_from_path,
    # )
    ptl_model = load_from_nemo(
        reward_model_cls,
        cfg,
        trainer,
        strict=True,
        modify_config_fn=_modify_config,
        restore_path=cfg.model.restore_from_path,
        # return_updated_cfg=True,
    )

    # pull values from checkpoint
    trainer_restore_path = trainer.ckpt_path
    if trainer_restore_path is not None:
        custom_trainer_state_dict = retrieve_custom_trainer_state_dict(trainer)
        consumed_samples = custom_trainer_state_dict["consumed_samples"]
    else:
        custom_trainer_state_dict = None
        consumed_samples = 0

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    # use the entire dataset
    train_valid_test_num_samples = [-1 * cfg.model.global_batch_size] * 3

    if reward_model_type == RewardModelType.BINARY_RANKING:
        dataset_builder = build_train_valid_test_rm_datasets
    elif reward_model_type == RewardModelType.REGRESSION:
        dataset_builder = build_train_valid_test_regression_rm_datasets
    else:
        raise ValueError(f"Only support binary_ranking and regression reward model, but get {reward_model_type} ")

    train_ds, validation_ds, _ = dataset_builder(
        cfg=cfg.model,
        data_prefix=cfg.model.data.data_prefix,
        data_impl=cfg.model.data.data_impl,
        splits_string=cfg.model.data.splits_string,
        train_valid_test_num_samples=train_valid_test_num_samples,
        seq_length=cfg.model.data.seq_length,
        seed=cfg.model.seed,
        tokenizer=ptl_model.tokenizer,
    )

    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=consumed_samples,
        mbs=cfg.model.micro_batch_size,
        gbs=cfg.model.global_batch_size,
        load_gbs=True,
        use_random_sampler=True,
        # collate_fn=partial(
        #     custom_collate,
        #     eos_id=ptl_model.tokenizer.eos_id,
        #     reset_position_ids=cfg.model.data.get("reset_position_ids", False),
        #     reset_attention_mask=cfg.model.data.get("reset_attention_mask", False),
        #     eod_mask_loss=cfg.model.data.get("eod_mask_loss", False),
        # ),
    )

    val_dataloader = build_dataloader(
        cfg=cfg,
        dataset=validation_ds,
        consumed_samples=0,
        mbs=cfg.model.micro_batch_size,
        gbs=cfg.model.global_batch_size,
        load_gbs=True,
        use_random_sampler=False,
    )

    init_using_ptl(trainer, ptl_model, train_dataloader, train_ds)
    optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(ptl_model)

    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)

    logger.log_hyperparams(OmegaConf.to_container(cfg))

    timer = Timer(cfg.exp_manager.get("max_time_per_run"))

    rm_trainer = SupervisedTrainer(
        cfg=cfg.trainer.rm,
        model=ptl_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=None,
        logger=logger,
        ckpt_callback=ckpt_callback,
        run_timer=timer,
    )

    if custom_trainer_state_dict is not None:
        rm_trainer.load_state_dict(custom_trainer_state_dict)

    rm_trainer.fit()


if __name__ == "__main__":
    main()
