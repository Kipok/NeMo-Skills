from nemo_aligner.data.nlp.datasets import RewardModelDataset


class RegressionRewardModelDataset(RewardModelDataset):
    """This class assumes each line of the dataset file is a dictionary with "text" and "label" field,
    where "text" is a string representing the input prompt, and "label" is a list of float or int values.
    Note that when training the model with multiple datasets which contain different attributes,
    we should set missing attributes to model.regression.loss_mask_val(according to training_rm.yaml)
    in the dataset files so that their losses are masked. At least one attribute should be present for each sample.

    WARNING: It's recommended to preprocess your data in advance to ensure all samples are within self.seq_length.
             Otherwise if all samples in a batch are longer than self.seq_length, you may get NaN loss.
    """

    def __init__(
        self,
        cfg,
        tokenizer,
        name,
        data_prefix,
        documents,
        data,
        seq_length,
        seed,
        drop_last=True,
    ):

        assert cfg.data.data_impl.startswith(
            "json"
        ), f"data.data_impl must be either json or jsonl, but got {cfg.data.data_impl}"

        super().__init__(
            cfg=cfg,
            tokenizer=tokenizer,
            name=name,
            data_prefix=data_prefix,
            documents=documents,
            data=data,
            seq_length=seq_length,
            seed=seed,
            drop_last=drop_last,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns one training sample, its label, and its respective length.
        """

        orig_idx = idx = idx % len(self)
        while True:
            sample = self.data[idx]
            sample_text, sample_length = self.encode(sample["text"])
            sample_label = sample["label"]
            if idx == orig_idx:
                orig_length = sample_length
            if sample_length <= self.seq_length:
                break

            idx = (idx + 1) % len(self)
            if idx == orig_idx:
                raise RuntimeError(f"All samples have length > {self.seq_length}")

        assert isinstance(sample_label, list) and all(
            isinstance(value, (float, int)) for value in sample_label
        ), "label should be a list of float or int values"

        sample_label = [float(value) for value in sample_label]

        label_tensor = torch.tensor(sample_label, dtype=torch.float)

        text_np = np.array(sample_text, dtype=np.int64)
        text_np_pad = np.pad(
            text_np, (0, max(0, self.seq_length - text_np.shape[0])), mode="constant", constant_values=self.eos_id
        )

        text_tensor = torch.tensor(text_np_pad)
        attention_mask, loss_mask, position_ids = _create_ltor_masks_and_position_ids(
            text_tensor,
            self.eos_id,
            self.reset_position_ids,
            self.reset_attention_mask,
            self.eod_mask_loss,
        )

        # Negative index comes when we pad the last batch in MegatronPretrainingBatchSampler
        # We make the loss_mask zero to mask out loss from these samples
        if idx == -1:
            logging.waring("WARNING: Got -1 as item index. Masking loss from this sample")
            loss_mask = torch.zeros_like(loss_mask)

        # Replace current sample (when it exceeds max length) with another sample but mask its loss
        if idx != orig_idx:
            logging.warning(
                f"Sample {orig_idx} in dataset '{self.name}' has length "
                f"{orig_length} > {self.seq_length} "
                f"=> replacing it with sample {idx} and masking its loss"
            )
            loss_mask = torch.zeros_like(loss_mask)

        output = {
            "inputs": text_tensor,
            "lengths": text_np.shape[0],
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "labels": label_tensor,
        }
        return output
