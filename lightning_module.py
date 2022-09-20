"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import math
import random
import re
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import Levenshtein
from pytorch_lightning.utilities import rank_zero_only
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from donut import DonutConfig, DonutModel


class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.get("pretrained_model_name_or_path", False):
            self.model = DonutModel.from_pretrained(
                self.config.pretrained_model_name_or_path,
                input_size=self.config.input_size,
                max_length=self.config.max_length,
                align_long_axis=self.config.align_long_axis,
                ignore_mismatched_sizes=True,
            )
        else:
            self.model = DonutModel(
                config=DonutConfig(
                    input_size=self.config.input_size,
                    max_length=self.config.max_length,
                    align_long_axis=self.config.align_long_axis,
                    # with DonutConfig, the architecture customization is available, e.g.,
                    # encoder_layer=[2,2,14,2], decoder_layer=4, ...
                )
            )

    def training_step(self, batch, batch_idx):
        image_tensors = batch[0]
        decoder_input_ids = batch[1][:, :-1]
        decoder_labels = batch[2][:, 1:]
        loss = self.model(image_tensors, decoder_input_ids, decoder_labels)[0]
        self.log_dict({"train_loss": loss}, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image_tensors, decoder_input_ids, prompt_end_idxs, answers = batch
        decoder_prompts = pad_sequence(
            [input_id[: end_idx + 1] for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)],
            batch_first=True,
        )

        preds = self.model.inference(
            image_tensors=image_tensors,
            prompt_tensors=decoder_prompts,
            return_json=False,
            return_attentions=False,
        )["predictions"]

        scores = list()
        for pred, answer in zip(preds, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.model.decoder.tokenizer.eos_token, "")

            if self.config.get("verbose", False) and len(scores) == 1:
                self.print(f"Prediction: {pred}")
                self.print(f"    Answer: {answer}")
                self.print(f" Normed ED: {scores[0]}")
            scores.append(1 - Levenshtein.ratio(pred, answer))

        return scores

    def validation_epoch_end(self, validation_step_outputs):
        total_count = 0
        total_metric = 0
        for scores in validation_step_outputs:
            total_count += len(scores)
            total_metric += np.sum(scores)

        val_metric = total_metric / total_count
        self.log_dict({"val_metric": val_metric}, sync_dist=True)

    def configure_optimizers(self):
        max_iter = None
        if int(self.config.get("max_epochs", -1)) > 0:
            max_iter = (self.config.max_epochs * self.config.num_training_samples_per_epoch) / (
                self.config.train_batch_sizes[0] * torch.cuda.device_count() * self.config.get("num_nodes", 1)
            )
        if int(self.config.get("max_steps", -1)) > 0:
            max_iter = min(self.config.max_steps, max_iter) if max_iter is not None else self.config.max_steps
        assert max_iter is not None

        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, self.config.warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items["exp_name"] = f"{self.config.get('exp_name', '')}"
        items["exp_version"] = f"{self.config.get('exp_version', '')}"
        return items

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = Path(self.config.result_path) / self.config.exp_name / self.config.exp_version
        self.model.save_pretrained(save_path)
        self.model.decoder.tokenizer.save_pretrained(save_path)


class DonutDataPLModule(pl.LightningDataModule):
    def __init__(self, config, train_dataset=None, val_dataset=None):
        super().__init__()
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.g = torch.Generator()
        self.g.manual_seed(self.config.seed)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            worker_init_fn=self.seed_worker,
            generator=self.g,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            pin_memory=True,
            shuffle=False,
        )

    @staticmethod
    def seed_worker(wordker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

