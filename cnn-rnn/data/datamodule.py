"""
DataModule for im2latex: PyTorch Lightning DataModule for managing data loading and preprocessing.

This module defines the DataModule class, which encapsulates all logic for loading, batching, and
preprocessing data for the im2latex project. It supports training, validation, testing, and
prediction phases, handling both image and formula data. The module includes custom collation
to pad images andtokenized formulas to uniform sizes within each batch.

Classes:
    DataModule: Handles dataset splits, DataLoader creation, and
    batch collation for images and formulas.

Typical usage example:
    dm = DataModule(train_set, val_set, test_set, predict_set, batch_size=32, text=text_encoder)
    trainer.fit(model, datamodule=dm)
"""

from typing import List, Tuple
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms as tvt


class DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the im2latex project.

    This module handles all data loading and preprocessing logic for training,
    validation, testing, and prediction phases. It expects datasets that return
    tuples of (image_tensor, formula_string).

    Args:
        train_set (Dataset): Training dataset.
        val_set (Dataset): Validation dataset.
        test_set (Dataset): Test dataset.
        predict_set (Dataset): Prediction dataset.
        num_workers (int): Number of workers for data loading (default=4).
        batch_size (int): Batch size (default=20).
        text (TextEncoder): Helper object for converting formulas to integer token IDs.
    """

    def __init__(
        self,
        train_set,
        val_set,
        test_set,
        predict_set,
        num_workers: int = 4,
        batch_size=20,
        text=None,
    ):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.predict_set = predict_set
        self.batch_size = batch_size
        self.text = text
        self.num_workers = num_workers
        self.persistent_workers = False

    def train_dataloader(self):
        """
        Returns the training DataLoader with shuffling and batching.
        """
        return DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        """
        Returns the validation DataLoader without shuffling.
        """
        return DataLoader(
            self.val_set,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        """
        Returns the test DataLoader without shuffling.
        """
        return DataLoader(
            self.test_set,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        """
        Returns the prediction DataLoader.
        """
        return DataLoader(
            self.predict_set,
            shuffle=False,
            batch_size=self.batch_size,
        )

    def collate_fn(self, batch: List[Tuple[torch.Tensor, str]]):
        """
        Custom collate function for batching samples.

        Each sample is a tuple (image_tensor, formula_string).
        This function:
            - Pads tokenized formulas to the same length with SOS/EOS tokens.
            - Pads images to the same spatial dimensions (max height and width in batch).
            - Returns a batch of padded images and formulas.

        Args:
            batch (List[Tuple[Tensor, str]]): List of samples from the dataset.

        Returns:
            images (Tensor): Batched and padded image tensors, shape (B, C, H, W).
            formulas (Tensor): Batched tokenized formulas with <s> and <e>, shape (B, L).
            formula_len (Tensor): Lengths of original formulas (before padding), shape (B,).
        """
        size = len(batch)
        formulas = [self.text.text2int(i[1]) for i in batch]
        formula_len = torch.LongTensor([i.size(-1) + 1 for i in formulas])
        formulas = pad_sequence(formulas, batch_first=True)

        sos = torch.full((size, 1), self.text.word2id["<s>"], dtype=torch.long)
        eos = torch.full((size, 1), self.text.word2id["<e>"], dtype=torch.long)

        formulas = torch.cat((sos, formulas, eos), dim=-1).to(dtype=torch.long)

        images = [i[0] for i in batch]
        max_width, max_height = 0, 0
        for img in images:
            _, h, w = img.size()
            max_width = max(max_width, w)
            max_height = max(max_height, h)

        def padding(img):
            _, h, w = img.size()
            padder = tvt.Pad((0, 0, max_width - w, max_height - h))
            return padder(img)

        images = torch.stack(list(map(padding, images))).to(dtype=torch.float)
        return images, formulas, formula_len
