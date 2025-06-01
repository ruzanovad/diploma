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
import torch.multiprocessing as mp


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
        text2int_fn,  # just a function
        word2id: dict,  # plain dict
        sos_id: int,
        eos_id: int,
        num_workers: int = 4,
        batch_size=20,
    ):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.predict_set = predict_set
        self.batch_size = batch_size
        self.text2int_fn = text2int_fn
        self.word2id = word2id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.num_workers = num_workers
        self.persistent_workers = num_workers > 0
        # pin_memory only if GPU will be used
        self.pin_memory = torch.cuda.is_available()
        self.mp_ctx = mp.get_context("spawn")

    def train_dataloader(self):
        """
        Returns the training DataLoader with shuffling and batching.
        """
        return DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            # multiprocessing_context=self.mp_ctx,
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
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            # multiprocessing_context=self.mp_ctx,
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
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            # multiprocessing_context=self.mp_ctx,
        )

    def predict_dataloader(self):
        """
        Returns the prediction DataLoader.
        """
        return DataLoader(
            self.predict_set,
            shuffle=False,
            batch_size=self.batch_size,
            # multiprocessing_context=self.mp_ctx,
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

        # 1) token sequence → LongTensor
        formulas = [self.text2int_fn(formula) for _, formula in batch]
        formula_lengths = torch.LongTensor([f.size(0) for f in formulas])
        formulas_padded = pad_sequence(formulas, batch_first=True)


        # prepend SOS, append EOS
        sos = torch.full((size, 1), self.sos_id, dtype=torch.long)
        eos = torch.full((size, 1), self.eos_id, dtype=torch.long)
        formulas = torch.cat((sos, formulas_padded, eos), dim=1)

        # 2) image padding
        images = [img for img, _ in batch]
        max_h = max(img.size(1) for img in images)
        max_w = max(img.size(2) for img in images)

        def pad_img(img):
            c, h, w = img.size()
            pad = tvt.Pad((0, 0, max_w - w, max_h - h))
            return pad(img)

        images = torch.stack([pad_img(img) for img in images]).to(dtype=torch.float)

        return images, formulas, formula_lengths
