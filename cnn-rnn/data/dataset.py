"""
Dataset definitions for the im2latex project.

This module provides dataset classes for loading and preprocessing image-formula pairs and
prediction images.
It includes:
    - LatexDataset: Loads images and corresponding LaTeX formulas from CSV files for training,
    validation, and testing.
    - LatexPredictDataset: Loads images for prediction/inference.

Both datasets handle image normalization and are compatible with PyTorch DataLoader.
"""

import os
import math

from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
import torchvision
from torchvision import transforms as tvt


class LatexDataset(Dataset):
    """
    Dataset class for the im2latex project
    """

    def __init__(
        self, data_path, img_path, data_type: str, n_sample: int = None, dataset="100k"
    ):
        super().__init__()
        assert data_type in ["train", "test", "validate"], "Not found data type"
        csv_path = data_path + f"/im2latex_{data_type}.csv"
        df = pd.read_csv(csv_path)
        if n_sample:
            df = df.head(n_sample)
        df["image"] = df.image.map(lambda x: img_path + "/" + x)
        self.walker = df.to_dict("records")
        self.transform = tvt.Compose(
            [
                tvt.Normalize((0.5), (0.5)),
            ]
        )

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        item = self.walker[idx]

        formula = item["formula"]
        image = torchvision.io.read_image(item["image"]).cpu()
        image = image.to(dtype=torch.float)
        image /= image.max()
        image = self.transform(image)  # transform image to [-1, 1]
        return image, formula


class LatexPredictDataset(Dataset):
    """
    A PyTorch Dataset for predicting LaTeX from a single image.

    Args:
        predict_img_path (str): Path to the image file to be used for prediction.

    Attributes:
        walker (list): List containing the image path if provided, otherwise empty.
        transform (torchvision.transforms.Compose): Transformation pipeline to normalize the image.

    Methods:
        __len__(): Returns the number of images in the dataset (0 or 1).
        __getitem__(idx): Loads and returns the normalized image tensor at the given index.

    Raises:
        AssertionError: If the provided image path does not exist.
    """

    def __init__(self, predict_img_path: str):
        super().__init__()
        if predict_img_path:
            assert os.path.exists(predict_img_path), "Image not found"
            self.walker = [predict_img_path]
        else:
            self.walker = []
        self.transform = tvt.Compose(
            [
                tvt.Normalize((0.5), (0.5)),
            ]
        )

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        img_path = self.walker[idx]

        image = torchvision.io.read_image(img_path).cpu().to(dtype=torch.float)
        image = image.to(dtype=torch.float)
        max_val = image.max()
        if max_val > 0:
            image = image / max_val
        image = self.transform(image)  # transform image to [-1, 1]

        return image
