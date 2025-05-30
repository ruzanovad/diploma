"""
Defines the Image2LatexModel LightningModule for the im2latex project.

This module implements the PyTorch LightningModule wrapper for the Image2Latex model,
handling training, validation, testing, and prediction steps. It integrates loss computation,
optimizer and scheduler configuration, and evaluation metrics such as edit distance, BLEU-4,
and exact match. The module is designed for end-to-end training and evaluation of the
image-to-LaTeX sequence model.

Classes:
    Image2LatexModel: PyTorch LightningModule for training and evaluating the im2latex model.
"""

import torch
from torch import nn
from torchaudio.functional import edit_distance
from evaluate import load
import lightning.pytorch as pl
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from .im2latex import Image2Latex
from .text import Text


class Image2LatexModel(pl.LightningModule):
    """
    PyTorch Lightning Module for Image-to-LaTeX Model.

    This module encapsulates the training, validation, testing, and
    prediction logic for an image-to-LaTeX sequence model. It leverages
    PyTorch Lightning for streamlined training and evaluation, and supports
    various encoder/decoder configurations.

    Args:
        lr (float): Learning rate for the optimizer.
        total_steps (int): Total number of training steps for learning rate scheduling.
        n_class (int): Number of output classes (vocabulary size).
        enc_dim (int, optional): Encoder output dimension. Default is 512.
        enc_type (str, optional): Encoder type (e.g., "resnet_encoder"). Default
        is "resnet_encoder".
        emb_dim (int, optional): Embedding dimension for tokens. Default is 80.
        dec_dim (int, optional): Decoder hidden dimension. Default is 512.
        attn_dim (int, optional): Attention mechanism dimension. Default is 512.
        num_layers (int, optional): Number of layers in the decoder. Default is 1.
        dropout (float, optional): Dropout rate. Default is 0.1.
        bidirectional (bool, optional): If True, use bidirectional encoder. Default is False.
        decode_type (str, optional): Decoding strategy ("greedy" or "beam"). Default is "greedy".
        text (Text, optional): Text processing utility for tokenization and detokenization.
        beam_width (int, optional): Beam width for beam search decoding. Default is 5.
        sos_id (int, optional): Start-of-sequence token ID. Default is 1.
        eos_id (int, optional): End-of-sequence token ID. Default is 2.
        log_step (int, optional): Logging frequency (in steps). Default is 100.
        log_text (bool, optional): If True, logs sample predictions during validation/testing.
        Default is False.
        nhead (int, optional): Number of attention heads (for transformer-based encoders).
        Default is 16.
        enc_layers (int, optional): Number of encoder layers. Default is 2.
        cnn_channels (int, optional): Number of channels in CNN encoder. Default is 32.

    Attributes:
        model (Image2Latex): The underlying image-to-LaTeX model.
        criterion (nn.CrossEntropyLoss): Loss function for training.
        lr (float): Learning rate.
        total_steps (int): Total training steps.
        text (Text): Text utility for tokenization.
        max_length (int): Maximum output sequence length for decoding.
        log_step (int): Logging frequency.
        log_text (bool): Whether to log text predictions.
        exact_match: Metric for exact match evaluation.
        bleu: Metric for BLEU score evaluation.

    Methods:
        configure_optimizers(): Sets up optimizer and learning rate scheduler.
        forward(images, formulas, formula_len, *args, **kwargs): Forward pass through the model.
        training_step(batch, batch_idx, *args, **kwargs): Training step logic.
        validation_step(batch, batch_idx, *args, **kwargs): Validation step logic, computes metrics.
        test_step(batch, batch_idx, *args, **kwargs): Test step logic, computes metrics.
        predict_step(batch, *args, **kwargs): Prediction step for inference.

    """

    def __init__(
        self,
        lr,
        total_steps,
        n_class: int,
        enc_dim: int = 512,
        enc_type: str = "resnet_encoder",
        emb_dim: int = 80,
        dec_dim: int = 512,
        attn_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        decode_type: str = "greedy",
        text: Text = None,
        beam_width: int = 5,
        sos_id: int = 1,
        eos_id: int = 2,
        log_step: int = 100,
        log_text: bool = False,
        nhead: int = 16,
        enc_layers: int = 2,
        cnn_channels: int = 32,
    ):
        super().__init__()
        self.model = Image2Latex(
            n_class=n_class,
            enc_dim=enc_dim,
            enc_type=enc_type,
            emb_dim=emb_dim,
            dec_dim=dec_dim,
            attn_dim=attn_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            decode_type=decode_type,
            beam_width=beam_width,
            text=text,
            sos_id=sos_id,
            eos_id=eos_id,
            nhead=nhead,
            enc_layers=enc_layers,
            cnn_channels=cnn_channels,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.total_steps = total_steps
        self.text = text
        self.max_length = 150

        self.example_input_array = (
            torch.randn(1, 3, 64, 384),  # images
            torch.randint(0, self.text.n_class, (1, self.max_length)),  # formulas
            torch.tensor([self.max_length]),  # formula_len
        )

        self.log_step = log_step
        self.log_text = log_text
        self.n_class = n_class
        self.exact_match = load("exact_match")
        self.bleu = load("bleu")
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.98))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.total_steps,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def forward(self, images, formulas, formula_len, *args, **kwargs):
        return self.model(images, formulas, formula_len)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        images, formulas, formula_len = batch

        images = images.to(self.device)
        formulas = formulas.to(self.device)
        formula_len = formula_len.to(self.device)

        formulas_in = formulas[:, :-1]
        formulas_out = formulas[:, 1:]

        outputs = self.model(images, formulas_in, formula_len)

        bs, t, _ = outputs.size()
        _o = outputs.reshape(bs * t, -1)
        _t = formulas_out.reshape(-1)
        loss = self.criterion(_o, _t)

        self.log("train loss", loss, sync_dist=True)

        return loss

    @torch.no_grad
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        images, formulas, formula_len = batch

        images = images.to(self.device)
        formulas = formulas.to(self.device)
        formula_len = formula_len.to(self.device)

        formulas_in = formulas[:, :-1]
        formulas_out = formulas[:, 1:]

        outputs = self.model(images, formulas_in, formula_len)

        bs, t, _ = outputs.size()
        _o = outputs.reshape(bs * t, -1)
        _t = formulas_out.reshape(-1)

        loss = self.criterion(_o, _t)

        predicts = [
            self.text.tokenize(self.model.decode(i.unsqueeze(0), self.max_length))
            for i in images
        ]
        truths = [self.text.tokenize(self.text.int2text(i)) for i in formulas]

        edit_dist = torch.mean(
            torch.Tensor(
                [edit_distance(tru, pre) for pre, tru in zip(predicts, truths)],
                device=self.device,
                dtype=torch.float,
            )
        )

        # edit_dist_norm = torch.mean(
        #     torch.Tensor(
        #         [
        #             edit_distance(tru, pre) / (max(1e-5, len(tru)))
        #             for pre, tru in zip(predicts, truths)
        #         ]
        #     )
        # )

        def safe_bleu(pre, tru):
            if not len(pre) or not len(tru):  # Если какая-то строка пустая
                return 0.0
            return self.bleu.compute(
                predictions=[" ".join(pre)], references=[" ".join(tru)]
            )["bleu"]

        # bleu4 = torch.mean(
        #     torch.Tensor(
        #         [
        #             torch.tensor(safe_bleu(pre, tru))
        #             for pre, tru in zip(predicts, truths)
        #         ]
        #     )
        # )

        # em = torch.mean(
        #     torch.Tensor(
        #         [
        #             torch.tensor(
        #                 self.exact_match.compute(
        #                     predictions=[" ".join(pre)], references=[" ".join(tru)]
        #                 )["exact_match"]
        #             )
        #             for pre, tru in zip(predicts, truths)
        #         ]
        #     )
        # )

        if self.log_text and ((batch_idx % self.log_step) == 0):
            truth, pred = truths[0], predicts[0]
            rank_zero_info("=" * 20)
            rank_zero_info(f"Truth: [{' '.join(truth)}] \nPredict: [{' '.join(pred)}]")
            rank_zero_info("=" * 20)

        self.log("val_loss", loss, sync_dist=True)
        # self.log("val_edit_distance_norm", edit_dist_norm, sync_dist=True)
        self.log("val_edit_distance", edit_dist, sync_dist=True)
        # self.log("val_bleu4", bleu4, sync_dist=True)
        # self.log("val_exact_match", em, sync_dist=True)

        return loss, edit_dist

    # bleu4,

    @torch.no_grad
    def test_step(self, batch, batch_idx, *args, **kwargs):
        images, formulas, formula_len = batch

        images = images.to(self.device)
        formulas = formulas.to(self.device)
        formula_len = formula_len.to(self.device)

        formulas_in = formulas[:, :-1]
        formulas_out = formulas[:, 1:]

        outputs = self.model(images, formulas_in, formula_len)

        bs, t, _ = outputs.size()
        _o = outputs.reshape(bs * t, -1)
        _t = formulas_out.reshape(-1)

        loss = self.criterion(_o, _t)

        predicts = [
            self.text.tokenize(self.model.decode(i.unsqueeze(0), self.max_length))
            for i in images
        ]
        truths = [self.text.tokenize(self.text.int2text(i)) for i in formulas]

        edit_dist = torch.mean(
            torch.Tensor(
                [edit_distance(tru, pre) for pre, tru in zip(predicts, truths)],
                device=self.device,
                dtype=torch.float,
            )
        )

        edit_dist_norm = torch.mean(
            torch.Tensor(
                [
                    edit_distance(tru, pre) / (max(1e-5, len(tru)))
                    for pre, tru in zip(predicts, truths)
                ],
                device=self.device,
                dtype=torch.float,
            )
        )

        def safe_bleu(pre, tru):
            if not len(pre) or not len(tru):  # Если какая-то строка пустая
                return 0.0
            return self.bleu.compute(
                predictions=[" ".join(pre)], references=[" ".join(tru)]
            )["bleu"]

        bleu_scores = [safe_bleu(pre, tru) for pre, tru in zip(predicts, truths)]
        bleu4 = torch.tensor(bleu_scores, device=self.device, dtype=torch.float).mean()

        em_scores = [
            self.exact_match.compute(
                predictions=[" ".join(pre)], references=[" ".join(tru)]
            )["exact_match"]
            for pre, tru in zip(predicts, truths)
        ]
        em = torch.tensor(em_scores, device=self.device, dtype=torch.float).mean()

        if self.log_text and ((batch_idx % self.log_step) == 0):
            truth, pred = truths[0], predicts[0]
            rank_zero_info("=" * 20)
            rank_zero_info(f"Truth: [{' '.join(truth)}] \nPredict: [{' '.join(pred)}]")
            rank_zero_info("=" * 20)

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_edit_distance", edit_dist, sync_dist=True)
        self.log("test_edit_distance_norm", edit_dist_norm, sync_dist=True)
        self.log("test_bleu4", bleu4, sync_dist=True)
        self.log("test_exact_match", em, sync_dist=True)

        return loss, edit_dist, edit_dist_norm, bleu4, em

    @torch.no_grad
    def predict_step(self, batch, *args, **kwargs):
        image = batch

        image = image.to(self.device)

        latex = self.model.decode(image, self.max_length)

        rank_zero_info(f"Predicted: {latex}")

        return latex

    def clone_for_stats(self):
        # Clone a model for FLOPs estimation — skip non-essential things like metrics
        return Image2LatexModel(
            lr=self.hparams.lr,
            total_steps=self.hparams.total_steps,
            n_class=self.hparams.n_class,
            enc_dim=self.hparams.enc_dim,
            enc_type=self.hparams.enc_type,
            emb_dim=self.hparams.emb_dim,
            dec_dim=self.hparams.dec_dim,
            attn_dim=self.hparams.attn_dim,
            num_layers=self.hparams.num_layers,
            dropout=self.hparams.dropout,
            bidirectional=self.hparams.bidirectional,
            decode_type=self.hparams.decode_type,
            text=self.hparams.text,  # required for dummy input and decoding
            beam_width=self.hparams.beam_width,
            sos_id=self.hparams.sos_id,
            eos_id=self.hparams.eos_id,
            log_step=self.hparams.log_step,
            log_text=False,  # disable text logging for stats copy
            nhead=self.hparams.nhead,
            enc_layers=self.hparams.enc_layers,
            cnn_channels=self.hparams.cnn_channels,
        )
