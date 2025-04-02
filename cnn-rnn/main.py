#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import torch
from torch import nn, Tensor
from model.model import Image2LatexModel
from data.dataset import LatexDataset, LatexPredictDataset
from data.datamodule import DataModule
from model.text import Text100k, Text170k
import pytorch_lightning as pl
import argparse
import numpy as np
from pytorch_lightning.loggers import Logger
import sys
from datetime import datetime
import os
import math
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.profilers import AdvancedProfiler


class FileLogger(Logger):
    def __init__(self, train: bool, val: bool, test: bool, predict: bool):
        super().__init__()

        modes = "-".join(
            [
                name
                for flag, name in zip(
                    [train, val, test, predict], ["train", "val", "test", "predict"]
                )
                if flag
            ]
        )
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.log_path = f"logs/{modes}-{timestamp}.log"
        self.log_file = open(self.log_path, "a")

    @property
    def name(self):
        return "file_logger"

    @property
    def version(self):
        return "1.0"

    @rank_zero_only
    def log_hyperparams(self, params):
        with open(self.log_path, "a") as f:
            print(f"[HYPERPARAMS] {params}", file=f)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        with open(self.log_path, "a") as f:
            print(f"[METRICS] Step {step}: {metrics}", file=f)

    @rank_zero_only
    def finalize(self, status):
        with open(self.log_path, "a") as f:
            print(f"[FINALIZE] {status}", file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training image2latex")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulate-batch", type=int, default=32)
    parser.add_argument("--data-path", type=str, help="data path")
    parser.add_argument("--img-path", type=str, help="image folder path")
    parser.add_argument(
        "--predict-img-path", type=str, help="image for predict path", default=None
    )

    parser.add_argument(
        "--dataset", type=str, help="choose dataset [100k, 170k]", default="100k"
    )
    parser.add_argument("--vocab_file", type=str, help="path to vocab file")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--val", action="store_true")
    # do not use ddp with test
    # https://github.com/Lightning-AI/pytorch-lightning/issues/12862
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--log-text", action="store_true")
    parser.add_argument("--train-sample", type=int, default=5000)
    parser.add_argument("--val-sample", type=int, default=1000)
    parser.add_argument("--test-sample", type=int, default=1000)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--log-step", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=12)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--enc-type", type=str, default="resnet_encoder")
    # conv_row_encoder, conv_encoder, conv_bn_encoder resnet_row_encoder

    parser.add_argument("--enc-dim", type=int, default=512)
    parser.add_argument("--emb-dim", type=int, default=80)
    parser.add_argument("--attn-dim", type=int, default=512)
    parser.add_argument("--dec-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--decode-type",
        type=str,
        default="greedy",
        help="Choose between [greedy, beamsearch]",
    )
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--model-name", type=str, default="conv_lstm")
    """
    Gradient clipping is a method where the error derivative is changed or clipped 
    to a threshold during backward propagation through the network, and the clipped 
    gradients are used to update the weights. 
    By rescaling the error derivative, the updates to the weights will also be rescaled, 
    dramatically decreasing the likelihood of an overflow or underflow.
    """
    parser.add_argument("--grad-clip", type=float, default=0)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--notebook", action="store_true")
    parser.add_argument("--rewrite-checkpoint-fitting", action="store_true")
    # parser.add_argument("--max-time", type=str, default="00:12:00:00")
    parser.add_argument("--checkpoints-path", type=str, default="checkpoints")
    parser.add_argument("--tb-logs-path", type=str, default="tb_logs")

    args = parser.parse_args()

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    text = None
    if args.dataset == "100k":
        text = Text100k(args.vocab_file)
    elif args.dataset == "170k":
        text = Text170k(args.vocab_file)

    train_set = LatexDataset(
        data_path=args.data_path,
        img_path=args.img_path,
        data_type="train",
        n_sample=args.train_sample,
        dataset=args.dataset,
    )
    val_set = LatexDataset(
        data_path=args.data_path,
        img_path=args.img_path,
        data_type="validate",
        n_sample=args.val_sample,
        dataset=args.dataset,
    )
    test_set = LatexDataset(
        data_path=args.data_path,
        img_path=args.img_path,
        data_type="test",
        n_sample=args.test_sample,
        dataset=args.dataset,
    )
    predict_set = LatexPredictDataset(predict_img_path=args.predict_img_path)

    steps_per_epoch = math.ceil(len(train_set) / args.batch_size)
    accumulate_grad_batches = args.accumulate_batch // args.batch_size
    assert accumulate_grad_batches > 0

    effective_steps_per_epoch = steps_per_epoch // accumulate_grad_batches
    total_steps = effective_steps_per_epoch * args.max_epochs

    dm = DataModule(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        predict_set=predict_set,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        text=text,
    )

    tb_logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir=args.tb_logs_path, name="image2latex_model", log_graph=True
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.checkpoints_path,
        filename="model-{epoch:02d}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        auto_insert_metric_name=False,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    logger = FileLogger(args.train, args.val, args.test, args.predict)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    trainer = pl.Trainer(
        logger=[tb_logger, logger],
        #     profiler=AdvancedProfiler(
        #     dirpath="logs/profiler",  # where to save
        #     filename=f"{timestamp}-profile.txt",  # filename
        # ),
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.gpu else "auto",
        strategy="auto" if not args.gpu else "ddp_notebook" if args.notebook else "ddp",
        log_every_n_steps=1,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=accumulate_grad_batches,
        devices=-1 if args.gpu else 1,
        num_sanity_val_steps=1,
        # max_time=args.max_time,
    )

    ckpt_path = args.ckpt_path
    model = None

    if ckpt_path:
        if args.rewrite_checkpoint_fitting:
            print("[INFO] Loading weights only, overriding hyperparameters.")
            # Создаем модель с НОВЫМИ параметрами
            model = Image2LatexModel(
                lr=args.lr,
                total_steps=total_steps,
                n_class=text.n_class,
                enc_dim=args.enc_dim,
                enc_type=args.enc_type,
                emb_dim=args.emb_dim,
                dec_dim=args.dec_dim,
                attn_dim=args.attn_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                sos_id=text.sos_id,
                eos_id=text.eos_id,
                decode_type=args.decode_type,
                text=text,
                beam_width=args.beam_width,
                log_step=args.log_step,
                log_text=args.log_text,
            )
            # Загружаем только веса
            state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            model.load_state_dict(state_dict)
        else:
            print(
                "[INFO] Loading full checkpoint (including hyperparameters, scheduler, etc)."
            )
            model = Image2LatexModel.load_from_checkpoint(ckpt_path, map_location="cpu")
    else:
        print("[INFO] Starting model from scratch.")
        model = Image2LatexModel(
            lr=args.lr,
            total_steps=total_steps,
            n_class=text.n_class,
            enc_dim=args.enc_dim,
            enc_type=args.enc_type,
            emb_dim=args.emb_dim,
            dec_dim=args.dec_dim,
            attn_dim=args.attn_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            sos_id=text.sos_id,
            eos_id=text.eos_id,
            decode_type=args.decode_type,
            text=text,
            beam_width=args.beam_width,
            log_step=args.log_step,
            log_text=args.log_text,
        )

    # === TRAIN ===
    if args.train:
        print("=" * 10 + "[Train]" + "=" * 10)
        trainer.fit(
            datamodule=dm,
            model=model,
            ckpt_path=None if args.rewrite_checkpoint_fitting else ckpt_path,
        )

    # === VALIDATE ===
    if args.val:
        print("=" * 10 + "[Validate]" + "=" * 10)
        trainer.validate(datamodule=dm, model=model)

    # === TEST ===
    if args.test:
        print("=" * 10 + "[Test]" + "=" * 10)
        trainer.test(datamodule=dm, model=model)

    # === PREDICT ===
    if args.predict:
        print("=" * 10 + "[Predict]" + "=" * 10)
        trainer.predict(datamodule=dm, model=model)
