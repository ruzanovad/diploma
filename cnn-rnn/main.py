"""
main.py -- entry point of the application
handles model training/validation/testing/prediction(inference)
"""

import datetime
import math
import os
from datetime import datetime

import numpy as np
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

import lightning.pytorch as pl
from lightning.pytorch.loggers.logger import Logger

# from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.throughput import measure_flops
from lightning.pytorch.callbacks import ModelSummary

from data.datamodule import DataModule
from data.dataset import LatexDataset, LatexPredictDataset
from model.model import Image2LatexModel
from model.text import Text100k, Text170k


class FileLogger(Logger):
    """
    FileLogger is a custom logger class for logging training,
    validation, testing, and prediction events to a file.

    Args:
        train (bool): Whether to log training events.
        val (bool): Whether to log validation events.
        test (bool): Whether to log testing events.
        predict (bool): Whether to log prediction events.

    Attributes:
        log_path (str): Path to the log file.
        log_file (file object): File object for the log file.

    Properties:
        name (str): Returns the name of the logger ("file_logger").
        version (str): Returns the version of the logger ("1.0").

    Methods:
        log_hyperparams(params): Logs hyperparameters to the log file.
        log_metrics(metrics, step): Logs metrics at a given step to the log file.
        finalize(status): Logs the finalization status to the log file.

    Notes:
        - Log files are saved in the "logs" directory, with
        filenames indicating the active modes and a timestamp.
        - Only the main process (rank zero) performs logging
        in distributed settings.
    """

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

    @property
    def name(self):
        return "file_logger"

    @property
    def version(self):
        return "1.0"

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        with open(self.log_path, "a", encoding="utf-8") as f:
            print(f"[HYPERPARAMS] {params}", file=f)

    @rank_zero_only
    def log_metrics(self, metrics, step=None, **kwargs):
        with open(self.log_path, "a", encoding="utf-8") as f:
            print(f"[METRICS] Step {step}: {metrics}", file=f)

    @rank_zero_only
    def finalize(self, status):
        with open(self.log_path, "a", encoding="utf-8") as f:
            print(f"[FINALIZE] {status}", file=f)


def profile_model(model: pl.LightningModule, logger=None):
    """Log FLOPs, parameter count, and speed using dummy inputs."""
    try:
        if hasattr(model, "clone_for_stats"):
            prof_model = model.clone_for_stats().to("cpu")
        else:
            prof_model = model.to("cpu")

        # Dummy input
        img, frm, ln = model.example_input_array
        img = img.to("cpu")
        frm = frm.to("cpu")
        ln = ln.to("cpu")

        # Parameters
        total_params = sum(p.numel() for p in prof_model.parameters())

        # FLOPs (replace this with ptflops or torch.fx if needed)
        flops = measure_flops(
            prof_model, lambda: prof_model(img, frm, ln)
        )  # must define this
        gflops = flops / 1e9

        # Inference speed
        prof_model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = prof_model(img, frm, ln)
            import time

            start = time.time()
            for _ in range(100):
                _ = prof_model(img, frm, ln)
            end = time.time()
            speed_ms = ((end - start) / 100) * 1000

        metrics = {
            "model/parameters": total_params,
            "model/GFLOPs": gflops,
            "model/speed_PyTorch(ms)": speed_ms,
        }

        # Optional: print
        print("\n📊 Model Stats:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        # Optional: log to trainer's logger
        if logger is not None:
            if hasattr(logger, "log_metrics"):
                logger.log_metrics(metrics)
            elif hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
                logger.experiment.log(metrics)

        return metrics

    except Exception as e:
        print(f"[Warning] Profiling failed: {e}")
        return {}


@hydra.main(config_path="configs", config_name="main", version_base=None)
def main(args: DictConfig):
    """
    Main entry point for training, validating, testing, and predicting with the Image2Latex model.

    This function sets up datasets, data modules, loggers, callbacks,
    and the PyTorch Lightning trainer
    according to the provided configuration. It supports resuming from checkpoints,
    overriding checkpoint
    hyperparameters, and running different stages (train, validate, test, predict)
    as specified in the arguments.

    Args:
        args (DictConfig): Configuration object containing all necessary parameters
        for data loading, model initialization, training, validation, testing, prediction, logging,
        and checkpointing.

    Workflow:
        1. Seeds all relevant random number generators for reproducibility.
        2. Initializes the appropriate text processor and datasets based on the dataset selection.
        3. Sets up the data module and loggers (TensorBoard and Weights & Biases).
        4. Configures callbacks for checkpointing and learning rate monitoring.
        5. Initializes the PyTorch Lightning Trainer with the specified settings.
        6. Loads the model from a checkpoint or initializes a new model as required.
        7. Executes training, validation, testing, and/or prediction based on the flags in args.

    Note:
        - The function expects that all required classes (e.g., Text100k, LatexDataset, DataModule,
          Image2LatexModel)
          and libraries (e.g., torch, numpy, pytorch_lightning as pl, OmegaConf)
          are properly imported.
        - The function does not return any value; it runs the selected stages as side effects.
    """
    print(OmegaConf.to_yaml(args))

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    pl.seed_everything(args.random_state)

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

    text2int_fn = text.text2int
    word2id = text.word2id
    sos_id = text.sos_id
    eos_id = text.eos_id

    dm = DataModule(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        predict_set=predict_set,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        text2int_fn=text2int_fn,
        word2id=word2id,
        sos_id=sos_id,
        eos_id=eos_id,
    )

    tb_logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir=args.tb_logs_path, name="image2latex_model", log_graph=True
    )

    wandb_logger = pl.loggers.wandb.WandbLogger(
        project="latex",
        name=args.model_name,
        save_dir=args.tb_logs_path,
        log_model=True,
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

    # logger = FileLogger(args.train, args.val, args.test, args.predict)

    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # stats_logger = ModelStatsLogger()

    trainer = pl.Trainer(
        logger=[tb_logger, wandb_logger],
        #     profiler=AdvancedProfiler(
        #     dirpath="logs/profiler",  # where to save
        #     filename=f"{timestamp}-profile.txt",  # filename
        # ),
        callbacks=[
            lr_monitor,
            checkpoint_callback,
            ModelSummary(3),
        ],
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.gpu else "auto",
        strategy=args.strategy,
        log_every_n_steps=1,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=accumulate_grad_batches,
        devices=args.devices,
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
                cnn_channels=args.cnn_channels,
                nhead=args.nhead,
                enc_layers=args.enc_layers,
            )

            profile_model(model, logger=wandb_logger)

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
            cnn_channels=args.cnn_channels,
            nhead=args.nhead,
            enc_layers=args.enc_layers,
        )

        profile_model(model, logger=wandb_logger)

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


if __name__ == "__main__":
    # because of linux `fork` method
    # torch.multiprocessing.set_start_method("spawn", force=True)
    main()
