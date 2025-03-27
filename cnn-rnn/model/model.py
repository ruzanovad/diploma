import torch
from torch import nn, Tensor
from .im2latex import Image2Latex
from .text import Text
from .utils import exact_match
import pytorch_lightning as pl
from torchaudio.functional import edit_distance
from evaluate import load
from pytorch_lightning.utilities.rank_zero import rank_zero_info


class Image2LatexModel(pl.LightningModule):
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
    ):
        super().__init__()
        self.model = Image2Latex(
            n_class,
            enc_dim,
            enc_type,
            emb_dim,
            dec_dim,
            attn_dim,
            num_layers,
            dropout,
            bidirectional,
            decode_type,
            text,
            beam_width,
            sos_id,
            eos_id,
        )
        self.example_input_array = (
            torch.randn([1, 3, 64, 384]),  # image
            torch.randint(0, n_class, [1, 20]),  # formula
            torch.tensor([20]),  # length
        )
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.total_steps = total_steps
        self.text = text
        self.max_length = 150
        self.log_step = log_step
        self.log_text = log_text
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

    def forward(self, images, formulas, formula_len):
        return self.model(images, formulas, formula_len)

    def training_step(self, batch, batch_idx):
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        images, formulas, formula_len = batch

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
    def validation_step(self, batch, batch_idx):
        images, formulas, formula_len = batch

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
                [
                    edit_distance(tru, pre) / len(tru)
                    for pre, tru in zip(predicts, truths)
                ]
            )
        )

        def safe_bleu(pre, tru):
            if not len(pre) or not len(tru):  # Если какая-то строка пустая
                return 0.0
            return self.bleu.compute(
                predictions=[" ".join(pre)], references=[" ".join(tru)]
            )["bleu"]

        bleu4 = torch.mean(
            torch.Tensor(
                [
                    torch.tensor(safe_bleu(pre, tru))
                    for pre, tru in zip(predicts, truths)
                ]
            )
        )

        em = torch.mean(
            torch.Tensor(
                [
                    torch.tensor(
                        self.exact_match.compute(
                            predictions=[" ".join(pre)], references=[" ".join(tru)]
                        )["exact_match"]
                    )
                    for pre, tru in zip(predicts, truths)
                ]
            )
        )

        if self.log_text and (batch_idx % self.log_step == 0):
            truth, pred = truths[0], predicts[0]
            rank_zero_info("=" * 20)
            rank_zero_info(f"Truth: [{' '.join(truth)}] \nPredict: [{' '.join(pred)}]")
            rank_zero_info("=" * 20)

        self.log("val_loss", loss, sync_dist=True)
        self.log("val_edit_distance", edit_dist, sync_dist=True)
        self.log("val_bleu4", bleu4, sync_dist=True)
        self.log("val_exact_match", em, sync_dist=True)

        return edit_dist, bleu4, em, loss

    @torch.no_grad
    def test_step(self, batch, batch_idx):
        images, formulas, formula_len = batch

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

        edit_dists = [
            edit_distance(tru, pre) / len(tru) if len(tru) > 0 else 0.0
            for pre, tru in zip(predicts, truths)
        ]
        edit_dist = torch.tensor(edit_dists).mean()

        def safe_bleu(pre, tru):
            if not len(pre) or not len(tru):  # Если какая-то строка пустая
                return 0.0
            return self.bleu.compute(
                predictions=[" ".join(pre)], references=[" ".join(tru)]
            )["bleu"]

        bleu_scores = [safe_bleu(pre, tru) for pre, tru in zip(predicts, truths)]
        bleu4 = torch.tensor(bleu_scores).mean()

        em_scores = [
            self.exact_match.compute(
                predictions=[" ".join(pre)], references=[" ".join(tru)]
            )["exact_match"]
            for pre, tru in zip(predicts, truths)
        ]
        em = torch.tensor(em_scores).mean()

        if self.log_text and batch_idx % self.log_step == 0:
            truth, pred = truths[0], predicts[0]
            rank_zero_info("=" * 20)
            rank_zero_info(f"Truth: [{' '.join(truth)}] \nPredict: [{' '.join(pred)}]")
            rank_zero_info("=" * 20)

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_edit_distance", edit_dist, sync_dist=True)
        self.log("test_bleu4", bleu4, sync_dist=True)
        self.log("test_exact_match", em, sync_dist=True)

        return edit_dist, bleu4, em, loss

    
    def predict_step(self, batch, batch_idx):
        image = batch

        latex = self.model.decode(image, self.max_length)

        rank_zero_info(f"Predicted: {latex}")

        return latex
