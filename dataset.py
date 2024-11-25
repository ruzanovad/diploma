# https://huggingface.co/datasets/Helsinki-NLP/opus-100

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    """
    This dataset class prepares the data for training a
    Transformer model on a translation task by handling
    tokenization, padding, special tokens, and masking,
    ensuring the model receives correctly formatted inputs. Let’s take a look in more detail.
    """

    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq):
        super().__init__()
        self.seq = seq

        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        # The source language identifier (e.g., ‘en’ for English).
        self.src_lang = src_lang
        # The target language identifier (e.g., ‘it’ for Italian).
        self.tgt_lang = tgt_lang

        # Start of sentence
        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        # End of sentence
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        # Padding token
        # Padding adds a special padding token to ensure
        # shorter sequences will have the same length as
        # either the longest sequence in a batch or the maximum length
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        """
        The __getitem__ method retrieves a data sample at a given index idx
        and processes it as follows:
        """
        # Retrieve Texts
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # Transform the text into tokens (tokenizing)
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Padding calculation
        # Calculates the number of padding tokens needed to make the sequence
        # length equal to self.seq.

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = (
            self.seq - len(enc_input_tokens) - 2
        )  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq - len(dec_input_tokens) - 1

        # Validation
        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Add only <s> token (We don't expect decoder to decode end of sentence token)
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Represents the target output (ground truth) used to compute the loss during training.
        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq long
        assert encoder_input.size(0) == self.seq
        assert decoder_input.size(0) == self.seq
        assert label.size(0) == self.seq

        return {
            "encoder_input": encoder_input,  # (seq)
            "decoder_input": decoder_input,  # (seq)
            "encoder_mask": (encoder_input != self.pad_token)
            # Adding a dimension of size 1 does not change the
            # data of the tensor but alters its shape.
            .unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int()
            & causal_mask(decoder_input.size(0)),  # (1, seq) & (1, seq, seq),
            "label": label,  # (seq)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }



# 1. причинность
# 2. причинная связь
# WE CAN SEE ONLY CURRENT AND OR PREVIOUS TOKEN
def causal_mask(size):
    """
    The causal_mask function generates an upper triangular
    matrix (excluding the diagonal) filled with ones, and
    then converts it into a boolean mask where zeros represent
    allowed positions (i.e., positions that can attend to previous
    positions in the sequence). This mask ensures that during training,
    each position in the decoder can only attend to previous positions,
    enforcing causality.

    Without the causal mask:

    The decoder could attend to future tokens during training.
    This would allow it to "cheat," leading to incorrect training dynamics 
    and poor generalization during inference (where future tokens aren't available).
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
