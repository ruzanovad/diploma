import json
from os import PathLike
import re
import torch
from torch import Tensor
from abc import ABC, abstractmethod


class Text(ABC):
    """
    Abstract base class for text processing in formula recognition.

    This class defines the interface for:
    - Tokenization (`tokenize`)
    - Text-to-index conversion (`text2int`)
    - Index-to-text conversion (`int2text`)
    """
    def __init__(self):
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2

    @abstractmethod
    def tokenize(self, formula: str):
        """
        Convert a formula string into a list of tokens.
        To be implemented in subclasses.
        """
        pass

    def int2text(self, x: Tensor):
        """
        Convert a tensor of token indices into a formula string.

        Args:
            x (Tensor): Tensor of token IDs (e.g., [1, 23, 45, 2])

        Returns:
            str: Token sequence as a string (space-separated)
        """
        return " ".join([self.id2word[i] for i in x if i > self.eos_id])

    def text2int(self, formula: str):
        """
        Convert a formula string into a tensor of token indices.

        Args:
            formula (str): The input formula (e.g., "\\frac a b")

        Returns:
            Tensor: Token indices as LongTensor
        """
        return torch.LongTensor([self.word2id[i] for i in self.tokenize(formula)])


class Text100k(Text):
    """
    Text handler for vocabularies with ~100k tokens.

    Uses regular expressions to tokenize LaTeX-style formulas more accurately.
    """
    def __init__(self, vocab_file: str | PathLike):
        """
        Args:
            vocab_file (str | PathLike): Path to a JSON file containing the vocabulary list
        """
        super().__init__()
        # self.id2word = json.load(open("data/vocab/100k_vocab.json", "r"))
        self.id2word = json.load(open(vocab_file, "r"))
        self.word2id = dict(zip(self.id2word, range(len(self.id2word))))
        self.TOKENIZE_PATTERN = re.compile(
            "(\\\\[a-zA-Z]+)|" + '((\\\\)*[$-/:-?{-~!"^_`\[\]])|' + "(\w)|" + "(\\\\)"
        )
        self.n_class = len(self.id2word)

    def tokenize(self, formula: str):
        """
        Tokenize a formula using a LaTeX-aware regex.

        Args:
            formula (str): The input formula

        Returns:
            List[str]: List of tokens
        """
        tokens = re.finditer(self.TOKENIZE_PATTERN, formula)
        tokens = list(map(lambda x: x.group(0), tokens))
        tokens = [x for x in tokens if x is not None and x != ""]
        return tokens


class Text170k(Text):
    """
    Text handler for vocabularies with ~170k tokens.

    Uses basic whitespace tokenization (formula must already be pre-tokenized).
    """
    def __init__(self, vocab_file: str | PathLike):
        """
        Args:
            vocab_file (str | PathLike): Path to a JSON file containing the vocabulary list
        """
        super().__init__()
        # self.id2word = json.load(open("data/vocab/100k_vocab.json", "r"))
        self.id2word = json.load(open(vocab_file, "r"))
        self.word2id = dict(zip(self.id2word, range(len(self.id2word))))
        self.n_class = len(self.id2word)

    def tokenize(self, formula: str):
                """
        Tokenize a formula by splitting on whitespace.

        Args:
            formula (str): Pre-tokenized input string

        Returns:
            List[str]: List of tokens
        """
        return formula.split()
