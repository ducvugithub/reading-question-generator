"""
Encoder fine-tune for Question Difficulty Estimation.

Input: [CLS] question [SEP] passage [SEP]  (answer appended to question)
Output: 3-class softmax  EASY / MEDIUM / HARD
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class EncoderQDE(nn.Module):
    def __init__(self, model_name: str, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name).float()
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        # Use [CLS] token representation
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.drop(cls.to(self.classifier.weight.dtype)))
