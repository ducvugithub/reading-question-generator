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
    def __init__(self, model_name: str, n_classes: int = 3, dropout: float = 0.1,
                 freeze_encoder: bool = False, unfreeze_last_n_layers: int = 0):
        """freeze_encoder: freeze all encoder params (train only the classifier head --
        a linear probe). unfreeze_last_n_layers: with freeze_encoder, re-enable gradients
        for the top N transformer blocks on top of the head -- a middle ground between a
        pure linear probe and full fine-tuning, intended for small datasets (e.g. this
        project's ~1K-item cascade-IRT pilot) where full fine-tuning of all ~125M params
        overfits fast. Assumes a BERT/RoBERTa-style `.encoder.layer` ModuleList; other
        architectures (e.g. DeBERTa-v2/v3) may need a different attribute path."""
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name).float()
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, n_classes)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            if unfreeze_last_n_layers > 0:
                layers = self.encoder.encoder.layer
                for layer in layers[-unfreeze_last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        # Use [CLS] token representation
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.drop(cls.to(self.classifier.weight.dtype)))
