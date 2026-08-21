"""
Contrastive encoder for Question Difficulty Estimation.

Produces L2-normalised embeddings — similar difficulty → close in embedding space.
After contrastive training, a logistic regression head is fit on frozen embeddings
for the final EASY / MEDIUM / HARD classification.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ContrastiveQDE(nn.Module):
    def __init__(self, model_name: str, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name).float()
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        # Two-layer projection head — standard in contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        cls = self.drop(out.last_hidden_state[:, 0, :])
        emb = self.projector(cls.to(next(self.projector.parameters()).dtype))
        return F.normalize(emb, dim=-1)  # unit sphere → cosine == dot product
