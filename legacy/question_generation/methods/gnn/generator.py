"""
GNN-based question generation: GNN encodes graph structure → T5 decoder.

Node features: entity text encoded via XLM-R / FinT5 encoder.
GNN: 2-3 layer GraphSAGE or GAT (PyTorch Geometric).
Decoder: T5 cross-attending to GNN node embeddings.

Advantage over seq2seq: explicit structural encoding (multi-hop paths,
node centrality, neighborhood context).
Status: NOT YET IMPLEMENTED
"""
from __future__ import annotations
from question_generation.methods.base import QGMethod


class GNNMethod(QGMethod):
    def generate(self, text, anchor, cefr, lang, subgraph=None):
        raise NotImplementedError("GNNMethod is not yet implemented")
