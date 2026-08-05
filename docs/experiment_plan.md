# Experiment Plan: Difficulty-Controlled KG-Enhanced Question Generation

## Research Questions

1. Does KG information improve question generation quality over passage+answer alone?
2. Does GNN-encoded KG (structured) outperform text-linearized KG (flat)?
3. Can we reliably condition generation on cognitive difficulty?
4. Does cross-attention GNN fusion into the decoder give the strongest difficulty control?

---

## Experimental Conditions

| ID | Name | Input | Novelty |
|---|---|---|---|
| M1 | Baseline | `passage + answer` | — |
| M2 | Difficulty-controlled baseline | `passage + answer + <difficulty>` | — |
| M3 | Linearized KG + difficulty | `passage + answer + kg_text + <difficulty>` | — |
| M4 | GNN prefix + difficulty | `passage + answer + GNN_prefix_tokens + <difficulty>` | Partial |
| M5 | GNN cross-attention + difficulty | `passage + answer` + GNN embeddings fused into decoder cross-attention | **Main novelty** |

### Input formats

**M1 — Baseline**
```
answer: Houston context: Beyoncé was born in Houston, Texas...
```

**M2 — Difficulty token only**
```
difficulty: hard answer: Houston context: Beyoncé was born in Houston, Texas...
```

**M3 — Linearized KG**
```
difficulty: hard answer: Houston
knowledge: Beyoncé | born_in | Houston ; Beyoncé | be | singer ; she | perform_in | competition
context: Beyoncé was born in Houston, Texas...
```

**M4 — GNN prefix injection**
- KG encoded with GAT → node embeddings projected to T5 token dimension
- Prepended as soft prefix tokens to encoder input
- Rest of input same as M2

**M5 — GNN cross-attention fusion (main novelty)**
- KG encoded with GAT → node + edge embeddings
- Injected into decoder cross-attention via an additional KG cross-attention layer
- Decoder attends to both encoder (passage) and GNN (KG) simultaneously
- Difficulty token still prepended to encoder input

---

## Model Candidates

### Seq2seq backbone

| Model | Params | Languages | Notes |
|---|---|---|---|
| `t5-base` | 220M | EN only | Strong EN baseline, well-studied for QG |
| `google/mt5-small` | 300M | 101 langs (EN, FI, RU) | Multilingual, efficient |
| `google/mt5-base` | 580M | 101 langs | Better quality, higher cost |
| `TurkuNLP/t5-v1_1-large-finnish` | 800M | FI only | Best for Finnish (later phase) |

**Primary choice: `t5-base` for EN experiments (M1–M5). Switch to `mt5-small` for multilingual phase.**

### GNN encoder (M4, M5)

| Architecture | Handles relation types | Notes |
|---|---|---|
| GAT (Graph Attention Network) | No (needs edge features) | Simple, interpretable attention weights |
| RGCN (Relational GCN) | Yes | Natural fit — KG triples have typed relations |
| GraphSAGE | No | Scalable but less suited to typed KGs |

**Primary choice: RGCN** — KG triples have explicit relation types (`born_in`, `perform_in`, etc.) which RGCN handles natively. GAT as fallback.

---

## Difficulty Signal

Source: `GraphCognitiveDifficultyEstimator` (rule-based, see `docs/cognitive_difficulty_estimation.md`)

```
score = 0.45 × s_qtype + 0.30 × s_coref + 0.15 × s_coverage + 0.10 × s_density
label = easy (< 0.33) | medium (0.33–0.67) | hard (> 0.67)
```

Control token: `<easy>`, `<medium>`, `<hard>` prepended to encoder input for M2–M5.

**Status: post-processing script needed** to add `cognitive_difficulty` field to all JSONL records before training.

---

## Dataset

| Split | EN | FI | RU |
|---|---|---|---|
| train | 71,969 | 5,295 | 4,082 |
| eval | 18,201 | 1,395 | 1,002 |

Phase 1: EN only (`t5-base`). Phase 2: multilingual (`mt5-small`, EN+FI+RU).

**Note:** FI and RU have `kg_coref = null` (Stanza coref is EN-only). M5 uses `kg_raw` for all languages.

---

## Training Plan

### Phase 1 — EN ablation (M1–M5)

- Base model: `t5-base`
- Batch size: 16 (gradient accumulation × 4 = effective 64)
- Learning rate: 5e-4 with linear warmup
- Max input length: 512 tokens
- Max output length: 64 tokens
- Epochs: 5
- Hardware: 1× A100 (Mahti gpusmall)

### Phase 2 — Multilingual (best model from Phase 1)

- Base model: `mt5-small`
- Same hyperparameters
- Languages: EN + FI (+ RU if time permits)

---

## Evaluation

### Automatic metrics (per difficulty bucket: easy / medium / hard)

| Metric | What it measures |
|---|---|
| BLEU-4 | N-gram overlap with reference |
| ROUGE-L | Longest common subsequence |
| BERTScore | Semantic similarity (multilingual BERT) |
| Difficulty accuracy | Does generated question match target difficulty label? |

### Difficulty accuracy (key metric for M2–M5)
- Run `GraphCognitiveDifficultyEstimator` on generated questions
- Compare predicted label vs. control token used during generation
- Measures how well the model honours the difficulty conditioning

### Closest prior work to beat
- Kumar et al. (2019) ISWC — KG hop count difficulty, LSTM-based
- Our M3 (linearized) should already beat this; M5 is the target

---

## Implementation Order

1. `scripts/add_cognitive_difficulty.py` — post-process dataset, add `cognitive_difficulty` field
2. `scripts/train_seq2seq.py` — M1, M2, M3 (no GNN, pure seq2seq)
3. `models/gnn_encoder.py` — RGCN encoder shared by M4 and M5
4. `models/t5_gnn_prefix.py` — M4: prefix injection
5. `models/t5_gnn_decoder.py` — M5: cross-attention fusion
6. `scripts/evaluate_qg.py` — BLEU/ROUGE/BERTScore + difficulty accuracy
7. `slurms/training/` — SLURM jobs for each condition
