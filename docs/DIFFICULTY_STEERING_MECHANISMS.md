# Difficulty-Controllable Question Generation: Steering Mechanisms

Five approaches to control question generation difficulty in T5-based QG models.

---

## 1. Token-Only Conditioning (Current Implementation)

**Flow:**
```
Passage + [DIFFICULTY_TOKEN]
     ↓
T5 Encoder
     ↓
T5 Decoder
     ↓
Generated Question
```

**How it works:**
- Prepend `[EASY]`, `[MEDIUM]`, or `[HARD]` token to input
- Model learns to interpret tokens from training examples
- No model modifications needed

**Pros:**
- ✅ Simple to implement
- ✅ No additional parameters
- ✅ Zero computational overhead
- ✅ Already trained and working

**Cons:**
- ❌ Feels like "prompt engineering" (weak novelty for main conferences)
- ❌ Limited expressiveness (tokens compete with passage for attention)
- ❌ Hard to analyze what the model learned

**Implementation complexity:** Trivial

**Paper angle:** Not suitable alone; use as baseline

**Estimated scores:**
- Main conference: ⭐☆☆☆☆ (too simple)
- BEA: ⭐⭐⭐☆☆ (solid for workshop)

---

## 2. Adapter Modules ⭐ RECOMMENDED

**Flow:**
```
Passage
     ↓
T5 Encoder
     ↓
Select Adapter by difficulty
┌──────────────────────┐
│  ADAPTER MODULES     │
│ ┌──────────────────┐ │
│ │ Adapter_EASY     │ │
│ │ Adapter_MEDIUM   │ │
│ │ Adapter_HARD     │ │
│ └──────────────────┘ │
└──────────────────────┘
     ↓
T5 Decoder
     ↓
Generated Question
```

**How it works:**
- Add small trainable adapter layer (2-4% extra params) for each difficulty level
- Adapters applied after encoder → before decoder
- Each adapter learns difficulty-specific transformations
- Base T5 weights frozen or lightly fine-tuned

**Pros:**
- ✅ Novel technical contribution (adapters + difficulty control)
- ✅ Parameter-efficient (adapters are tiny)
- ✅ Highly interpretable (can visualize what each adapter does)
- ✅ Modular (easy to extend to 5+ levels)
- ✅ Strong paper narrative: "Modular difficulty control"

**Cons:**
- ⚠️ Requires retraining (4-6 GPU hours)
- ⚠️ Slightly more complex implementation

**Implementation complexity:** Medium

**Paper angle:** 
- "Adapter-based difficulty control for controllable QG"
- Analyze adapter weights to understand what changes with difficulty
- Visualize: harder questions → what encoder changes are needed?

**Estimated scores:**
- Main conference: ⭐⭐⭐⭐☆ (solid technical novelty)
- BEA: ⭐⭐⭐⭐⭐ (perfect for workshop)

**Implementation sketch:**
```python
class DifficultyAdapter(nn.Module):
    def __init__(self, hidden_dim, difficulty_levels=3):
        self.adapters = nn.ModuleDict({
            f"adapter_{level}": nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, hidden_dim)
            )
            for level in ["easy", "medium", "hard"]
        })
    
    def forward(self, hidden_states, difficulty):
        return hidden_states + self.adapters[difficulty](hidden_states)
```

---

## 3. Prefix Tuning

**Flow:**
```
[DIFFICULTY_PREFIX] (learnable)  Passage
       ↓                              ↓
       └──────────────────────────────┘
                    ↓
            T5 Encoder
                    ↓
            T5 Decoder
                    ↓
        Generated Question
```

**How it works:**
- Prepend learnable embeddings (20-30 tokens) specific to each difficulty
- Prefix is frozen to task-specific task but not shared with main model
- Lighter version of fine-tuning

**Pros:**
- ✅ Parameter-efficient (only ~0.1% extra params)
- ✅ Task-specific but model-agnostic
- ✅ Can be trained in isolation
- ✅ Interpretable: examine prefix embeddings

**Cons:**
- ⚠️ Still requires training (3-4 GPU hours)
- ⚠️ Slightly less expressive than adapters
- ⚠️ Novel but not groundbreaking

**Implementation complexity:** Medium

**Paper angle:** 
- "Parameter-efficient difficulty control via prefix tuning"
- Compare with full fine-tuning (adapters) on efficiency frontier
- Analyze: do prefix embeddings cluster by difficulty?

**Estimated scores:**
- Main conference: ⭐⭐⭐☆☆ (decent novelty, but adapters are better)
- BEA: ⭐⭐⭐⭐☆ (good for workshop)

---

## 4. Auxiliary Classification Loss

**Flow:**
```
Passage
   ↓
T5 Encoder
   ├────────────────────────────────┐
   ↓                                ↓
T5 Decoder                   Classification Head
   ↓                                ↓
Generated Question        Predicted Difficulty
   ↓                                ↓
   └─────────────┬──────────────────┘
                 ↓
    Combined Loss = QG_Loss + α × Classification_Loss
```

**How it works:**
- Add classification head on top of encoder
- Train jointly: generate questions + predict difficulty
- Multi-task learning forces encoder to learn difficulty-aware representations
- No model modifications needed (just add loss term)

**Pros:**
- ✅ Easiest to implement (just add loss term)
- ✅ No architectural changes needed
- ✅ Interpretable: classification head learns difficulty patterns
- ✅ Can reuse existing model checkpoint
- ✅ Strong paper narrative: "Multi-task learning for controllable QG"

**Cons:**
- ⚠️ Requires retraining (4-6 GPU hours)
- ⚠️ Less novel than adapters (multi-task learning is well-known)
- ⚠️ May hurt QG performance if α not tuned carefully

**Implementation complexity:** Easy

**Paper angle:**
- "Joint QG + difficulty prediction for controllable generation"
- Analyze what the classification head learns
- Ablation: what happens with/without auxiliary loss?
- Paper narrative: "Using classification signal to steer generation"

**Estimated scores:**
- Main conference: ⭐⭐⭐☆☆ (solid contribution, but adapters are stronger)
- BEA: ⭐⭐⭐⭐☆ (good for workshop)

**Implementation sketch:**
```python
class QGWithAuxiliaryLoss(nn.Module):
    def __init__(self, base_model):
        self.base_model = base_model
        self.difficulty_classifier = nn.Linear(768, 3)  # 3 difficulty levels
    
    def forward(self, input_ids, attention_mask, labels=None, difficulty_labels=None):
        encoder_output = self.base_model.encoder(input_ids, attention_mask)
        
        # QG path
        qg_output = self.base_model.decoder(encoder_output, labels=labels)
        qg_loss = qg_output.loss
        
        # Classification path
        cls_output = self.difficulty_classifier(encoder_output[:, 0, :])
        cls_loss = F.cross_entropy(cls_output, difficulty_labels)
        
        total_loss = qg_loss + 0.5 * cls_loss
        return total_loss
```

---

## 5. Constrained Decoding

**Flow:**
```
Passage
   ↓
T5 Encoder
   ↓
Beam Search with Constraints
├─ EASY:   penalize(complex_vocab), reward(short)
├─ MEDIUM: neutral constraints
└─ HARD:   penalize(short), reward(complex_vocab)
   ↓
Generated Question
```

**How it works:**
- No training changes needed
- At inference time, apply difficulty-specific constraints to beam search
- Penalize/reward based on target difficulty
- Examples:
  - EASY: length < 15 tokens, use top 1000 vocab words only
  - HARD: length > 20 tokens, encourage rare vocabulary

**Pros:**
- ✅ No retraining needed (instant to implement)
- ✅ Can experiment without GPU
- ✅ Highly interpretable (rules are explicit)
- ✅ Very fast inference
- ✅ Paper angle: "How to steer QG without retraining"

**Cons:**
- ❌ Feels hacky (not learned, just heuristics)
- ❌ Limited novelty for main conference
- ❌ May hurt ROUGE/BLEU scores (constrained output less natural)
- ❌ Hard to tune constraint thresholds

**Implementation complexity:** Easy

**Paper angle:**
- Probably not suitable as main contribution
- Use as: "Baseline for comparison" or "Analysis of what makes questions harder"
- Could work for: "What linguistic patterns indicate difficulty?"

**Estimated scores:**
- Main conference: ⭐☆☆☆☆ (too heuristic)
- BEA: ⭐⭐☆☆☆ (interesting analysis but weak contribution)

**Implementation sketch:**
```python
def constrained_generation(model, input_ids, difficulty, max_length=50):
    if difficulty == "EASY":
        length_penalty = lambda logits, len: logits + 0.1 * len  # penalize long
        vocab_mask = get_top_k_vocab(1000)  # restrict vocabulary
    elif difficulty == "HARD":
        length_penalty = lambda logits, len: logits - 0.1 * len  # reward long
        vocab_mask = get_all_vocab()  # allow rare words
    else:
        length_penalty = lambda logits, len: logits
        vocab_mask = get_all_vocab()
    
    return model.generate(
        input_ids,
        max_length=max_length,
        logits_processor=[vocab_mask, length_penalty],
        num_beams=4
    )
```

---

## Comparison Table

| Mechanism | Novelty | Complexity | Training | Interpretability | Main Conf | Paper Angle |
|-----------|---------|-----------|----------|------------------|-----------|------------|
| **Token-Only** | ⭐ | Trivial | No | Low | ⭐ | Baseline only |
| **Adapter Modules** | ⭐⭐⭐⭐ | Medium | Yes (4h) | High | ⭐⭐⭐⭐ | **Recommended** |
| **Prefix Tuning** | ⭐⭐⭐ | Medium | Yes (3h) | Medium | ⭐⭐⭐ | Parameter-efficient |
| **Auxiliary Loss** | ⭐⭐⭐ | Easy | Yes (4h) | High | ⭐⭐⭐ | Multi-task learning |
| **Constrained Decoding** | ⭐ | Easy | No | High | ⭐ | Analysis only |

---

## Recommendation for Main Conference Paper

**Best approach: Adapter Modules**

Why:
1. ✅ Novel technical contribution (adapters + difficulty control)
2. ✅ Strongest paper narrative
3. ✅ Most interpretable (can analyze what adapters learn)
4. ✅ Modular design allows extension to more difficulty levels
5. ✅ Competitive results while maintaining quality

**Experimental strategy:**
1. Implement all 5 for comparison
2. Show adapter > auxiliary loss > prefix tuning > constrained > token-only
3. Deep analysis of adapter behavior: what changes for HARD vs EASY?
4. Qualitative examples: show generated questions at each difficulty

---

## Next Steps

- [ ] Implement Adapter Modules
- [ ] Train 3 variants (baseline, diff-control with adapter, focus-span with adapter)
- [ ] Run evaluation on test set
- [ ] Compare ROUGE/BLEU across all 5 approaches
- [ ] Analyze adapter weights and visualize
- [ ] Write paper focusing on adapter contribution
