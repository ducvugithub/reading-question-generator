# Difficulty-Controllable Question Generation: Steering Mechanisms

Five approaches to control question generation difficulty in T5-based QG models.

---

## ⚠️ Update (2026-11-13): as-built FiLM mechanism + loading bug found/fixed

The adapter design below (Section 2) was superseded by a simpler as-built
version in `question_generation/scripts/diff_film_model.py` — no bottleneck
MLP, just a per-layer affine transform.

**One-line summary: FiLM-per-difficulty-class, selectively updated.** The
embedding lookup means each difficulty's γ/β only receives gradient from
examples tagged with that difficulty — gradient never crosses between
classes. The shared encoder/decoder, by contrast, gets gradient from every
example. This selective routing — not the loss function itself — is what
forces specialization.

- **Two parallel paths, merging at each decoder layer.** Passage →
  encoder → cross-attention into every decoder layer (normal T5). Difficulty
  tag → `difficulty_film` embedding lookup (3 rows: EASY/MEDIUM/HARD) → γ,β
  vectors per decoder layer. The embedding's only input is the difficulty id
  — never the passage.
- **Applied via forward hook, after each decoder layer:**
  `new_output = layer_output * γ + β`. γ starts at 1, β at 0 (identity), so
  an untrained model behaves exactly like plain T5.
- **One table, N layers × 2 (γ,β) × d_model each**, shared globally across
  all passages — a class's row is only ever touched by examples of that
  class.
- **Why training specializes it correctly:** the embedding lookup is a hard
  index. Backprop only routes gradient to the row that was actually
  selected for a given example — EASY's row only gets updated by EASY
  examples, MEDIUM's only by MEDIUM's, etc. Decoder/encoder weights, by
  contrast, get gradient from every example regardless of tag. That
  selective routing — not anything in the loss function itself — is what
  forces each difficulty's γ/β to specialize while the shared backbone
  stays general-purpose.
- **Bug found 2026-11-13:** `from_pretrained` unconditionally zeroed
  `difficulty_film.weight` after every load — including loading the
  model's own already-trained checkpoint. Checkpoints had real learned
  values (abs mean ~0.028), but inference always discarded them and ran
  with the identity transform, so EASY/MEDIUM/HARD generated identical
  output. Same bug existed in the earlier additive-bias model
  (`diff_cond_model.py`, "v0") — but v0 re-randomizes with noise
  (`std=0.02`) instead of zeroing, so its "weak but real" differentiation
  on ~half of test passages was actually just random noise occasionally
  breaking beam search ties, not learned difficulty signal either.
- **Fix:** only re-init when the checkpoint genuinely lacks the key (fresh
  base-model load); load it as-is otherwise. Also removed weight decay on
  `difficulty_film` (was pulling γ/β back toward identity every step) and
  raised its LR multiplier 20x → 50x, and raised early-stopping patience
  3 → 8 (its signal moves slower than the overall seq2seq loss that
  normally triggers the stop).
- **Status:** v3 (bug-fixed + above tuning) trained to step 2500/epoch~5
  on Roihu (job 3057523), checkpoint pulled down — spot-check pending.

### Results across versions (100-passage sample, `eval_film_diff_quality.py`)

Collapse rate = how often EASY/MEDIUM/HARD generate the same question for one
passage; ROUGE-L is against the real gold question for whichever bucket
actually has one in the test set.

| Version | Change | Full collapse | Partial (2/3) | Fully distinct | ROUGE-L (EASY/MED/HARD) |
|---|---|---|---|---|---|
| v2 | loading bug fixed only | 62% | 31% | 7% | .223/.233/.246 |
| v3 | + no weight decay on FiLM, LR mult 20x→50x, patience 3→8 | 29% | 54% | 17% | .241/.225/.247 |
| v4 | + MC-dependent stems ("which of the following is TRUE") filtered from training data | 34% | 43% | 23% | .241/.226/.269 |
| v5 | + LR mult 50x→100x, patience 8→16 | **17%** | 42% | **41%** | .243/**.207**/.243 |

v5 is the current best: full collapse nearly halved vs v4, fully-distinct
nearly doubled. EASY/HARD quality held steady; MEDIUM dipped (.226→.207) —
an isolated tradeoff, not a broad regression. Checkpoint:
`question_generation/models/qg_10k_blind_film_v5/final` (trained on
`data/qg_v3/diff-control-irt-10k-blind`, job 3064000, step 4250/8660).

Pushed further: v6 (LR mult 100x→150x, patience 16→24, same data) —
job 3066282, step 6250/8660, train_loss 0.764.

| v6 | + LR mult 150x, patience 24 | **11%** | 46% | **43%** | .238/.211/.227 |

v6 vs v5: collapse keeps dropping but with diminishing returns (17%→11%,
smaller jump than v4→v5's 34%→17%), and the quality cost spreads across
EASY/HARD this time instead of concentrating in MEDIUM (which actually
improved, .207→.211). **Locked in as current best** — qualitative spot-check
(8 passages, separate seed) confirms real passage-grounded differentiation
on clear cases (e.g. "Macao": EASY/MEDIUM/HARD each ask a genuinely
different angle), with remaining collapse now mostly generic-but-valid
fallback phrasings ("What can be inferred from the passage?", "What's the
best title for the passage?") rather than the old garbage or
options-dependent junk.

**Stopping the LR/patience-push approach here** — diminishing returns,
next lever (if pursued) should be the direct anti-collapse loss term
(penalize similarity between the three difficulty-conditioned outputs for
the same passage) rather than pushing these two knobs further.

Checkpoints: `question_generation/models/qg_10k_blind_film_v5/final`,
`question_generation/models/qg_10k_blind_film_v6/final` (both on
`data/qg_v3/diff-control-irt-10k-blind`).

### Anti-collapse loss term (v7, v8) — `train_seq2seq_film_anticollapse.py`

Implemented the lever flagged above: for every training example, encode once
and decode the SAME teacher-forced target sequence three times (once per
difficulty class), then penalize the mean pairwise cosine similarity between
the three resulting token-probability distributions, added to the normal
cross-entropy loss (weight 0.3). Only the "own" class's `.loss` is real
supervision — the other two classes' logits are used purely for the
diversity penalty, never trained toward this example's actual target text.

| Version | Change | Full collapse | Partial | Fully distinct | ROUGE-L (E/M/H) |
|---|---|---|---|---|---|
| v7 | anti-collapse loss alone, same LR/patience as v5 (100x/16) | 16% | 46% | 38% | .228/.232/.257 |
| **v8** | anti-collapse loss + v6's LR/patience (150x/24) | **9%** | 39% | **52%** | .228/**.229**/**.257** |

v7 alone didn't beat v6 on collapse rate, but clearly improved quality
(best MEDIUM/HARD ROUGE-L yet) — the loss term doesn't concentrate enough
force on the stubborn passages by itself. Combined with the stronger
LR/patience push (v8), the two levers compound rather than conflict: new
best on every structural metric (9% collapse, 52% fully-distinct) while
keeping v7's quality gains on MEDIUM/HARD. **v8 is the new best model,
locked in.**

Checkpoints: `question_generation/models/qg_10k_blind_film_v7/final`,
`question_generation/models/qg_10k_blind_film_v8/final` (both on
`data/qg_v3/diff-control-irt-10k-blind`, script
`train_seq2seq_film_anticollapse.py`).

---

## ⚠️ Update: token-only conditioning empirically tested, doesn't steer

Section 1 below was originally speculative ("Current Implementation"). It's
since been trained (`diff-control-race`, flan-t5-base) and tested directly: a
forced-token generation test (same passage, `<EASY>`/`<MEDIUM>`/`<HARD>` forced
in turn) showed the token producing no more variation than `baseline-race`'s
pure sampling noise floor (which never even sees the token). Most likely root
cause: every RACE passage belongs to exactly one difficulty subset, so the
token is perfectly collinear with passage-level style/vocabulary during
training — the model has no forcing pressure to learn genuine token-based
steering over just reading passage style. Full writeup:
`question_generation/docs/training_details.md`'s "Known limitation" section.

**This same confound also applies to Section 2 (Adapter Modules) as
originally scoped below** — a discrete `Adapter_EASY`/`Adapter_MEDIUM`/
`Adapter_HARD` selected by the same subset-inherited label would hit the
identical problem; changing the conditioning *mechanism* doesn't fix a *data*
confound. Current direction: extract a genuine per-question difficulty signal
(not passage-inherited) and use it to condition a *continuous* adapter via
FiLM-style modulation, rather than discrete per-class adapter modules. See
`question_difficulty/docs/cognitive_difficulty_estimation.md`'s "Method 4" for
the signal-extraction plan and the `DifficultySignal` interface design.

---

## 1. Token-Only Conditioning (tested, does not steer — see update above)

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
- ✅ Trained (`diff-control-race`, flan-t5-base/flan-t5-large)

**Cons:**
- ❌ **Empirically confirmed not to produce measurable steering** (see update above)
- ❌ Feels like "prompt engineering" (weak novelty for main conferences)
- ❌ Limited expressiveness (tokens compete with passage for attention)
- ❌ Hard to analyze what the model learned

**Implementation complexity:** Trivial

**Paper angle:** Not suitable alone; use as baseline / negative result

**Estimated scores:**
- Main conference: ⭐☆☆☆☆ (too simple, and doesn't work)
- BEA: ⭐⭐⭐☆☆ (solid for workshop, especially as a documented negative result)

---

## 2. Adapter Modules ⭐ RECOMMENDED (needs continuous conditioning, not discrete — see update above)

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
- ⚠️ **As originally scoped (discrete per-class adapter selected by the
  subset-inherited label), inherits the exact same passage-confound problem
  as token-only conditioning** — see update at top of doc

**Implementation complexity:** Medium

**Paper angle:**
- "Adapter-based difficulty control for controllable QG"
- Analyze adapter weights to understand what changes with difficulty
- Visualize: harder questions → what encoder changes are needed?

**Estimated scores:**
- Main conference: ⭐⭐⭐⭐☆ (solid technical novelty, contingent on solving the data confound)
- BEA: ⭐⭐⭐⭐⭐ (perfect for workshop)

**Original discrete implementation sketch (superseded — see continuous/FiLM
version below):**
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

**Current direction — continuous FiLM-conditioned adapter:** one shared
adapter module, modulated by a continuous difficulty representation (from a
per-question difficulty encoder, see
`question_difficulty/docs/cognitive_difficulty_estimation.md`'s "Method 4")
rather than discrete per-class modules:

```python
class FiLMConditionedAdapter(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim, difficulty_dim):
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.to_film = nn.Linear(difficulty_dim, 2 * hidden_dim)  # -> gamma, beta

    def forward(self, hidden_states, difficulty_repr):
        gamma, beta = self.to_film(difficulty_repr).chunk(2, dim=-1)
        adapted = self.up(F.relu(self.down(hidden_states)))
        return hidden_states + gamma * adapted + beta
```

Training: for each real (passage, question) pair, run the *target* question
through the difficulty encoder to get its representation, feed it into the
adapter alongside the passage. Inference: no target question exists yet, so
use a precomputed representative code per difficulty level (e.g. the average
encoder output over many real EASY/MEDIUM/HARD questions) instead.

### Validated signal source (2026-09-01)

Ran `question_difficulty/scripts/validate_difficulty_signals.py` at increasing
scale (n=3, 30, 102 passages, balanced across EASY/MEDIUM/HARD from n=102
onward) — see `question_difficulty/docs/cognitive_difficulty_estimation.md`'s
"Method 4" for the full methodology and per-layer numbers. Findings:

- **Sentence-level entropy at layer 6** of `deepset/roberta-base-squad2`
  consistently shows the highest within-passage mean spread across all 3
  runs (0.187±0.142 at n=102, clear gap over the next-highest layer).
- **Token-level entropy at layer 11** (the last layer) took over as the
  token-level leader once HARD-level passages were properly included
  (n=102) — layer 6 was only competitive at token-level with the earlier,
  incomplete n=30 sample.
- Manual qualitative review of ~20 real (question, answer) pairs at the
  entropy extremes confirmed a sensible pattern independent of the
  statistics: high-entropy questions tend to be whole-passage synthesis
  questions ("best title", "what did X learn") vs. low-entropy questions
  tending to be single-sentence factual lookups — holds up within multiple
  individual passages, not just in aggregate.
- Non-attention text signal `answer_extractiveness_overlap` stayed stable
  across all 3 runs (0.579 → 0.625 → 0.558) — a reasonably trustworthy
  secondary signal, not currently wired into the adapter design below.

### Forms of the signal considered

| Form | Fixed-size? | Needs training? | Notes |
|---|---|---|---|
| Scalar entropy (layer 6 or 11) | Yes (1 number) | No — pure computation | **Starting point** — simplest, already validated above |
| Weighted-average passage embedding (`Σ attention_i × token_embedding_i`) | Yes (matches QA model's hidden dim) | No — pure computation | Richer (keeps *where*, not just *how spread*), no new trainable component |
| Learned pooling network (RNN/small attention-pooling over the raw distribution) | Yes (by construction) | **Yes** — trained jointly with the adapter via the same generation loss | Most expressive, most complexity/risk |
| Binary/multi-tier thresholded span markup (`<HIGH_FOCUS>`/`<MED_FOCUS>` tags in passage text) | N/A — not a vector, it's text markup | No | Reuses `focus-control-hotpot`'s existing architecture instead of a new adapter; more human-interpretable/controllable at inference (see below) |

### The train/inference asymmetry (applies to every form above)

Training always has the real target question available, so the QA model can
compute a real signal from it. At inference there is no target question yet
(that's what's being generated), so the QA model cannot run. Fix, same
principle regardless of signal form: **use a representative value/vector
borrowed or averaged from real training examples**, not something
hand-synthesized from scratch — e.g. average the real signal across several
training questions that read as "hard" in the manual review, rather than
inventing an arbitrary value/weighting.

### Alternative worth revisiting: multi-tier focus-span markup

Thresholding the raw attention distribution into discrete tiers
(`attention > 0.25 → <HIGH_FOCUS>`, `0.10-0.25 → <MED_FOCUS>`, else
unmarked) and inserting those tags directly into the passage text turns this
into a variant of the *already-built* `focus-control-hotpot` architecture —
no adapter needed at all. Advantages over the FiLM/vector approach:
more human-interpretable and controllable at inference (choose how many
spans at which tier, rather than picking a continuous vector), and reuses
working infrastructure. Downside: loses fine-grained weighting (discrete
tiers, not continuous), and still needs an inference-time decision rule for
how many spans/tiers to mark — just a simpler decision than picking FiLM
vector values.

**Plan: start with scalar entropy + FiLM adapter (simplest) to test whether
difficulty control is achievable at all before investing in the richer
vector forms or the multi-tier markup alternative.**

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
- ⚠️ Same passage-confound risk as token-only/discrete-adapter if keyed on
  the subset-inherited label instead of a genuine per-question signal

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
- ⚠️ Same passage-confound risk if the classification target is the
  subset-inherited label

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
| **Token-Only** | ⭐ | Trivial | Yes — tested, doesn't steer | Low | ⭐ | Baseline / negative result |
| **Adapter Modules (continuous/FiLM)** | ⭐⭐⭐⭐ | Medium | Not yet — blocked on per-question signal | High | ⭐⭐⭐⭐ | **Recommended, pending signal extraction** |
| **Prefix Tuning** | ⭐⭐⭐ | Medium | Yes (3h) | Medium | ⭐⭐⭐ | Parameter-efficient |
| **Auxiliary Loss** | ⭐⭐⭐ | Easy | Yes (4h) | High | ⭐⭐⭐ | Multi-task learning |
| **Constrained Decoding** | ⭐ | Easy | No | High | ⭐ | Analysis only |

---

## Recommendation

**Best approach: Adapter Modules, continuous/FiLM-conditioned on a genuine
per-question difficulty signal** (not the discrete per-class version
originally sketched, which shares token-only's data confound).

Why:
1. ✅ Novel technical contribution (adapters + difficulty control)
2. ✅ Strongest paper narrative
3. ✅ Most interpretable (can analyze what adapters learn)
4. ✅ Modular design allows extension to more difficulty levels
5. ⚠️ Contingent on validating a real per-question difficulty signal exists
   first — see `question_difficulty/docs/cognitive_difficulty_estimation.md`'s
   "Method 4"

**Current status / next steps:**
- [x] Implement token-only conditioning (`diff-control-race`) — trained, tested, doesn't steer
- [x] Validate per-question difficulty signal exists — confirmed via
      within-passage variance (statistical) and manual review (qualitative);
      see "Validated signal source" above. Layer 6 (sentence-level) and
      layer 11 (token-level) of `roberta-base-squad2` both hold up.
- [ ] Scale signal extraction from the ~100-passage validation sample to the
      full ~86K RACE training questions
- [ ] Implement the FiLM-conditioned adapter, **starting with scalar entropy**
      (simplest form) — frozen `baseline-race` checkpoint, adapter-only
      training, single standard cross-entropy loss
- [ ] Re-run the same-passage forced-signal generation test to check for real steering
- [ ] If scalar entropy doesn't steer well: try the weighted-embedding form,
      or the multi-tier focus-span markup alternative (reuses
      `focus-control-hotpot` infrastructure, more interpretable/controllable
      at inference — see "Alternative worth revisiting" above)
- [ ] Compare against prefix tuning / auxiliary loss if all adapter variants fail
