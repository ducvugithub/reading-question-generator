"""
Model-agnostic manual attention inspection -- renders a passage as
HTML with attention weights highlighted, for any QAModel subclass
(ExtractiveQAModel, GenerativeQAModel, DecoderOnlyQAModel; see
question_answering/qa_model.py). Extracted from
question_difficulty/notebooks/entropy_manual_review.ipynb so both that
notebook and question_difficulty/notebooks/attention_entropy_vs_irt_difficulty.ipynb
(decoder-only) can reuse the same visualizer instead of each defining
their own copy.
"""
from __future__ import annotations

from IPython.display import HTML, display

from question_answering.qa_model import QAModel


def render_attention_html(text: str, spans: list[tuple[int, int]], weights: list[float],
                           color: tuple[int, int, int] = (220, 20, 60)) -> str:
    """Renders `text` as HTML with each (start, end) span in `spans`
    colored by its corresponding weight in `weights`, normalized to the
    MAX weight in this distribution -- so the single most-attended span
    is fully saturated and everything else scales down from there.
    Gaps between spans (whitespace, truncated regions, and any span not
    included -- e.g. punctuation, if you pass a punctuation-filtered
    spans/weights pair -- are left plain)."""
    order = sorted(range(len(spans)), key=lambda i: spans[i][0])
    max_w = max(weights) if weights else 1.0
    out = []
    pos = 0
    r, g, b = color
    for i in order:
        s, e = spans[i]
        if s > pos:
            out.append(text[pos:s])
        alpha = (weights[i] / max_w) if max_w > 0 else 0.0
        out.append(f'<span style="background-color: rgba({r},{g},{b},{alpha:.3f})" '
                    f'title="weight={weights[i]:.4f}">{text[s:e]}</span>')
        pos = e
    if pos < len(text):
        out.append(text[pos:])
    return "".join(out)


def inspect_attention(passage: str, question: str, model: QAModel, layer: int | None = None,
                       head: int | None = None, show_token_attention: bool = True,
                       show_sentence_attention: bool = True) -> None:
    """Runs any passage+question through `model` and visualizes attention
    two ways -- per-token and per-sentence. Both the printed
    entropy_norm/pr_norm numbers AND the color maps use the "_no_punct"
    distributions (qa_model.py: punctuation/attention-sink tokens --
    periods etc. that absorb a large, near-identical share of attention
    regardless of the question -- excluded and the remainder properly
    renormalized). Raw (punctuation-included) numbers are NOT shown here
    since they were dominated by the sink artifact; see
    model.get_attention_distribution's "distribution"/"token_distribution"
    if you need them.

    layer: model-specific default if None (each QAModel subclass picks
    its own -- e.g. 11 for the 12-layer encoder models, last layer for a
    decoder-only model).
    head: None (default) averages across all attention heads. Pass an int
    (0-indexed, check model.model.config.num_attention_heads for the
    range) to look at ONE head's raw attention instead -- see
    inspect_question_token_attention for why this can matter.
    show_token_attention / show_sentence_attention: toggle each
    visualization block independently."""
    pred, conf = model.predict_answer(passage, question)
    detail = model.get_attention_distribution(passage, question, layer=layer, head=head)

    SEP = "-" * 80

    print("=" * 80)
    print(f"Question: {question!r}" + (f"  (head={head})" if head is not None else "  (heads averaged)"))
    print(f"Predicted answer: {pred!r}  (confidence={conf:.3f})")
    print(f"Sentence-level: entropy_norm={detail['entropy_norm_no_punct']:.3f}  "
          f"pr_norm={model.normalized_participation_ratio(detail['distribution_no_punct']):.3f}  "
          f"({len(detail['sentences'])} sentences)")
    print(f"Token-level:    entropy_norm={detail['tok_entropy_norm_no_punct']:.3f}  "
          f"pr_norm={model.normalized_participation_ratio(detail['token_distribution_no_punct']):.3f}  "
          f"({detail['num_tokens_no_punct']} non-punctuation tokens, of {detail['num_tokens']} total)")

    if show_token_attention:
        print(SEP)
        print(">>> Token-level attention (darker = more attention; punctuation tokens excluded):")
        is_punct = [model.is_punctuation_only(passage[s:e]) for s, e in detail["token_char_spans"]]
        spans_no_punct = [s for s, p in zip(detail["token_char_spans"], is_punct) if not p]
        display(HTML(render_attention_html(passage, spans_no_punct, detail["token_distribution_no_punct"])))

    if show_sentence_attention:
        print(SEP)
        print(">>> Sentence-level attention (darker = more attention; punctuation tokens excluded):")
        sent_spans = model.sentence_spans(passage)
        display(HTML(render_attention_html(passage, sent_spans, detail["distribution_no_punct"])))

    print("=" * 80)


def inspect_question_token_attention(passage: str, question: str, model: QAModel,
                                      layer: int | None = None, head: int | None = None) -> None:
    """For EACH token in the question, renders a SEPARATE full-text
    highlight of the passage showing that token's own attention -- not
    averaged with the other question tokens like inspect_attention does.
    Use this to find which specific question word is pulling attention
    toward an unexpected part of the passage (the aggregate view in
    inspect_attention can't distinguish this).

    Same convention as inspect_attention: only punctuation is excluded,
    stopwords are kept -- so both views stay directly comparable, and
    negation words ("not"/"no") are never silently dropped.

    layer: model-specific default if None, same as inspect_attention.
    head: None (default) averages across all attention heads, same as
    inspect_attention. Pass an int to look at ONE head's raw attention
    instead -- averaging all heads can wash out per-question-token
    differentiation a single specialized head might actually show."""
    result = model.get_per_question_token_attention(passage, question, layer=layer, head=head)

    print("=" * 80)
    print(f"Question: {question!r}" + (f"  (head={head})" if head is not None else "  (heads averaged)"))
    for tok, dist in zip(result["question_tokens"], result["distributions"]):
        print("-" * 80)
        print(f">>> Question token: {tok!r}")
        display(HTML(render_attention_html(passage, result["token_char_spans"], dist)))
    print("=" * 80)
