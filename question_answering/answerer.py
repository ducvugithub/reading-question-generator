"""
Multiple-choice answerers for a cascade experiment -- deliberately separate
from qa_model.py's QAModel hierarchy, which is attention-first (every
QAModel subclass must expose get_attention_distribution). Remote models
(Claude via the API or Bedrock) can never expose attention weights at all,
so forcing them into that ABC would mean stubbing an abstract method with
no real implementation. This module only asks the one question a cascade
actually needs: given a passage+question+options, which letter does this
model pick, and how confident is it?

Local decoder models (DecoderOnlyQAModel) already have this exact method
(predict_multiple_choice_answer) for the attention-entropy notebook's
correctness check -- LocalDecoderAnswerer below just adapts it to this
shared Answerer interface so it can sit in the same cascade list as the
extractive and remote answerers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Answerer(ABC):
    """One participant in a difficulty cascade: given a passage, question,
    and 4 options, picks a letter (or None if it couldn't parse one) and
    reports a confidence in [0, 1] (or None if this model/API doesn't
    expose one)."""

    @abstractmethod
    def answer(self, passage: str, question: str, options: list[str]) -> tuple[str | None, float | None]:
        """Returns (letter in {"A","B","C","D"} or None, confidence or None)."""
        ...


class LocalDecoderAnswerer(Answerer):
    """Wraps a DecoderOnlyQAModel (question_answering/qa_model.py) --
    local, open-weight, causal decoder (e.g. Qwen). Confidence is always
    populated (mean per-step token probability)."""

    def __init__(self, model):
        self.model = model  # a DecoderOnlyQAModel instance
        self.name = model.model_name

    def answer(self, passage: str, question: str, options: list[str]) -> tuple[str | None, float | None]:
        letter, confidence = self.model.predict_multiple_choice_answer(passage, question, options)
        return letter, confidence


class LocalExtractiveAnswerer(Answerer):
    """Wraps an ExtractiveQAModel (question_answering/qa_model.py) --
    local, open-weight, BERT/RoBERTa/DeBERTa-style span extractor. These
    models don't pick a letter directly -- they extract a passage SPAN --
    so this picks whichever option's text best matches the extracted
    span, via QAEvaluator.token_f1 (already used elsewhere in this project
    for exactly this kind of predicted-vs-gold text comparison). No
    calibrated confidence for the *letter* choice exists (the model's own
    confidence is over the span, not over which option matches it), so
    this always returns None for confidence."""

    def __init__(self, model):
        from question_answering.qa_evaluator import QAEvaluator

        self.model = model  # an ExtractiveQAModel instance
        self.name = model.model_name
        self._evaluator = QAEvaluator()

    def answer(self, passage: str, question: str, options: list[str]) -> tuple[str | None, float | None]:
        span, _ = self.model.predict_answer(passage, question)
        scores = [self._evaluator.token_f1(span, opt) for opt in options]
        best_idx = max(range(len(options)), key=lambda i: scores[i])
        return "ABCD"[best_idx], None


class ClaudeAPIAnswerer(Answerer):
    """Claude via the first-party Anthropic API (official `anthropic` SDK
    -- see question_answering/docs/qa_model_architecture.md for why this
    project always calls Claude through the SDK, never raw HTTP).

    model: pass the exact model ID for whichever cascade tier this
    instance represents, e.g. "claude-haiku-4-5" for a fast/cheap tier or
    "claude-opus-4-8" for a strong tier -- deliberately NOT defaulted,
    since a cascade needs multiple *different* tiers side by side, not
    one fixed default model.

    Every call is a real, billed API request -- there is no local
    inference here. Costs real money per item; check pricing before
    running this over many items."""

    def __init__(self, model: str, client=None):
        import anthropic

        self.model = model
        self.name = model
        self.client = client or anthropic.Anthropic()

    def answer(self, passage: str, question: str, options: list[str]) -> tuple[str | None, float | None]:
        prompt = (
            f"Passage: {passage}\n\nQuestion: {question}\nOptions:\n"
            f"A) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\n"
            "Answer with the letter only."
        )
        response = self.client.messages.create(
            model=self.model, max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        letter = next((c for c in text.upper() if c in "ABCD"), None)
        return letter, None  # the API doesn't expose a token-probability confidence


class BedrockConverseAnswerer(Answerer):
    """Any non-Anthropic model on Amazon Bedrock (Nova, GLM, etc.), via
    boto3's provider-agnostic bedrock-runtime `converse` API -- unlike
    Claude, these providers have no Anthropic-SDK client, so this talks
    to Bedrock directly. (Claude ALSO works through `converse`, but gave
    a verbose non-letter answer to the same prompt that the
    Anthropic-SDK-based ClaudeBedrockAnswerer answers correctly -- so
    Haiku/Opus stay on that class; this one is for the providers that
    have no alternative.)

    model: the Bedrock model ID, with cross-region inference profile
    prefix if the bare ID 404s for on-demand invocation (same caveat as
    ClaudeBedrockAnswerer) -- e.g. "eu.amazon.nova-micro-v1:0". GLM
    (zai.glm-4.7-flash) had no such profile as of 2026-09-30; pass the
    bare ID for models like that.

    aws_region required; aws_profile optional (falls back to the
    environment's default/active profile). Every call is a real, billed
    Bedrock request -- check pricing before running this over many items."""

    def __init__(self, model: str, aws_region: str, aws_profile: str | None = None, client=None):
        import boto3

        self.model = model
        self.name = model
        if client is not None:
            self.client = client
        else:
            session = boto3.Session(profile_name=aws_profile, region_name=aws_region)
            self.client = session.client("bedrock-runtime")

    def answer(self, passage: str, question: str, options: list[str]) -> tuple[str | None, float | None]:
        prompt = (
            f"Passage: {passage}\n\nQuestion: {question}\nOptions:\n"
            f"A) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\n"
            "Answer with the letter only."
        )
        response = self.client.converse(
            modelId=self.model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 8},
        )
        text = response["output"]["message"]["content"][0]["text"]
        letter = next((c for c in text.upper() if c in "ABCD"), None)
        return letter, None


class ClaudeBedrockAnswerer(Answerer):
    """Claude via Amazon Bedrock, through the Anthropic SDK's legacy
    AnthropicBedrock client (the bedrock-runtime InvokeModel path).

    The newer Mantle client (AnthropicBedrockMantle, the Messages-API
    Bedrock endpoint) is what Anthropic recommends for new code, but it
    404'd against this project's available AWS profile/region
    (fsecure-golden-retriever-ci, eu-north-1) even though the model is
    genuinely enabled there (confirmed working via both a raw boto3
    bedrock-runtime.converse() call and this legacy client) -- Mantle is
    apparently not yet available for this account/region combination, so
    this class deliberately uses the older, confirmed-working path
    instead of the recommended one.

    Bedrock model IDs need the cross-region inference profile form for
    on-demand invocation on this account, e.g.
    "eu.anthropic.claude-haiku-4-5-20251001-v1:0" -- the bare
    "anthropic.claude-haiku-4-5-20251001-v1:0" form 404s. Pass the FULL
    ID (with region prefix) as `model`; if a raw model ID 404s, check
    `aws bedrock list-inference-profiles --region <region>` for the
    correct prefixed form.

    aws_region is required (Bedrock is region-scoped); AWS credentials
    are resolved the normal boto3 way (profile/env/instance role) -- pass
    aws_profile explicitly, or leave it None to use the environment's
    default/active profile.

    Every call is a real, billed Bedrock request -- Bedrock pricing is
    partner-operated and separate from first-party API pricing (see AWS's
    Bedrock pricing page); check before running this over many items."""

    def __init__(self, model: str, aws_region: str, aws_profile: str | None = None, client=None):
        from anthropic import AnthropicBedrock

        self.model = model
        self.name = model
        self.client = client or AnthropicBedrock(aws_region=aws_region, aws_profile=aws_profile)

    def answer(self, passage: str, question: str, options: list[str]) -> tuple[str | None, float | None]:
        prompt = (
            f"Passage: {passage}\n\nQuestion: {question}\nOptions:\n"
            f"A) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\n"
            "Answer with the letter only."
        )
        response = self.client.messages.create(
            model=self.model, max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        letter = next((c for c in text.upper() if c in "ABCD"), None)
        return letter, None
