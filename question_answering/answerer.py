"""
Multiple-choice answerers for a cascade experiment -- deliberately separate
from qa_model.py's QAModel hierarchy, which is attention-first (every
QAModel subclass must expose get_attention_distribution). Remote models
(Claude via the API or Bedrock) can never expose attention weights at all,
so forcing them into that ABC would mean stubbing an abstract method with
no real implementation. This module only asks the one question a cascade
actually needs: given a passage+question+options, which letter does this
model pick, and how confident is it?

LocalDecoderAnswerer and LocalExtractiveAnswerer are both BLIND by design:
neither sees the 4 options when producing its answer (free-text generation
or span extraction), and both map their free-text answer to the closest
option the same way, via semantic_matcher.best_option_index -- see each
class's docstring for why this replaced showing options directly to the
decoder (predict_multiple_choice_answer, still used by DecoderOnlyQAModel
for the attention-entropy notebook's own correctness check, just no longer
by this module).
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
    local, open-weight, causal decoder (e.g. Qwen). BLIND by design: the
    model never sees the 4 options, it only generates a free-text answer
    (predict_answer, same method GenerativeQAModel/ExtractiveQAModel are
    scored with elsewhere) -- the resulting text is then mapped to the
    closest option via semantic_matcher.best_option_index, exactly the
    same mapping LocalExtractiveAnswerer uses below.

    This replaces the earlier design (predict_multiple_choice_answer,
    which showed all 4 options directly in the prompt and asked for a
    letter) -- that gave decoders a structural advantage over
    LocalExtractiveAnswerer (which never saw the options either, but was
    scored via literal token-overlap against them). The asymmetry
    confounded the cascade's "which questions are hard" signal with "which
    answerer style this question favors" rather than genuine difficulty
    (see question_difficulty cascade investigation, 2026-10 -- extractive
    models looked artificially weak specifically on main-idea/paraphrase-
    style questions as a result). Confidence is still the mean per-step
    token probability of the free-text generation."""

    def __init__(self, model):
        self.model = model  # a DecoderOnlyQAModel instance
        self.name = model.model_name
        self.last_raw_text: str | None = None  # the pre-mapping free-text answer, for
        # auditing/re-scoring the semantic match later without re-running the model --
        # see run_cascade()'s results["raw_text"], added 2026-11-11 after finding the
        # mapping can be noisy on short/paraphrase-heavy options (e.g. titles).

    def answer(self, passage: str, question: str, options: list[str]) -> tuple[str | None, float | None]:
        from question_answering.semantic_matcher import best_option_index

        # 32 tokens + the "one short sentence" prompt keeps the answer's
        # key content near the start (less truncation risk than a bare
        # "answer concisely") while still bounding wall-clock cost --
        # autoregressive decoding is sequential, see
        # DecoderOnlyQAModel.predict_answer's docstring.
        text, confidence = self.model.predict_answer(passage, question, max_new_tokens=32)
        self.last_raw_text = text
        if not text.strip():
            return None, confidence
        best_idx = best_option_index(text, options, question=question)
        return "ABCD"[best_idx], confidence


class LocalExtractiveAnswerer(Answerer):
    """Wraps an ExtractiveQAModel (question_answering/qa_model.py) --
    local, open-weight, BERT/RoBERTa/DeBERTa-style span extractor. These
    models don't pick a letter directly -- they extract a passage SPAN --
    so this picks whichever option's text best matches the extracted
    span, via semantic_matcher.best_option_index (embedding cosine
    similarity, not literal token overlap -- token_f1 was tried first and
    systematically penalized a correct-but-paraphrased option, which is
    common for main-idea/inference questions whose gold option is rarely a
    verbatim passage span; see semantic_matcher's module docstring). No
    calibrated confidence for the *letter* choice exists (the model's own
    confidence is over the span, not over which option matches it), so
    this always returns None for confidence."""

    def __init__(self, model):
        self.model = model  # an ExtractiveQAModel instance
        self.name = model.model_name
        self.last_raw_text: str | None = None  # see LocalDecoderAnswerer's matching docstring

    def answer(self, passage: str, question: str, options: list[str]) -> tuple[str | None, float | None]:
        from question_answering.semantic_matcher import best_option_index

        span, _ = self.model.predict_answer(passage, question)
        self.last_raw_text = span
        if not span.strip():
            return None, None
        best_idx = best_option_index(span, options, question=question)
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


class LayaAnswerer(Answerer):
    """convaiinnovations/laya -- local, non-autoregressive "System 1" decision
    model (ModernBERT-large backbone, 421M params, Apache 2.0, pip package
    `laya`). Instead of generating text, it scores a typed "choice" question
    directly against the option texts as criteria and returns calibrated
    per-option probabilities in a single forward pass (~tens of ms on CPU
    once warm). No API cost, no cloud call.

    Tried manually on 5-7 sample items (2026-10-02): picked the correct
    option on only ~2/7, i.e. close to the 25% chance floor for 4-option MC
    -- plus a startup warning that this checkpoint "ships invalid
    temperatures", so its reported `answer_confidence` is flagged
    uncalibrated on top of that. 512-token context was NOT the limiting
    factor in practice (a 635-word passage + question + options loaded fine,
    no truncation) -- the `laya-multilingual` checkpoint (mmBERT-base, 322M,
    1024 ctx) was tried for the longer-context headroom and did WORSE, not
    better, on accuracy.

    Net: out of the box, Laya does not read-comprehend at a useful level for
    this task -- it would likely need fine-tuning on reading-comprehension
    MC data (RACE/OneStopQA, or similar) before being a meaningful cascade
    rung. Left here, wired but NOT added to CANDIDATES in
    run_answerer_cascade.py, in case that's worth doing later."""

    def __init__(self, repo: str = "convaiinnovations/laya"):
        import laya

        self.repo = repo
        self.name = repo
        self.agent = laya.load(repo)

    def answer(self, passage: str, question: str, options: list[str]) -> tuple[str | None, float | None]:
        letters = ["A", "B", "C", "D"][: len(options)]
        criteria = {letters[i]: options[i] for i in range(len(options))}
        state = f"Passage: {passage}\n\nQuestion: {question}"
        questions = {
            "answer": {
                "type": "choice",
                "instructions": "Which option correctly answers the question about the passage?",
                "criteria": criteria,
            }
        }
        result = self.agent.predict(state, questions)
        answer = result["answers"]["answer"]
        return answer["choice"], answer["answer_confidence"]


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


class BlindClaudeBedrockAnswerer(ClaudeBedrockAnswerer):
    """BLIND variant of ClaudeBedrockAnswerer: never shows the model the 4
    options. Free-text-generates an answer (same style as
    LocalDecoderAnswerer) and maps it to the closest option via
    semantic_matcher.best_option_index.

    Exists because plain ClaudeBedrockAnswerer shows all 4 options
    directly in the prompt ("Answer with the letter only") -- a structural
    MCQ advantage the local blind answerers (LocalDecoderAnswerer,
    LocalExtractiveAnswerer) don't have. Mixing the options-shown variant
    into an otherwise-blind cascade (as run_answerer_cascade_race_pool_blind.job
    intends) would bias both the correctness signal and the IRT ability
    estimates toward "good at MCQ elimination" rather than genuine reading
    comprehension -- found 2026-11-11 while auditing the cascade before
    adding Haiku as a candidate, see cascade_common.py's stem-filter
    comment for the related quality-audit context.

    Use this (not ClaudeBedrockAnswerer) for any candidate meant to run
    inside the blind cascade."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_raw_text: str | None = None  # see LocalDecoderAnswerer's matching docstring --
        # saved so a future matching-logic change can re-score without a new (billed) Bedrock call.

    def _llm_tiebreak(self, text: str, options: list[str]) -> int:
        """semantic_matcher's tier-4 llm_tiebreak_fn -- reuses this
        instance's own Bedrock client/credentials rather than duplicating
        AWS setup in semantic_matcher.py (which has no AWS dependency
        otherwise, since LocalDecoderAnswerer/LocalExtractiveAnswerer run
        offline on Roihu with no Bedrock access). See
        question_answering/docs/semantic_matcher_tiers.md -- tested at
        67.1% accuracy on the ambiguous slice this fires for, vs 38.5% for
        the embedding tier alone."""
        prompt = (
            f"A model answered a reading-comprehension question with this free-text answer:\n"
            f'"{text}"\n\n'
            f"Which of these 4 options does the answer most closely match in meaning?\n"
            f"A) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\n\n"
            "Respond with just the letter (A, B, C, or D)."
        )
        response = self.client.messages.create(
            model=self.model, max_tokens=4,
            messages=[{"role": "user", "content": prompt}],
        )
        reply = next((b.text for b in response.content if b.type == "text"), "")
        letter = next((c for c in reply.upper() if c in "ABCD"), None)
        return "ABCD".index(letter) if letter else 0

    def answer(self, passage: str, question: str, options: list[str]) -> tuple[str | None, float | None]:
        from question_answering.semantic_matcher import best_option_index

        prompt = (
            f"Passage: {passage}\n\nQuestion: {question}\n\n"
            "Answer the question using only information from the passage. "
            "Respond with just the answer, one short sentence, no preamble."
        )
        response = self.client.messages.create(
            model=self.model, max_tokens=32,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        self.last_raw_text = text
        if not text.strip():
            return None, None
        best_idx = best_option_index(text, options, question=question, llm_tiebreak_fn=self._llm_tiebreak)
        return "ABCD"[best_idx], None
