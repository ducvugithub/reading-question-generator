from __future__ import annotations

import json
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage

_PRONOUNS = frozenset({
    "i", "me", "my", "we", "our", "us",
    "he", "him", "his", "she", "her",
    "it", "its", "they", "them", "their",
    "this", "that", "these", "those",
})

COGNITIVE_LABELS = ["easy", "medium", "hard"]

_EASY_THRESHOLD = 1 / 3
_HARD_THRESHOLD = 2 / 3

# Question word → reasoning demand score [0, 1]
_QUESTION_WORD_SCORE: dict[str, float] = {
    "when":  0.1,
    "where": 0.15,
    "who":   0.2,
    "what":  0.25,
    "which": 0.35,
    "how many": 0.4,
    "how much": 0.4,
    "how":   0.75,
    "why":   0.9,
}


def _label(score: float) -> str:
    if score < _EASY_THRESHOLD:
        return "easy"
    if score < _HARD_THRESHOLD:
        return "medium"
    return "hard"


class CognitiveDifficultyEstimator(ABC):
    """
    Estimates cognitive difficulty of a question given its KG and answer.

    Cognitive difficulty reflects reasoning demand:
      easy   — direct factual recall (answer explicitly in passage)
      medium — inference, multi-fact connection, or pronoun resolution
      hard   — causal/procedural reasoning, abstract thinking

    Works post-hoc from serialised JSONL fields (kg_raw, kg_coref as [s,r,o] lists).
    """

    @abstractmethod
    def score(
        self,
        question: str,
        answer: str,
        kg_raw: list[list[str]],
        kg_coref: list[list[str]] | None = None,
    ) -> float:
        """Return cognitive difficulty score [0, 1]."""

    def estimate(
        self,
        question: str,
        answer: str,
        kg_raw: list[list[str]],
        kg_coref: list[list[str]] | None = None,
    ) -> dict:
        s = self.score(question, answer, kg_raw, kg_coref)
        return {"score": round(s, 4), "label": _label(s)}


class GraphCognitiveDifficultyEstimator(CognitiveDifficultyEstimator):
    """
    Rule-based estimator using question word + KG structure.

    Four signals:
      s_qtype    — reasoning demand implied by the question word (why/how > what/who > when/where)
      s_coref    — fraction of answer-covering raw triples whose subject/object is a pronoun
      s_coverage — binary: 0 if answer found in KG, 1 if not found
      s_density  — KG density as a proxy for passage complexity
    """

    _WEIGHTS = {"qtype": 0.45, "coref": 0.30, "coverage": 0.15, "density": 0.10}

    def score(
        self,
        question: str,
        answer: str,
        kg_raw: list[list[str]],
        kg_coref: list[list[str]] | None = None,
    ) -> float:
        s = (
            self._WEIGHTS["qtype"]    * self._qtype_score(question)
            + self._WEIGHTS["coref"]    * self._coref_score(answer, kg_raw, kg_coref)
            + self._WEIGHTS["coverage"] * self._coverage_score(answer, kg_raw)
            + self._WEIGHTS["density"]  * self._density_score(kg_raw)
        )
        return min(s, 1.0)

    def _qtype_score(self, question: str) -> float:
        q = question.lower().strip()
        for phrase, s in sorted(_QUESTION_WORD_SCORE.items(), key=lambda x: -len(x[0])):
            if q.startswith(phrase):
                return s
        return 0.5

    def _coref_score(self, answer: str, kg_raw: list[list[str]], kg_coref: list[list[str]] | None) -> float:
        """Fraction of answer-covering raw triples that have a pronoun subject/object.

        If the triple that contains the answer uses 'she/it/they' as subject, the
        reader must resolve the pronoun to understand who the fact belongs to —
        i.e. the question requires coreference. Passage-level pronoun counts are
        ignored; only the triples directly relevant to the answer are checked.
        """
        if not kg_coref or kg_raw == kg_coref:
            return 0.0
        answer_l = answer.lower()
        covering = [
            t for t in kg_raw
            if answer_l in t[0].lower() or answer_l in t[2].lower()
        ]
        if not covering:
            return 0.0
        pronoun_covering = [
            t for t in covering
            if t[0].lower() in _PRONOUNS or t[2].lower() in _PRONOUNS
        ]
        return len(pronoun_covering) / len(covering)

    def _coverage_score(self, answer: str, kg_raw: list[list[str]]) -> float:
        """Binary: 0 if the answer string appears in any KG triple, 1 if absent.

        Absent means the reader cannot retrieve the answer from explicit KG facts
        and must infer or synthesise it — the hardest case.
        """
        answer_l = answer.lower()
        for t in kg_raw:
            if answer_l in t[0].lower() or answer_l in t[2].lower():
                return 0.0
        return 1.0

    def _density_score(self, kg_raw: list[list[str]]) -> float:
        return min(len(kg_raw) / 15.0, 1.0)


class LLMCognitiveDifficultyJudge(CognitiveDifficultyEstimator):
    """
    LLM-as-judge producing a structured difficulty assessment stored as `llm_diff_judge`.

    Four fields, two passage-level (fixed per passage) and two question-level:
      passage_readability    — structural complexity: sentence length, discourse, cohesion
      passage_vocab_diff     — lexical difficulty: rare/technical/domain words in the passage
      question_cognitive_diff — reasoning demand: question type, explicitness, coreference, depth
      question_vocab_diff    — lexical difficulty of the question wording itself

    The primary training control signal is question_cognitive_diff.
    question_vocab_diff is secondary. Passage fields are metadata.

    Use judge() to get the full llm_diff_judge dict.
    score()/estimate() return question_cognitive_diff for compatibility with the ABC.
    """

    _SYSTEM = """\
You are an expert in educational assessment and reading comprehension difficulty.
Given a passage, a question, its answer, and knowledge graph triples extracted from the passage,
produce a structured difficulty assessment across four independent dimensions.

━━━ PASSAGE-LEVEL (evaluate the passage text) ━━━

1. passage_readability (0.0–1.0)
   How complex is the passage structure — sentence length, subordinate clauses, discourse cohesion?
   0.0 = short, simple sentences, direct narrative
   0.5 = moderate complexity, some embedded clauses
   1.0 = long, dense, heavily nested or abstract prose

2. passage_vocab_diff (0.0–1.0)
   How rare or domain-specific is the vocabulary in the passage?
   0.0 = everyday common words only
   0.5 = some technical or low-frequency words
   1.0 = highly specialised, rare, or domain-specific throughout

━━━ QUESTION-LEVEL (evaluate the question + answer pair) ━━━

3. question_cognitive_diff (0.0–1.0)
   How demanding is the reasoning required to answer?
   Consider: question word (why/how > what/who > when/where), whether the answer is
   explicitly in the passage or must be inferred, how many reasoning steps are needed,
   and whether pronoun/coreference resolution is required.
   0.0 = single direct lookup, answer word-for-word in passage, no pronouns
   0.5 = two-step reasoning, or one pronoun to resolve, or answer combines two facts
   1.0 = causal/procedural reasoning, deep inference, answer absent from passage

4. question_vocab_diff (0.0–1.0)
   How rare or technical are the words in the question itself?
   0.0 = question uses only common words
   0.5 = one or two technical/uncommon terms
   1.0 = question is phrased in specialised or academic vocabulary

Think step by step, then output a single JSON object."""

    _USER = """\
Question : {question}
Answer   : {answer}
Passage  : {passage}
KG triples (subject | relation | object):
{triples}

Reply with ONLY valid JSON — no prose, no markdown fences:
{{
  "reasoning": "<3-4 sentences covering passage structure, answer explicitness, reasoning steps needed>",
  "passage_readability":     <float 0-1>,
  "passage_vocab_diff":      <float 0-1>,
  "question_cognitive_diff": <float 0-1>,
  "question_vocab_diff":     <float 0-1>
}}"""

    def __init__(self, llm) -> None:
        """
        Args:
            llm: Any LangChain BaseChatModel — e.g. ChatBedrockConverse, ChatAnthropic.
                 Create via the project's llm_factory or directly:
                   from langchain_aws import ChatBedrockConverse
                   llm = ChatBedrockConverse(
                       model="anthropic.claude-haiku-4-5-20251001-v1:0",
                       region_name="us-east-1",
                       max_tokens=400,
                       temperature=0.0,
                   )
        """
        self._llm = llm

    def score(
        self,
        question: str,
        answer: str,
        kg_raw: list[list[str]],
        kg_coref: list[list[str]] | None = None,
        passage: str = "",
    ) -> float:
        return self.judge(question, answer, kg_raw, kg_coref, passage)["question_cognitive_diff"]

    def estimate(
        self,
        question: str,
        answer: str,
        kg_raw: list[list[str]],
        kg_coref: list[list[str]] | None = None,
        passage: str = "",
    ) -> dict:
        s = self.judge(question, answer, kg_raw, kg_coref, passage)["question_cognitive_diff"]
        return {"score": s, "label": _label(s)}

    def judge(
        self,
        question: str,
        answer: str,
        kg_raw: list[list[str]],
        kg_coref: list[list[str]] | None = None,
        passage: str = "",
    ) -> dict:
        """Return the full llm_diff_judge dict with all four fields."""
        triples = kg_coref or kg_raw or []
        triples_str = "\n".join(f"  {t[0]} | {t[1]} | {t[2]}" for t in triples) or "  (none)"
        user_msg = self._USER.format(
            question=question,
            answer=answer,
            passage=passage,
            triples=triples_str,
        )
        response = self._llm.invoke([
            SystemMessage(content=self._SYSTEM),
            HumanMessage(content=user_msg),
        ])
        from question_generation.difficulty.annotator import _extract_text, _parse_json
        return _parse_json(_extract_text(response))

    def batch_judge(
        self,
        records: list[dict],
        passage_key: str = "passage",
        question_key: str = "question",
        answer_key: str = "answer",
        kg_raw_key: str = "kg_raw",
        kg_coref_key: str = "kg_coref",
        on_error: str = "skip",
    ) -> list[dict | None]:
        """Annotate a list of dataset records."""
        results = []
        for rec in records:
            try:
                results.append(self.judge(
                    question=rec[question_key],
                    answer=rec[answer_key],
                    kg_raw=rec.get(kg_raw_key) or [],
                    kg_coref=rec.get(kg_coref_key),
                    passage=rec.get(passage_key, ""),
                ))
            except Exception:
                if on_error == "skip":
                    results.append(None)
                else:
                    raise
        return results


# Backwards-compat alias
LLMCognitiveDifficultyEstimator = LLMCognitiveDifficultyJudge
