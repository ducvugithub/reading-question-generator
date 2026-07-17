from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

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


class LLMCognitiveDifficultyEstimator(CognitiveDifficultyEstimator):
    """
    LLM-based estimator — useful for generating ground-truth labels or
    auditing the rule-based estimator on a sample.

    Expects an Anthropic client. Uses a cheap/fast model by default.
    """

    _PROMPT = """\
You are an expert in language learning difficulty assessment.

Given a question, its answer, and knowledge graph triples from the passage, \
estimate the COGNITIVE difficulty of the question.

Cognitive difficulty = how hard is the reasoning required, not vocabulary level:
  easy   (0.0–0.33) — direct factual recall; answer is stated explicitly
  medium (0.33–0.67) — requires connecting facts, multi-step reasoning, or pronoun resolution
  hard   (0.67–1.0) — causal/procedural reasoning, abstract inference, or no direct KG evidence

Question : {question}
Answer   : {answer}
KG triples (subject | relation | object):
{triples}

Reply with ONLY a JSON object — no prose, no markdown:
{{"score": <float 0-1>, "label": "<easy|medium|hard>", "reasoning": "<one sentence>"}}"""

    def __init__(self, client, model: str = "claude-haiku-4-5-20251001") -> None:
        self._client = client
        self._model = model

    def score(
        self,
        question: str,
        answer: str,
        kg_raw: list[list[str]],
        kg_coref: list[list[str]] | None = None,
    ) -> float:
        return self.estimate_full(question, answer, kg_raw, kg_coref)["score"]

    def estimate(
        self,
        question: str,
        answer: str,
        kg_raw: list[list[str]],
        kg_coref: list[list[str]] | None = None,
    ) -> dict:
        return self.estimate_full(question, answer, kg_raw, kg_coref)

    def estimate_full(
        self,
        question: str,
        answer: str,
        kg_raw: list[list[str]],
        kg_coref: list[list[str]] | None = None,
    ) -> dict:
        triples = kg_coref or kg_raw
        triples_str = "\n".join(f"  {t[0]} | {t[1]} | {t[2]}" for t in triples)
        prompt = self._PROMPT.format(
            question=question, answer=answer, triples=triples_str
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        result = json.loads(response.content[0].text)
        result["label"] = _label(result["score"])
        return result
