"""
Linguistic feature extraction for question difficulty estimation.

Features are computed from (passage, question, answer) triples using only
wordfreq for vocabulary difficulty — no heavy NLP models required.
"""
from __future__ import annotations

import re
from functools import lru_cache

from wordfreq import zipf_frequency

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "that", "this", "it", "its",
    "and", "or", "but", "not", "no", "what", "which", "who", "whom",
    "when", "where", "why", "how",
}

_WH_WORDS = ["what", "who", "where", "when", "why", "how", "which", "whose", "whom"]

_SENT_RE = re.compile(r'[.!?]+')


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _content_words(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


@lru_cache(maxsize=4096)
def _zipf(word: str) -> float:
    return zipf_frequency(word, "en")


def _avg_zipf(tokens: list[str]) -> float:
    scores = [_zipf(t) for t in tokens if t.isalpha()]
    return sum(scores) / len(scores) if scores else 0.0


def _frac_rare(tokens: list[str], threshold: float = 3.0) -> float:
    alpha = [t for t in tokens if t.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for t in alpha if _zipf(t) < threshold) / len(alpha)


def _wh_type(question: str) -> int:
    first = question.lower().strip().split()[0] if question.strip() else ""
    for i, w in enumerate(_WH_WORDS):
        if first == w:
            return i
    return len(_WH_WORDS)  # "other"


def extract(passage: str, question: str, answer: str) -> dict[str, float]:
    q_tokens = _tokenize(question)
    p_tokens = _tokenize(passage)
    a_tokens = _tokenize(answer)
    q_content = _content_words(q_tokens)
    p_set = set(p_tokens)

    # question features
    q_n_tokens    = len(q_tokens)
    q_avg_wlen    = sum(len(t) for t in q_tokens) / len(q_tokens) if q_tokens else 0.0
    q_wh_type     = _wh_type(question)
    q_avg_zipf    = _avg_zipf(q_tokens)
    q_frac_rare   = _frac_rare(q_tokens)

    # passage features
    sents         = [s for s in _SENT_RE.split(passage) if s.strip()]
    p_n_tokens    = len(p_tokens)
    p_n_sents     = max(len(sents), 1)
    p_avg_slen    = p_n_tokens / p_n_sents
    p_ttr         = len(set(p_tokens)) / p_n_tokens if p_n_tokens else 0.0
    p_avg_zipf    = _avg_zipf(p_tokens)

    # answer features
    a_n_tokens    = len(a_tokens)
    a_avg_zipf    = _avg_zipf(a_tokens)

    # interaction features
    overlap       = (
        sum(1 for t in q_content if t in p_set) / len(q_content)
        if q_content else 0.0
    )
    a_in_passage  = float(answer.lower() in passage.lower())

    return {
        "q_n_tokens":  q_n_tokens,
        "q_avg_wlen":  q_avg_wlen,
        "q_wh_type":   q_wh_type,
        "q_avg_zipf":  q_avg_zipf,
        "q_frac_rare": q_frac_rare,
        "p_n_tokens":  p_n_tokens,
        "p_n_sents":   p_n_sents,
        "p_avg_slen":  p_avg_slen,
        "p_ttr":       p_ttr,
        "p_avg_zipf":  p_avg_zipf,
        "a_n_tokens":  a_n_tokens,
        "a_avg_zipf":  a_avg_zipf,
        "q_p_overlap": overlap,
        "a_in_passage": a_in_passage,
    }


FEATURE_NAMES = list(extract("x.", "What?", "x").keys())
