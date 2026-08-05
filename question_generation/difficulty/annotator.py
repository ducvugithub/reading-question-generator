from __future__ import annotations

import json
from langchain_core.messages import HumanMessage, SystemMessage


def _model_name(llm) -> str:
    """Extract model identifier from any LangChain chat model."""
    for attr in ("model_id", "model", "model_name"):
        val = getattr(llm, attr, None)
        if val:
            return str(val)
    return type(llm).__name__


def _extract_text(response) -> str:
    """Extract plain text from a LangChain AIMessage, handling content-block lists."""
    content = response.content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                content = block["text"]
                break
            if isinstance(block, str):
                content = block
                break
    return str(content)


def _parse_json(text: str) -> dict:
    """Parse JSON, stripping markdown code fences if the model added them."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text.strip())


_SYSTEM = """\
You are an expert in educational assessment and multilingual reading comprehension.
Given a passage, its knowledge-graph triples, and existing question-answer pairs, you must:

  1. Score every existing question-answer pair across four difficulty dimensions.
  2. Generate new question-answer pairs to reach the requested total.

━━━ DIMENSIONS (all scores 0.0–1.0, use the full range) ━━━

passage_readability   — How structurally complex is the passage?
                        Consider sentence length, syntactic nesting, and abstraction level.

passage_vocab_diff    — How difficult is the vocabulary in the passage?
                        Consider word frequency and domain-specificity.

question_cognitive_diff — How much reasoning is required to answer?
                        Consider whether the answer is explicit or inferred, the number of
                        reasoning steps, and whether coreference resolution is needed.

question_vocab_diff   — Does answering require vocabulary knowledge, or can the reader pattern-match?
                        When the question uses the same words as the passage, the reader can locate
                        the answer without understanding vocabulary (just text matching). When the
                        question uses paraphrases, synonyms, or terms NOT in the passage, the reader
                        must understand the question wording to map it to the answer.
                        Proper nouns (names, titles) are identifiers, not vocab.

━━━ GENERATION RULES ━━━
- Answers must be exact text spans from the passage (extractive, no paraphrasing).
- Generate questions that cover difficulty levels not yet represented in the existing set.
- Do not repeat or paraphrase existing questions.
- Vary question words (what / who / where / when / why / how).

Output a single JSON object. No prose, no markdown fences."""


_USER = """\
Passage:
{passage}

KG triples (subject | relation | object):
{triples}

Existing questions ({n_existing} total):
{existing_block}

{generate_instruction}

Reply with ONLY valid JSON:
{{
  "reasoning": "<2-3 sentences on passage difficulty and existing question variety>",
  "passage_readability": <float 0-1>,
  "passage_vocab_diff":  <float 0-1>,
  "scored": [
    {{"question": "<text>", "answer": "<text>", "question_cognitive_diff": <float>, "question_vocab_diff": <float>}}
  ],
  "generated": [
    {{"question": "<text>", "answer": "<exact passage span>", "question_cognitive_diff": <float>, "question_vocab_diff": <float>}}
  ]
}}"""


class DifficultyAnnotator:
    """
    Single LLM call per passage:
      - Scores all existing QA pairs across 4 difficulty dimensions
      - Generates new QA pairs to reach `target_count`, spread across difficulty levels
      - Passage-level scores computed once and shared across all QA pairs

    Results are stored under `llm_diff_judge` keyed by model name, so multiple
    annotators (Haiku, Sonnet, GPT) can each add their own entry without overwriting:

      record["llm_diff_judge"] = {
          "anthropic.claude-haiku-4-5-20251001-v1:0": { ... scores ... },
          "claude-sonnet-4-6":                         { ... scores ... },
      }

    Works with any LangChain BaseChatModel — Bedrock, Anthropic direct, OpenAI, Ollama.

    Example:
        from langchain_aws import ChatBedrockConverse
        llm = ChatBedrockConverse(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            region_name="us-east-1",
            max_tokens=700,
            temperature=0.3,
        )
        annotator = DifficultyAnnotator(llm, target_count=5)
        result = annotator.annotate(
            passage="...",
            qa_pairs=[("What city?", "Paris")],
            kg_raw=[["Paris", "capital_of", "France"]],
        )
        # result["scored"][0]["llm_diff_judge"]["anthropic.claude-haiku-4-5..."] = {...}
    """

    def __init__(self, llm, target_count: int = 5, max_questions_per_call: int = 8) -> None:
        self._llm = llm
        self._model_name = _model_name(llm)
        self._target_count = target_count
        self._max_q = max_questions_per_call

    @property
    def model_name(self) -> str:
        return self._model_name

    def annotate(
        self,
        passage: str,
        qa_pairs: list[tuple[str, str]],
        kg_raw: list[list[str]] | None = None,
        kg_coref: list[list[str]] | None = None,
    ) -> dict:
        """
        Score all existing QA pairs and generate new ones to reach target_count.

        Passages with more than max_questions_per_call questions are split into
        chunks — generation only happens in the first chunk, passage-level scores
        are taken from the first call and reused for subsequent chunks.

        Returns:
          {
            "passage_readability": float,
            "passage_vocab_diff":  float,
            "scored":    [{"question", "answer", "question_cognitive_diff", "question_vocab_diff"}, ...],
            "generated": [{"question", "answer", "question_cognitive_diff", "question_vocab_diff",
                           "target", "generated_by"}, ...],
          }
        """
        if len(qa_pairs) <= self._max_q:
            return self._call(passage, qa_pairs, kg_raw, kg_coref)

        # Split into chunks — first chunk may generate, rest are score-only
        chunks = [qa_pairs[i:i + self._max_q] for i in range(0, len(qa_pairs), self._max_q)]
        first  = self._call(passage, chunks[0], kg_raw, kg_coref)
        all_scored = list(first.get("scored", []))

        for chunk in chunks[1:]:
            # Score-only: tell the model not to generate
            rest = self._call(passage, chunk, kg_raw, kg_coref, score_only=True)
            all_scored.extend(rest.get("scored", []))

        first["scored"] = all_scored
        return first

    def _call(
        self,
        passage: str,
        qa_pairs: list[tuple[str, str]],
        kg_raw: list[list[str]] | None,
        kg_coref: list[list[str]] | None,
        score_only: bool = False,
    ) -> dict:
        n_existing  = len(qa_pairs)
        n_generate  = 0 if score_only else max(0, self._target_count - n_existing)
        triples     = kg_coref or kg_raw or []
        triples_str = "\n".join(f"  {t[0]} | {t[1]} | {t[2]}" for t in triples) or "  (none)"

        existing_block = "\n".join(
            f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(qa_pairs)
        ) or "(none)"

        if n_generate > 0:
            generate_instruction = (
                f"Generate {n_generate} new question-answer pair(s). "
                "Look at the question_cognitive_diff scores of the existing questions above "
                "and generate questions that cover difficulty levels not yet represented — "
                "aim to spread across the full 0.0–1.0 range of question_cognitive_diff."
            )
        else:
            generate_instruction = (
                "Score only — do not generate any new questions. Set \"generated\" to []."
            )

        user_msg = _USER.format(
            passage=passage,
            triples=triples_str,
            n_existing=n_existing,
            existing_block=existing_block,
            generate_instruction=generate_instruction,
        )

        response = self._llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        result = _parse_json(_extract_text(response))

        for item in result.get("generated", []):
            item["generated_by"] = self._model_name

        return result

    def merge_into_records(
        self,
        records: list[dict],
        passage_key:  str = "passage",
        question_key: str = "question",
        answer_key:   str = "answer",
        kg_raw_key:   str = "kg_raw",
        kg_coref_key: str = "kg_coref",
        on_error:     str = "skip",
    ) -> tuple[list[dict], list[dict]]:
        """
        Annotate records grouped by passage, merge scores into each record under
        `record["llm_diff_judge"][model_name]`, and return new generated records.

        Groups records by passage text so one LLM call covers all questions per passage.

        Returns:
          (updated_records, new_generated_records)
          - updated_records: same list with llm_diff_judge[model_name] added in-place
          - new_generated_records: new JSONL-ready dicts for generated QA pairs,
            with generated=True and generated_by=model_name set
        """
        from collections import defaultdict

        # Group by passage
        passage_groups: dict[str, list[int]] = defaultdict(list)
        for i, rec in enumerate(records):
            passage_groups[rec[passage_key]].append(i)

        generated_records: list[dict] = []

        for passage_text, indices in passage_groups.items():
            first = records[indices[0]]
            qa_pairs = [(records[i][question_key], records[i][answer_key]) for i in indices]

            try:
                result = self.annotate(
                    passage=passage_text,
                    qa_pairs=qa_pairs,
                    kg_raw=first.get(kg_raw_key),
                    kg_coref=first.get(kg_coref_key),
                )
            except Exception:
                if on_error == "skip":
                    continue
                raise

            passage_scores = {
                "passage_readability": result["passage_readability"],
                "passage_vocab_diff":  result["passage_vocab_diff"],
            }

            # Merge scored results back into existing records
            for idx, scored in zip(indices, result.get("scored", [])):
                rec = records[idx]
                rec.setdefault("llm_diff_judge", {})[self._model_name] = {
                    **passage_scores,
                    "question_cognitive_diff": scored["question_cognitive_diff"],
                    "question_vocab_diff":     scored["question_vocab_diff"],
                }

            # Build new records for generated QA pairs
            for gen in result.get("generated", []):
                new_rec = {
                    **{k: first[k] for k in (passage_key, kg_raw_key, kg_coref_key,
                                              "source", "lang", "cefr") if k in first},
                    question_key: gen["question"],
                    answer_key:   gen["answer"],
                    "generated":    True,
                    "generated_by": self._model_name,
                    "llm_diff_judge": {
                        self._model_name: {
                            **passage_scores,
                            "question_cognitive_diff": gen["question_cognitive_diff"],
                            "question_vocab_diff":     gen["question_vocab_diff"],
                        }
                    },
                }
                generated_records.append(new_rec)

        return records, generated_records
