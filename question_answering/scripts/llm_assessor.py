#!/usr/bin/env python3
"""
LLM-based quality assessment for generated questions.

Loads multiple 7B open-source LLMs (Qwen, Llama 2, Mistral) and assesses
generated questions on quality, answerability, and coherence.
"""
from __future__ import annotations

import abc
from typing import Optional


_ASSESSMENT_PROMPT = """\
Assess this reading comprehension question on the following criteria:

PASSAGE:
{passage}

QUESTION:
{question}

Rate the question from 1-5 on these dimensions:
1. Grammaticality: Is the question well-formed grammatically?
2. Answerability: Can this question be answered from the passage?
3. Clarity: Is the question clear and unambiguous?
4. Relevance: Does the question test comprehension of the passage?

Respond in JSON format:
{{
  "grammaticality": <1-5>,
  "answerability": <1-5>,
  "clarity": <1-5>,
  "relevance": <1-5>,
  "overall_quality": <1-5>,
  "reasoning": "<one sentence explanation>"
}}

Be strict but fair. Questions requiring inference (not just surface matching) are good.
"""


class LLMAssessor(abc.ABC):
    """Abstract base for LLM-based question assessment."""

    @abc.abstractmethod
    def assess(self, passage: str, question: str) -> dict:
        """
        Assess a question.

        Returns dict with:
        - quality scores (1-5 scale)
        - reasoning
        - overall_quality (average)
        - error (if failed)
        """
        pass

    @abc.abstractmethod
    def get_model_name(self) -> str:
        """Return human-readable model name."""
        pass


class TransformersLLMAssessor(LLMAssessor):
    """Assessor using Hugging Face transformers pipeline."""

    def __init__(self, model_name: str, device: int = 0):
        """
        Load a model via transformers pipeline.

        Args:
            model_name: HF model ID (e.g., 'Qwen/Qwen2-7B-Instruct')
            device: GPU device ID (-1 for CPU)
        """
        try:
            from transformers import pipeline
            import json as json_module
        except ImportError:
            raise ImportError("transformers required: pip install transformers torch")

        self.model_name = model_name
        self.device = device
        self.json_module = json_module

        print(f"Loading {model_name}...", flush=True)
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": "auto"},
        )

    def assess(self, passage: str, question: str) -> dict:
        """Assess question quality."""
        prompt = _ASSESSMENT_PROMPT.format(passage=passage, question=question)

        try:
            # Generate response
            outputs = self.pipe(
                prompt,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
            )
            response_text = outputs[0]["generated_text"][len(prompt):].strip()

            # Extract JSON from response
            # Try to find JSON block
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = self.json_module.loads(json_str)

                # Compute overall quality as average
                scores = [
                    result.get("grammaticality", 3),
                    result.get("answerability", 3),
                    result.get("clarity", 3),
                    result.get("relevance", 3),
                ]
                result["overall_quality"] = sum(scores) / len(scores)
                return result
            else:
                return {
                    "error": f"Could not extract JSON from response: {response_text[:100]}",
                    "overall_quality": None,
                }

        except Exception as e:
            return {
                "error": str(e),
                "overall_quality": None,
            }

    def get_model_name(self) -> str:
        """Return model name."""
        return self.model_name


class LLMAssessorPool:
    """Manage multiple LLM assessors and run assessments in parallel."""

    def __init__(self, model_names: list[str], device: int = 0):
        """
        Load multiple models.

        Args:
            model_names: List of HF model IDs
            device: GPU device ID
        """
        self.assessors = {}
        for model_name in model_names:
            try:
                self.assessors[model_name] = TransformersLLMAssessor(model_name, device)
            except Exception as e:
                print(f"Failed to load {model_name}: {e}", flush=True)

        if not self.assessors:
            raise RuntimeError("No LLM assessors loaded")

        print(f"Loaded {len(self.assessors)} LLM assessors")

    def assess_all(self, passage: str, question: str) -> dict:
        """
        Run assessment with all loaded models.

        Returns dict mapping model_name → assessment results.
        """
        results = {}
        for model_name, assessor in self.assessors.items():
            results[model_name] = assessor.assess(passage, question)
        return results

    def get_consensus(self, assessments: dict) -> dict:
        """
        Compute consensus metrics across all models.

        Args:
            assessments: dict from assess_all()

        Returns dict with:
            - avg_overall_quality
            - quality_agreement (std dev)
        """
        valid_scores = [
            a.get("overall_quality")
            for a in assessments.values()
            if a.get("overall_quality") is not None
        ]

        if not valid_scores:
            return {"avg_overall_quality": None, "quality_std": None}

        avg = sum(valid_scores) / len(valid_scores)
        variance = sum((x - avg) ** 2 for x in valid_scores) / len(valid_scores)
        std = variance ** 0.5

        return {
            "avg_overall_quality": avg,
            "quality_std": std,
            "num_models_succeeded": len(valid_scores),
        }
