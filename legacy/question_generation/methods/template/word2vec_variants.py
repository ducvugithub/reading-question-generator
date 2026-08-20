"""
Word2Vec-based question variant generation using gensim.

Uses gensim Word2Vec models (Google News for EN, FastText-wiki for FI) combined with
word frequency percentile-based CEFR classification to create difficulty variants by
replacing main verbs with semantically similar synonyms at different CEFR levels.
"""

from __future__ import annotations

import pandas as pd
from typing import Optional
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    from gensim.downloader import load
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("⚠ Warning: gensim not installed. Install with: pip install gensim")

# CEFR percentile bins (dynamic, calculated from word frequency data)
# Distribution: A1=30%, A2=25%, B1=20%, B2=15%, C1=7%, C2=3%
CEFR_PERCENTILES = {
    "A1": (0.70, 1.00),    # Top 30% (most frequent)
    "A2": (0.45, 0.70),    # 30-55%
    "B1": (0.25, 0.45),    # 55-75%
    "B2": (0.10, 0.25),    # 75-90%
    "C1": (0.03, 0.10),    # 90-97%
    "C2": (0.00, 0.03),    # 97-100% (least frequent)
}

# Cache for percentile thresholds (will be populated lazily)
_freq_percentiles_en = None
_freq_percentiles_fi = None


def get_frequency_percentiles(lang: str = "en"):
    """Get frequency percentile thresholds for a language."""
    global _freq_percentiles_en, _freq_percentiles_fi

    if lang == "en":
        if _freq_percentiles_en is not None:
            return _freq_percentiles_en
        # Empirical percentile thresholds for English (from wordfreq scale)
        # A1=30%, A2=25%, B1=20%, B2=15%, C1=7%, C2=3%
        _freq_percentiles_en = {
            "A1": (0.001, float('inf')),      # Most frequent 30%
            "A2": (0.0003, 0.001),            # Next 25%
            "B1": (0.0001, 0.0003),           # Next 20%
            "B2": (0.00003, 0.0001),          # Next 15%
            "C1": (0.000008, 0.00003),        # Next 7%
            "C2": (0, 0.000008),              # Least frequent 3%
        }
        return _freq_percentiles_en
    elif lang == "fi":
        if _freq_percentiles_fi is not None:
            return _freq_percentiles_fi
        # Empirical percentile thresholds for Finnish
        _freq_percentiles_fi = {
            "A1": (0.1, float('inf')),        # Most frequent 30%
            "A2": (0.03, 0.1),                # Next 25%
            "B1": (0.01, 0.03),               # Next 20%
            "B2": (0.003, 0.01),              # Next 15%
            "C1": (0.0008, 0.003),            # Next 7%
            "C2": (0, 0.0008),                # Least frequent 3%
        }
        return _freq_percentiles_fi

    return None


class Word2VecVariantGenerator:
    """Generate question variants by replacing main verbs using gensim Word2Vec."""

    def __init__(self, lang: str = "en"):
        """
        Initialize the variant generator.

        Args:
            lang: Language code ("en" or "fi")
        """
        self.lang = lang
        self.model = None
        self.freq_data = None
        self._load_resources()

    def _load_resources(self):
        """Load gensim Word2Vec model and frequency data."""
        if not GENSIM_AVAILABLE:
            print("⚠ gensim not available. Skipping Word2Vec model loading.")
            self.model = None
        else:
            self._load_word2vec_model()

        # Load frequency data
        self._load_frequency_data()

    def _load_word2vec_model(self):
        """Load pre-trained Word2Vec model from gensim."""
        try:
            if self.lang == "en":
                print("Loading Google Word2Vec (English)...")
                self.model = load("word2vec-google-news-300")
                print("✓ Loaded word2vec-google-news-300")
            elif self.lang == "fi":
                print("Loading FastText-wiki (Finnish)...")
                self.model = load("fasttext-wiki-300")
                print("✓ Loaded fasttext-wiki-300")
            else:
                raise ValueError(f"Unsupported language: {self.lang}")
        except Exception as e:
            print(f"⚠ Failed to load Word2Vec model: {e}")
            print("  Variant generation will be limited")
            self.model = None

    def _load_frequency_data(self):
        """Load word frequency data from wordfreq or local files."""
        try:
            if self.lang == "en":
                from wordfreq import word_frequency
                self.freq_lookup = lambda w: word_frequency(w, 'en')
                print("✓ Using wordfreq library for English frequencies")
            elif self.lang == "fi":
                # Load Finnish vocab file
                import os
                vocab_file = os.path.join(
                    os.path.dirname(__file__),
                    "..",  # Go up one level to repo root
                    "data/fi/vocab/fi_merged_clean_reposition.csv"
                )
                if os.path.exists(vocab_file):
                    df = pd.read_csv(vocab_file)
                    self.freq_data = dict(zip(df['Word'].str.lower(), df['freq']))
                    self.freq_lookup = lambda w: self.freq_data.get(w.lower(), 0.0)
                    print(f"✓ Loaded Finnish vocab: {len(self.freq_data)} words")
                else:
                    print(f"⚠ Finnish vocab file not found: {vocab_file}")
                    self.freq_lookup = lambda w: 0.0
        except ImportError:
            print("⚠ wordfreq not installed. Install with: pip install wordfreq")
            self.freq_lookup = lambda w: 0.0

    def get_cefr_for_word(self, word: str) -> Optional[str]:
        """
        Get CEFR level for a word based on frequency percentile.

        Args:
            word: Word to check

        Returns:
            CEFR level (A1-C2) or None
        """
        freq = self.freq_lookup(word)
        if freq == 0:
            return None

        percentiles = get_frequency_percentiles(self.lang)
        if not percentiles:
            return None

        # Map frequency to percentile-based CEFR level
        for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
            min_freq, max_freq = percentiles.get(level, (0, 0))
            if min_freq <= freq <= max_freq:
                return level

        # Default to C2 if below all thresholds
        return "C2"

    def find_verb_synonym(
        self,
        verb: str,
        target_cefr: str,
        num_candidates: int = 20,
        similarity_threshold: float = 0.5
    ) -> Optional[str]:
        """
        Find a synonym for a verb at target CEFR level using Word2Vec.

        Args:
            verb: Original verb
            target_cefr: Target CEFR level (A1-C2)
            num_candidates: Number of Word2Vec neighbors to check
            similarity_threshold: Minimum similarity score

        Returns:
            Best matching synonym or None if not found
        """
        if self.model is None:
            return None

        try:
            # Get Word2Vec neighbors (gensim API)
            verb_lower = verb.lower()
            if verb_lower not in self.model.key_to_index:
                return None

            neighbors = self.model.most_similar(verb_lower, topn=num_candidates)
        except Exception:
            return None

        # Find neighbor matching target CEFR
        candidates = []
        cefr_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
        target_idx = cefr_order.index(target_cefr) if target_cefr in cefr_order else 0

        for neighbor_word, similarity in neighbors:
            if similarity < similarity_threshold:
                continue

            neighbor_cefr = self.get_cefr_for_word(neighbor_word)
            if neighbor_cefr is None:
                continue

            # Accept exact match or neighbors close in difficulty
            neighbor_idx = cefr_order.index(neighbor_cefr)
            # Prefer exact match, but accept ±2 levels if similarity is high enough
            difficulty_match = abs(neighbor_idx - target_idx)
            if difficulty_match == 0 or (difficulty_match <= 2 and similarity > 0.55):
                candidates.append((similarity, neighbor_word))

        if candidates:
            # Return best match (highest similarity)
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return None

    def create_variant(
        self,
        question: str,
        verb: str,
        target_cefr: str
    ) -> Optional[str]:
        """
        Create a question variant by replacing main verb.

        Args:
            question: Original question text
            verb: Main verb to replace
            target_cefr: Target CEFR level for replacement

        Returns:
            Question with replaced verb, or original if replacement fails
        """
        if target_cefr == self.get_cefr_for_word(verb):
            # Already at target level
            return question

        synonym = self.find_verb_synonym(verb, target_cefr)
        if synonym and synonym != verb:
            # Replace verb (case-insensitive, preserve original case)
            import re
            pattern = r'\b' + re.escape(verb) + r'\b'
            variant = re.sub(pattern, synonym, question, flags=re.IGNORECASE)
            return variant

        # Fallback: return original
        return question


# Singleton instance
_generator_en = None
_generator_fi = None


class QuestionVariantGenerator:
    """Generate question variants with different vocabulary difficulty."""

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.generator = Word2VecVariantGenerator(lang=lang)

    def get_verb_variant(self, verb: str, target_cefr: str) -> str:
        """Get a synonym for a verb at target CEFR level using Word2Vec."""
        # Try Word2Vec
        synonym = self.generator.find_verb_synonym(verb, target_cefr)
        if synonym:
            return synonym

        # No synonym found: return original verb
        return verb

    def create_variants(
        self,
        question: str,
        main_verb: str,
        cefr_levels: list = None
    ) -> dict:
        """Create question variants at different CEFR levels."""
        if cefr_levels is None:
            cefr_levels = ["A1", "A2", "B1", "B2", "C1"]

        variants = {}
        import re

        for cefr in cefr_levels:
            synonym = self.get_verb_variant(main_verb, cefr)

            if synonym and synonym != main_verb:
                pattern = r'\b' + re.escape(main_verb) + r'\b'
                variant = re.sub(pattern, synonym, question, flags=re.IGNORECASE)
            else:
                variant = question

            variants[cefr] = variant

        return variants


def get_generator(lang: str = "en") -> Word2VecVariantGenerator:
    """Get or create a Word2VecVariantGenerator instance."""
    global _generator_en, _generator_fi

    if lang == "en":
        if _generator_en is None:
            _generator_en = Word2VecVariantGenerator(lang="en")
        return _generator_en
    elif lang == "fi":
        if _generator_fi is None:
            _generator_fi = Word2VecVariantGenerator(lang="fi")
        return _generator_fi
    else:
        raise ValueError(f"Unsupported language: {lang}")

# Add singleton variables for variant generator
_variant_gen_en = None
_variant_gen_fi = None


def get_variant_generator(lang: str = "en") -> QuestionVariantGenerator:
    """Get or create a QuestionVariantGenerator instance."""
    global _variant_gen_en, _variant_gen_fi

    if lang == "en":
        if _variant_gen_en is None:
            _variant_gen_en = QuestionVariantGenerator(lang="en")
        return _variant_gen_en
    elif lang == "fi":
        if _variant_gen_fi is None:
            _variant_gen_fi = QuestionVariantGenerator(lang="fi")
        return _variant_gen_fi
    else:
        raise ValueError(f"Unsupported language: {lang}")
