# utils/hpo_sentence_extractor.py
"""
Extract sentences relevant to a given phenotype using SciSpaCy.
No linker or KB. Simply match by term / synonyms with biomedical tokenization.
"""
from __future__ import annotations
import spacy
import re
from typing import List

class HPOSentenceExtractor:
    def __init__(self, model: str = "en_core_sci_sm"):
        self.nlp = spacy.load(model, disable=["ner"])  # faster
        self.nlp.add_pipe("sentencizer")

    @staticmethod
    def _compile_patterns(terms: List[str]):
        """Compile regex to detect any synonym in text (case insensitive)."""
        escaped = [re.escape(t.strip()) for t in terms if t.strip()]
        if not escaped:
            return None
        return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)

    def extract_relevant_sentences(self, text: str, terms: List[str]) -> List[str]:
        """
        Split text into sentences and return only those mentioning any of the given terms.
        """
        if not text or not terms:
            return []

        pattern = self._compile_patterns(terms)
        if pattern is None:
            return []

        doc = self.nlp(text)
        relevant = []
        for sent in doc.sents:
            s = sent.text.strip()
            if pattern.search(s):
                relevant.append(s)
        return relevant