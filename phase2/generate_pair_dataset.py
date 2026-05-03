#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate sentence pairs with an HPO-based gold similarity score.

This script loads the sentence corpus created in Phase 1 (Parquet dataset
partitioned by ``hpo_id``) and samples sentence pairs grouped in three
semantic bands: positive (high similarity), intermediate, and negative
pairs. The similarity is computed with PyHPO (default: Lin, normalized to
[0, 1]) or with the custom RelativeBestPair implementation for ontology-
driven experiments.

Example:
    python phase2/generate_pair_dataset.py \
        --output data/phase2/hpo_sentence_pairs.parquet \
        --positive-target 8000 \
        --intermediate-target 12000 \
        --negative-target 8000
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Tuple

import pandas as pd
import pyarrow.dataset as ds
import pyhpo
from tqdm import tqdm

import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import hpo_similarity


@dataclass(frozen=True)
class PairRecord:
    """Lightweight container for a sampled pair."""

    sentence1: str
    sentence2: str
    hpo_id1: str
    hpo_id2: str
    gold_similarity: float


def format_input_prompt(phenotype_name: str, fragment: str) -> str:
    """Standardize the model input prompt with the phenotype name upfront."""
    clean_name = " ".join((phenotype_name or "").split())
    clean_fragment = " ".join((fragment or "").split())
    return f"[CLS] {clean_name} [SEP] {clean_fragment} [SEP]"


class HPOSentencePairGenerator:
    """Sample sentence pairs from the corpus and score them with PyHPO."""

    _SYNONYM_ATTRS = ("synonyms", "alt_names", "alternative_names")
    _NEGATION_PATTERNS: Tuple[Pattern[str], ...] = (
        re.compile(r"\bno\b", re.IGNORECASE),
        re.compile(r"\bnot\b", re.IGNORECASE),
        re.compile(r"\bwithout\b", re.IGNORECASE),
        re.compile(r"\babsence of\b", re.IGNORECASE),
        re.compile(r"\babsent\b", re.IGNORECASE),
        re.compile(r"\bfree of\b", re.IGNORECASE),
        re.compile(r"\black of\b", re.IGNORECASE),
        re.compile(r"\blacking\b", re.IGNORECASE),
        re.compile(r"\bdenies\b", re.IGNORECASE),
        re.compile(r"\bdenied\b", re.IGNORECASE),
        re.compile(r"\bnegative for\b", re.IGNORECASE),
        re.compile(r"\b(?:rule|ruled|ruling)\s+out\b", re.IGNORECASE),
        re.compile(r"\b(?:exclude|excluded|excluding)\b", re.IGNORECASE),
        re.compile(r"\bno evidence of\b", re.IGNORECASE),
        re.compile(r"\bnot present\b", re.IGNORECASE),
    )

    def __init__(
        self,
        dataset_path: str,
        method: str = "lin",
        positive_threshold: float = 0.7,
        negative_threshold: float = 0.3,
        max_sentences_per_hpo: int = 50,
        min_sentences_per_hpo: int = 1,
        batch_size: int = 5000,
        seed: int = 13,
        show_progress: bool = True,
        rbp_alpha: float = 0.01,
    ) -> None:
        if positive_threshold <= negative_threshold:
            raise ValueError("Positive threshold must be greater than negative threshold.")

        self.dataset_path = dataset_path
        self.method = method
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.max_sentences_per_hpo = max_sentences_per_hpo
        self.min_sentences_per_hpo = min_sentences_per_hpo
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.rbp_alpha = rbp_alpha

        self._rng = random.Random(seed)
        self._ontology = pyhpo.Ontology()
        self._term_cache: Dict[str, Optional[pyhpo.term.HPOTerm]] = {}
        self._rbp_enabled = self.method.lower() in {"relativebestpair", "rbp"}
        self._mention_cache: Dict[str, List[str]] = {}
        self._mention_pattern_cache: Dict[str, List[Pattern[str]]] = {}

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _get_term_name(self, hpo_id: str) -> Optional[str]:
        term = self._get_term(hpo_id)
        if term is None:
            return None
        name = getattr(term, "name", None)
        if not name:
            return None
        return str(name).strip()

    def _get_term_mentions(self, hpo_id: str) -> List[str]:
        if hpo_id in self._mention_cache:
            return self._mention_cache[hpo_id]

        mentions: List[str] = []
        term = self._get_term(hpo_id)
        if term is not None:
            candidates: List[str] = []
            if term.name:
                candidates.append(str(term.name))
            for attr in self._SYNONYM_ATTRS:
                syns = getattr(term, attr, None)
                if not syns:
                    continue
                try:
                    for raw in syns:
                        text = (raw or "").strip()
                        if text:
                            candidates.append(text)
                except Exception:
                    continue
            seen: set[str] = set()
            for cand in sorted(candidates, key=len, reverse=True):
                lower = cand.lower()
                if lower in seen:
                    continue
                seen.add(lower)
                mentions.append(cand)

        self._mention_cache[hpo_id] = mentions
        return mentions

    def _get_mention_patterns(self, hpo_id: str) -> List[Pattern[str]]:
        if hpo_id in self._mention_pattern_cache:
            return self._mention_pattern_cache[hpo_id]

        patterns: List[Pattern[str]] = []
        for mention in self._get_term_mentions(hpo_id):
            escaped = re.escape(mention)
            patterns.append(re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE))

        self._mention_pattern_cache[hpo_id] = patterns
        return patterns

    def _find_mention_span(self, sentence: str, hpo_id: str) -> Optional[Tuple[int, int]]:
        for pattern in self._get_mention_patterns(hpo_id):
            match = pattern.search(sentence)
            if match:
                return match.span()

        name = self._get_term_name(hpo_id)
        if name:
            idx = sentence.lower().find(name.lower())
            if idx != -1:
                return idx, idx + len(name)
        return None

    def _truncate_around_phenotype(self, sentence: str, hpo_id: str) -> str:
        words = list(re.finditer(r"\S+", sentence))
        if len(words) <= 60:
            return sentence.strip()

        span = self._find_mention_span(sentence, hpo_id)
        if span is None:
            return " ".join(token.group(0) for token in words[:60]).strip()

        mention_idx = next((idx for idx, token in enumerate(words) if token.start() <= span[0] < token.end()), None)
        if mention_idx is None:
            return " ".join(token.group(0) for token in words[:60]).strip()

        start_idx = max(0, mention_idx - 15)
        end_idx = min(len(words), mention_idx + 16)
        snippet_start = words[start_idx].start()
        snippet_end = words[end_idx - 1].end()
        return sentence[snippet_start:snippet_end].strip()

    def _looks_like_enumeration(self, sentence: str, hpo_id: str) -> bool:
        if sentence.count(",") < 2:
            return False

        mention_matches = sum(len(p.findall(sentence)) for p in self._get_mention_patterns(hpo_id))
        if mention_matches >= 3:
            return True

        fragments = [frag.strip() for frag in re.split(r"[;,]", sentence) if frag.strip()]
        short_chunks = 0
        for frag in fragments:
            cleaned = re.sub(r"^(and|or|with|without|as well as|plus)\s+", "", frag, flags=re.IGNORECASE)
            words = cleaned.split()
            if 1 <= len(words) <= 5:
                short_chunks += 1
            if short_chunks >= 3:
                return True
        return False

    def _contains_negation(self, sentence: str, hpo_id: str) -> bool:
        span = self._find_mention_span(sentence, hpo_id)
        if span is None:
            return False

        start, end = span
        window_start = max(0, start - 80)
        window_end = min(len(sentence), end + 40)
        window = sentence[window_start:window_end]
        return any(pattern.search(window) for pattern in self._NEGATION_PATTERNS)

    def _preprocess_sentence(self, hpo_id: str, raw_sentence: str, stats: Optional[Dict[str, int]] = None) -> Optional[str]:
        counter = stats if stats is not None else {}

        def _bump(key: str) -> None:
            counter[key] = counter.get(key, 0) + 1

        sentence = self._normalize_whitespace(str(raw_sentence))
        if not sentence:
            _bump("filtered_empty")
            return None
        if self._looks_like_enumeration(sentence, hpo_id):
            _bump("filtered_enumeration")
            return None
        if self._contains_negation(sentence, hpo_id):
            _bump("filtered_negation")
            return None

        words = sentence.split()
        truncated = len(words) > 60
        snippet = self._truncate_around_phenotype(sentence, hpo_id)
        if truncated:
            _bump("truncated")

        name = self._get_term_name(hpo_id)
        if not name:
            _bump("filtered_no_name")
            return None
        return format_input_prompt(name, snippet)

    def load_sentence_inventory(self) -> Dict[str, List[str]]:
        """
        Load up to ``max_sentences_per_hpo`` sentences per HPO term applying
        basic cleaning (negation/enumeration filter, truncation, prompt format).

        Returns:
            Mapping hpo_id -> list of example sentences.
        """
        dataset = ds.dataset(
            self.dataset_path,
            format="parquet",
            partitioning="hive",
        )
        batches = dataset.to_batches(
            columns=["hpo_id", "sentence"],
            batch_size=self.batch_size,
        )

        iterator: Iterable = batches
        total_rows = None
        if self.show_progress:
            total_rows = dataset.count_rows()
            total_batches = math.ceil(total_rows / self.batch_size)
            iterator = tqdm(
                batches,
                total=total_batches,
                desc="Loading sentences",
                unit="batch",
            )

        sentences: Dict[str, List[str]] = defaultdict(list)
        filter_stats: Dict[str, int] = defaultdict(int)
        for batch in iterator:
            data = batch.to_pydict()
            hpo_ids = data["hpo_id"]
            texts = data["sentence"]
            for hpo_id, sentence in zip(hpo_ids, texts):
                filter_stats["total_rows"] += 1
                if not hpo_id or not sentence:
                    filter_stats["filtered_missing_fields"] += 1
                    continue
                if self._get_term(hpo_id) is None:
                    filter_stats["filtered_unknown_hpo"] += 1
                    continue
                processed = self._preprocess_sentence(hpo_id, sentence, filter_stats)
                if not processed:
                    continue
                bucket = sentences[hpo_id]
                if len(bucket) >= self.max_sentences_per_hpo:
                    filter_stats["filtered_cap_reached"] += 1
                    continue
                bucket.append(processed)
                filter_stats["kept"] += 1

        # Filter out HPO terms with too few sentences or missing in ontology
        filtered: Dict[str, List[str]] = {}
        dropped_min = 0
        for hpo_id, examples in sentences.items():
            if len(examples) < self.min_sentences_per_hpo:
                dropped_min += 1
                continue
            term = self._get_term(hpo_id)
            if term is None:
                logging.warning("Skipping %s: not found in PyHPO ontology.", hpo_id)
                continue
            filtered[hpo_id] = examples

        logging.info(
            "Sentence filtering stats: total=%d kept=%d missing_fields=%d unknown_hpo=%d "
            "enumeration=%d negation=%d empty=%d no_name=%d truncated=%d capped=%d dropped_min_per_hpo=%d",
            filter_stats.get("total_rows", 0),
            filter_stats.get("kept", 0),
            filter_stats.get("filtered_missing_fields", 0),
            filter_stats.get("filtered_unknown_hpo", 0),
            filter_stats.get("filtered_enumeration", 0),
            filter_stats.get("filtered_negation", 0),
            filter_stats.get("filtered_empty", 0),
            filter_stats.get("filtered_no_name", 0),
            filter_stats.get("truncated", 0),
            filter_stats.get("filtered_cap_reached", 0),
            dropped_min,
        )

        logging.info(
            "Loaded sentences for %d HPO terms (out of %d in dataset).",
            len(filtered),
            len(sentences),
        )
        if not filtered:
            raise RuntimeError("No HPO term qualified for pair sampling.")
        return filtered

    def sample_pairs(
        self,
        inventory: Dict[str, List[str]],
        positive_target: int,
        intermediate_target: int,
        negative_target: int,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Sample sentence pairs and compute their similarity.

        Returns:
            df: DataFrame with columns (sentence1, sentence2, hpo_id1, hpo_id2, gold_similarity)
            counts: dict with the achieved counts per bucket.
        """
        targets = {
            "positive": max(0, positive_target),
            "intermediate": max(0, intermediate_target),
            "negative": max(0, negative_target),
        }
        requested_total = sum(targets.values())
        if requested_total == 0:
            raise ValueError("At least one target count must be greater than zero.")

        available_ids = list(inventory.keys())
        if len(available_ids) < 2:
            raise RuntimeError("Need at least two HPO terms with sentences to build pairs.")

        counts = {"positive": 0, "intermediate": 0, "negative": 0}
        records: List[PairRecord] = []
        seen_pairs: set[Tuple[str, str, str, str]] = set()

        max_attempts = requested_total * 400
        attempts = 0

        with tqdm(
            total=requested_total,
            desc="Sampling pairs",
            disable=not self.show_progress,
        ) as progress:
            while attempts < max_attempts and sum(counts.values()) < requested_total:
                attempts += 1
                hpo_id1, hpo_id2 = self._rng.sample(available_ids, 2)
                category, score = self._categorize_pair(hpo_id1, hpo_id2)
                if category is None or counts[category] >= targets[category]:
                    continue

                sentence1 = self._rng.choice(inventory[hpo_id1])
                sentence2 = self._rng.choice(inventory[hpo_id2])
                key = (hpo_id1, hpo_id2, sentence1, sentence2)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)

                records.append(
                    PairRecord(
                        sentence1=sentence1,
                        sentence2=sentence2,
                        hpo_id1=hpo_id1,
                        hpo_id2=hpo_id2,
                        gold_similarity=score,
                    )
                )
                counts[category] += 1
                progress.update(1)

        for label, target in targets.items():
            if counts[label] < target:
                logging.warning(
                    "Bucket %s: requested %d, achieved %d.",
                    label,
                    target,
                    counts[label],
                )

        if not records:
            raise RuntimeError("Failed to sample any sentence pairs; relax thresholds or targets.")

        df = pd.DataFrame([r.__dict__ for r in records])
        return df, counts

    def _categorize_pair(self, hpo_id1: str, hpo_id2: str) -> Tuple[Optional[str], float]:
        """Compute similarity and assign it to one of the semantic buckets."""
        term1 = self._get_term(hpo_id1)
        term2 = self._get_term(hpo_id2)
        if term1 is None or term2 is None:
            return None, 0.0
        try:
            if self._rbp_enabled:
                score = hpo_similarity.relative_best_pair(term1, term2, alpha=self.rbp_alpha)
            else:
                score = float(term1.similarity_score(term2, method=self.method))
        except Exception as exc:
            logging.debug(
                "Could not compute similarity for %s vs %s: %s",
                hpo_id1,
                hpo_id2,
                exc,
            )
            return None, 0.0
        if math.isnan(score) or math.isinf(score):
            return None, 0.0

        if score >= self.positive_threshold:
            category = "positive"
        elif score <= self.negative_threshold:
            category = "negative"
        else:
            category = "intermediate"
        return category, max(0.0, min(1.0, score))

    def _compute_similarity(self, hpo_id1: str, hpo_id2: str) -> Optional[float]:
        if hpo_id1 == hpo_id2:
            return 1.0
        term1 = self._get_term(hpo_id1)
        term2 = self._get_term(hpo_id2)
        if term1 is None or term2 is None:
            return None
        try:
            if self._rbp_enabled:
                score = hpo_similarity.relative_best_pair(term1, term2, alpha=self.rbp_alpha)
            else:
                score = float(term1.similarity_score(term2, method=self.method))
        except Exception:
            return None
        if score is None or math.isnan(score) or math.isinf(score):
            return None
        return max(0.0, min(1.0, score))

    def _select_hard_candidate(
        self,
        hpo_id: str,
        pool_ids: Sequence[str],
        hard_min_sim: float,
        hard_max_sim: float,
        max_attempts: int,
    ) -> Optional[Tuple[str, float]]:
        target_ids = [cand for cand in pool_ids if cand != hpo_id]
        if not target_ids:
            return None

        best_candidate: Optional[Tuple[str, float]] = None
        best_distance: Optional[float] = None
        desired_mid = (hard_min_sim + hard_max_sim) / 2.0

        for _ in range(max_attempts):
            cand = self._rng.choice(target_ids)
            score = self._compute_similarity(hpo_id, cand)
            if score is None:
                continue
            if hard_min_sim <= score <= hard_max_sim:
                return cand, score
            distance = abs(score - desired_mid)
            if best_distance is None or distance < best_distance:
                best_candidate = (cand, score)
                best_distance = distance
        if best_candidate is not None:
            return best_candidate
        # Fallback: random candidate with unknown score
        cand = self._rng.choice(target_ids)
        score = self._compute_similarity(hpo_id, cand) or 0.0
        return cand, score

    def _select_easy_candidate(
        self,
        hpo_id: str,
        pool_ids: Sequence[str],
        easy_max_sim: float,
        max_attempts: int,
    ) -> Optional[Tuple[str, float]]:
        target_ids = [cand for cand in pool_ids if cand != hpo_id]
        if not target_ids:
            return None

        best_candidate: Optional[Tuple[str, float]] = None
        best_score: Optional[float] = None
        for _ in range(max_attempts):
            cand = self._rng.choice(target_ids)
            score = self._compute_similarity(hpo_id, cand)
            if score is None:
                continue
            if score <= easy_max_sim:
                return cand, score
            if best_score is None or score < best_score:
                best_candidate = (cand, score)
                best_score = score
        if best_candidate is not None:
            return best_candidate
        # Fallback: random candidate with unknown score
        cand = self._rng.choice(target_ids)
        score = self._compute_similarity(hpo_id, cand) or 0.0
        return cand, score

    def _sample_two_sentences(self, sentences: Sequence[str]) -> Tuple[str, str]:
        if not sentences:
            return "", ""
        if len(sentences) == 1:
            # Reuse the only sentence so we still generate a positive pair
            return sentences[0], sentences[0]
        first, second = self._rng.sample(sentences, 2) if len(sentences) >= 2 else (sentences[0], sentences[0])
        return first, second

    def sample_structured_pairs(
        self,
        inventory: Dict[str, List[str]],
        passes: int = 1,
        hard_min_sim: float = 0.4,
        hard_max_sim: float = 0.7,
        easy_max_sim: float = 0.2,
        hard_attempts: int = 50,
        easy_attempts: int = 50,
    ) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
        """
        Build pairs deterministically per HPO term:
        - Positive: term with itself (different sentences when available)
        - Hard negative/intermediate: similar term (similarity in [hard_min_sim, hard_max_sim])
        - Easy negative: dissimilar term (similarity <= easy_max_sim)

        Returns: (df, counts, missing_counts)
        """
        if passes <= 0:
            raise ValueError("passes must be positive.")
        if len(inventory) < 2:
            raise RuntimeError("Need at least two HPO terms with sentences to build pairs.")

        hpo_ids = list(inventory.keys())
        pool_ids = [hid for hid, sents in inventory.items() if sents]
        if len(pool_ids) < 2:
            raise RuntimeError("Not enough HPO terms with sentences after filtering.")

        records: List[PairRecord] = []
        counts = {"positive": 0, "hard": 0, "easy": 0}
        missing = {"positive": 0, "hard": 0, "easy": 0}

        total_iters = passes * len(pool_ids)
        progress = tqdm(total=total_iters, desc="Structured sampling", unit="term") if self.show_progress else None

        for _ in range(passes):
            shuffled_ids = pool_ids[:]
            self._rng.shuffle(shuffled_ids)
            for hpo_id in shuffled_ids:
                sentences = inventory.get(hpo_id, [])
                if not sentences:
                    missing["positive"] += 1
                    missing["hard"] += 1
                    missing["easy"] += 1
                    continue

                # Positive
                s1, s2 = self._sample_two_sentences(sentences)
                if s1 and s2:
                    records.append(
                        PairRecord(
                            sentence1=s1,
                            sentence2=s2,
                            hpo_id1=hpo_id,
                            hpo_id2=hpo_id,
                            gold_similarity=1.0,
                        )
                    )
                    counts["positive"] += 1
                else:
                    missing["positive"] += 1

                # Hard negative/intermediate
                hard_candidate = self._select_hard_candidate(
                    hpo_id,
                    pool_ids,
                    hard_min_sim=hard_min_sim,
                    hard_max_sim=hard_max_sim,
                    max_attempts=hard_attempts,
                )
                if hard_candidate is not None:
                    cand_id, score = hard_candidate
                    records.append(
                        PairRecord(
                            sentence1=self._rng.choice(sentences),
                            sentence2=self._rng.choice(inventory[cand_id]),
                            hpo_id1=hpo_id,
                            hpo_id2=cand_id,
                            gold_similarity=score,
                        )
                    )
                    counts["hard"] += 1
                else:
                    missing["hard"] += 1

                # Easy negative
                easy_candidate = self._select_easy_candidate(
                    hpo_id,
                    pool_ids,
                    easy_max_sim=easy_max_sim,
                    max_attempts=easy_attempts,
                )
                if easy_candidate is not None:
                    cand_id, score = easy_candidate
                    records.append(
                        PairRecord(
                            sentence1=self._rng.choice(sentences),
                            sentence2=self._rng.choice(inventory[cand_id]),
                            hpo_id1=hpo_id,
                            hpo_id2=cand_id,
                            gold_similarity=score,
                        )
                    )
                    counts["easy"] += 1
                else:
                    missing["easy"] += 1

                if progress is not None:
                    progress.update(1)

        if progress is not None:
            progress.close()

        df = pd.DataFrame([r.__dict__ for r in records])
        return df, counts, missing

    def _get_term(self, hpo_id: str) -> Optional[pyhpo.term.HPOTerm]:
        """Retrieve and cache HPOTerm instances."""
        if hpo_id not in self._term_cache:
            try:
                self._term_cache[hpo_id] = self._ontology.get_hpo_object(hpo_id)
            except RuntimeError:
                self._term_cache[hpo_id] = None
        return self._term_cache[hpo_id]

    @staticmethod
    def save_pairs(df: pd.DataFrame, path: str) -> None:
        """Persist the resulting dataset to disk (Parquet or CSV)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if path.endswith(".parquet"):
            df.to_parquet(path, index=False)
        elif path.endswith(".csv"):
            df.to_csv(path, index=False)
        else:
            raise ValueError("Unsupported output format. Use .parquet or .csv")
        logging.info("Saved %d pairs to %s", len(df), path)


def split_pairs_by_hpo(
    df: pd.DataFrame,
    splits: Iterable[Tuple[str, float]],
    seed: int = 13,
    max_attempts: int = 100,
) -> Tuple[Dict[str, pd.DataFrame], int]:
    """
    Partition the pair dataset so that no HPO term leaks between splits.

    Args:
        df: DataFrame with columns `hpo_id1` and `hpo_id2`.
        splits: iterable of (name, ratio). Ratios must sum to 1.0.
        seed: random seed used when assigning terms.
        max_attempts: number of shuffling attempts to satisfy non-empty splits.

    Returns:
        (split_dfs, dropped) where split_dfs[name] = partitioned dataframe and
        `dropped` indicates how many rows were discarded because their HPO
        terms belonged to different splits.
    """
    if df.empty:
        raise ValueError("Dataset is empty; cannot create splits.")

    split_items = list(splits)
    if not split_items:
        raise ValueError("No splits specified.")

    total_ratio = sum(ratio for _, ratio in split_items)
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0 (got {total_ratio}).")

    unique_ids = sorted(set(df["hpo_id1"]).union(df["hpo_id2"]))
    if not unique_ids:
        raise ValueError("No HPO identifiers present in dataset.")

    rng = random.Random(seed)
    total_terms = len(unique_ids)

    def _targets() -> List[Tuple[str, int]]:
        counts: List[Tuple[str, int]] = []
        remaining = total_terms
        for index, (name, ratio) in enumerate(split_items):
            if index == len(split_items) - 1:
                count = remaining
            else:
                count = min(
                    remaining,
                    max(0, int(round(ratio * total_terms))),
                )
                remaining -= count
            counts.append((name, count))
        # Adjust in case rounding consumed all terms before reaching the end
        distributed = sum(count for _, count in counts)
        if distributed < total_terms:
            # Add missing terms to the largest split
            largest_split = max(counts, key=lambda item: item[1])[0]
            counts = [
                (name, count + (total_terms - distributed) if name == largest_split else count)
                for name, count in counts
            ]
        return counts

    def _materialize(mapping: Dict[str, str]) -> Tuple[Dict[str, pd.DataFrame], int]:
        split_indices: Dict[str, List[int]] = {name: [] for name, _ in split_items}
        dropped_pairs = 0
        for idx, (hpo_id1, hpo_id2) in enumerate(zip(df["hpo_id1"], df["hpo_id2"])):
            split_name = mapping.get(hpo_id1)
            if split_name is not None and split_name == mapping.get(hpo_id2):
                split_indices[split_name].append(idx)
            else:
                dropped_pairs += 1
        split_dfs = {
            name: df.iloc[indices].reset_index(drop=True)
            for name, indices in split_indices.items()
        }
        return split_dfs, dropped_pairs

    targets = _targets()
    best_result: Optional[Tuple[Dict[str, pd.DataFrame], int]] = None
    best_min_pairs = -1
    best_total_pairs = -1

    for attempt in range(max_attempts):
        shuffled = unique_ids[:]
        rng.shuffle(shuffled)
        mapping: Dict[str, str] = {}
        cursor = 0
        for name, count in targets:
            for hpo_id in shuffled[cursor : cursor + count]:
                mapping[hpo_id] = name
            cursor += count
        split_dfs, dropped = _materialize(mapping)
        lengths = [len(part) for part in split_dfs.values()]
        min_pairs = min(lengths) if lengths else 0
        total_pairs = sum(lengths)
        if (
            best_result is None
            or total_pairs > best_total_pairs
            or (total_pairs == best_total_pairs and min_pairs > best_min_pairs)
        ):
            best_result = (split_dfs, dropped)
            best_total_pairs = total_pairs
            best_min_pairs = min_pairs
        # Consider a split satisfied if all non-zero targets have at least one pair
        non_zero_targets = [name for name, count in targets if count > 0]
        if all(len(split_dfs[name]) > 0 for name in non_zero_targets):
            return split_dfs, dropped

    assert best_result is not None
    return best_result


def _allocate_counts(total: int, split_plan: Sequence[Tuple[str, float]]) -> Dict[str, int]:
    if total <= 0:
        return {name: 0 for name, _ in split_plan}

    counts: Dict[str, int] = {}
    fractional: List[Tuple[float, str]] = []
    for name, ratio in split_plan:
        exact = total * ratio
        floor_val = int(math.floor(exact))
        counts[name] = floor_val
        fractional.append((exact - floor_val, name))

    assigned = sum(counts.values())
    remainder = total - assigned
    fractional.sort(reverse=True)
    idx = 0
    while remainder > 0 and fractional:
        _, name = fractional[idx % len(fractional)]
        counts[name] += 1
        remainder -= 1
        idx += 1
    return counts


def assign_hpo_ids_to_splits(
    hpo_ids: Sequence[str],
    split_plan: Sequence[Tuple[str, float]],
    seed: int,
) -> Dict[str, List[str]]:
    if not hpo_ids:
        raise ValueError("No HPO identifiers available to distribute.")
    counts = _allocate_counts(len(hpo_ids), split_plan)
    rng = random.Random(seed)
    shuffled = list(hpo_ids)
    rng.shuffle(shuffled)

    assignments: Dict[str, List[str]] = {}
    cursor = 0
    for name, _ in split_plan:
        size = counts.get(name, 0)
        assignments[name] = shuffled[cursor : cursor + size]
        cursor += size
    return assignments


def sample_pairs_with_preassigned_splits(
    generator: HPOSentencePairGenerator,
    inventory: Dict[str, List[str]],
    total_targets: Dict[str, int],
    split_plan: Sequence[Tuple[str, float]],
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Dict[str, int]]]:
    if not split_plan:
        raise ValueError("Split plan cannot be empty.")

    split_target_counts: Dict[str, Dict[str, int]] = {name: {} for name, _ in split_plan}
    for bucket, total in total_targets.items():
        allocation = _allocate_counts(total, split_plan)
        for name, _ in split_plan:
            split_target_counts[name][bucket] = allocation.get(name, 0)

    assignments = assign_hpo_ids_to_splits(list(inventory.keys()), split_plan, seed)

    split_frames: Dict[str, pd.DataFrame] = {}
    achieved_counts: Dict[str, Dict[str, int]] = {}
    combined_frames: List[pd.DataFrame] = []

    for name, ids in assignments.items():
        sub_inventory = {hpo_id: inventory[hpo_id] for hpo_id in ids if hpo_id in inventory}
        targets = split_target_counts.get(name, {})
        if sum(targets.values()) == 0 or len(sub_inventory) < 2:
            empty = pd.DataFrame(columns=["sentence1", "sentence2", "hpo_id1", "hpo_id2", "gold_similarity"])
            split_frames[name] = empty
            achieved_counts[name] = {"positive": 0, "intermediate": 0, "negative": 0}
            continue

        df, counts = generator.sample_pairs(
            inventory=sub_inventory,
            positive_target=targets.get("positive", 0),
            intermediate_target=targets.get("intermediate", 0),
            negative_target=targets.get("negative", 0),
        )
        split_frames[name] = df
        achieved_counts[name] = counts
        combined_frames.append(df)

    combined_df = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame()
    return combined_df, split_frames, achieved_counts


def sample_structured_pairs_with_preassigned_splits(
    generator: HPOSentencePairGenerator,
    inventory: Dict[str, List[str]],
    split_plan: Sequence[Tuple[str, float]],
    passes: int,
    hard_min_sim: float,
    hard_max_sim: float,
    easy_max_sim: float,
    hard_attempts: int,
    easy_attempts: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Dict[str, int]]]:
    if not split_plan:
        raise ValueError("Split plan cannot be empty.")
    assignments = assign_hpo_ids_to_splits(list(inventory.keys()), split_plan, seed)

    split_frames: Dict[str, pd.DataFrame] = {}
    achieved_counts: Dict[str, Dict[str, int]] = {}
    combined_frames: List[pd.DataFrame] = []

    for name, ids in assignments.items():
        sub_inventory = {hpo_id: inventory[hpo_id] for hpo_id in ids if hpo_id in inventory}
        if len(sub_inventory) < 2:
            empty = pd.DataFrame(columns=["sentence1", "sentence2", "hpo_id1", "hpo_id2", "gold_similarity"])
            split_frames[name] = empty
            achieved_counts[name] = {"positive": 0, "hard": 0, "easy": 0}
            continue

        df, counts, _missing = generator.sample_structured_pairs(
            inventory=sub_inventory,
            passes=passes,
            hard_min_sim=hard_min_sim,
            hard_max_sim=hard_max_sim,
            easy_max_sim=easy_max_sim,
            hard_attempts=hard_attempts,
            easy_attempts=easy_attempts,
        )
        split_frames[name] = df
        achieved_counts[name] = counts
        combined_frames.append(df)

    combined_df = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame()
    return combined_df, split_frames, achieved_counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate sentence pairs scored with PyHPO.")
    parser.add_argument(
        "--dataset",
        default="data/hpo_sentences.parquet_dir",
        help="Path to the partitioned Parquet dataset from Phase 1.",
    )
    parser.add_argument(
        "--output",
        default="data/phase2/hpo_sentence_pairs.parquet",
        help="Destination file (.parquet or .csv).",
    )
    parser.add_argument("--method", default="lin", help="PyHPO similarity method (default: lin).")
    parser.add_argument(
        "--rbp-alpha",
        type=float,
        default=0.01,
        help="Alpha cap for RelativeBestPair similarity (ignored by other methods).",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for positive pairs.",
    )
    parser.add_argument(
        "--negative-threshold",
        type=float,
        default=0.3,
        help="Similarity threshold for negative pairs.",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=["structured", "bucketed"],
        default="structured",
        help="structured: per-term positive/hard/easy triplets; bucketed: legacy threshold-based sampling.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Number of passes over all HPO terms when using structured sampling.",
    )
    parser.add_argument(
        "--hard-min-sim",
        type=float,
        default=0.4,
        help="Lower bound for hard negative similarity (structured sampling).",
    )
    parser.add_argument(
        "--hard-max-sim",
        type=float,
        default=0.7,
        help="Upper bound for hard negative similarity (structured sampling).",
    )
    parser.add_argument(
        "--easy-max-sim",
        type=float,
        default=0.2,
        help="Maximum similarity for easy negatives (structured sampling).",
    )
    parser.add_argument(
        "--hard-attempts",
        type=int,
        default=60,
        help="Attempts to find a hard negative candidate per term (structured sampling).",
    )
    parser.add_argument(
        "--easy-attempts",
        type=int,
        default=60,
        help="Attempts to find an easy negative candidate per term (structured sampling).",
    )
    parser.add_argument(
        "--positive-target",
        type=int,
        default=8000,
        help="Number of highly similar pairs to sample.",
    )
    parser.add_argument(
        "--intermediate-target",
        type=int,
        default=12000,
        help="Number of medium similarity pairs to sample.",
    )
    parser.add_argument(
        "--negative-target",
        type=int,
        default=8000,
        help="Number of dissimilar pairs to sample.",
    )
    parser.add_argument(
        "--max-sentences-per-hpo",
        type=int,
        default=100,
        help="Cap of sentences stored per HPO term (controls memory).",
    )
    parser.add_argument(
        "--min-sentences-per-hpo",
        type=int,
        default=1,
        help="Discard HPO terms with fewer than this many sentences.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="PyArrow batch size when streaming the Parquet dataset.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars (useful for batch runs).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of HPO terms assigned to validation split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of HPO terms assigned to test split.",
    )
    parser.add_argument(
        "--split-output-dir",
        default="data/phase2/splits",
        help="Directory where split parquet files (train/val/test) will be stored.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Skip the train/val/test split step.",
    )
    parser.add_argument(
        "--preassign-splits",
        action="store_true",
        help="Assign HPO terms to splits before sampling to avoid cross-split leakage.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    generator = HPOSentencePairGenerator(
        dataset_path=args.dataset,
        method=args.method,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
        max_sentences_per_hpo=args.max_sentences_per_hpo,
        min_sentences_per_hpo=args.min_sentences_per_hpo,
        batch_size=args.batch_size,
        seed=args.seed,
        show_progress=not args.no_progress,
        rbp_alpha=args.rbp_alpha,
    )

    inventory = generator.load_sentence_inventory()
    total_targets = {
        "positive": args.positive_target,
        "intermediate": args.intermediate_target,
        "negative": args.negative_target,
    }
    train_ratio = 1.0 - args.val_ratio - args.test_ratio
    if train_ratio <= 0:
        raise ValueError("Invalid split ratios: train proportion must be positive.")

    split_plan = [("train", train_ratio), ("val", args.val_ratio), ("test", args.test_ratio)]

    if args.sampling_strategy == "structured":
        if args.preassign_splits and not args.no_split:
            combined_df, split_frames, achieved = sample_structured_pairs_with_preassigned_splits(
                generator,
                inventory,
                split_plan,
                passes=args.passes,
                hard_min_sim=args.hard_min_sim,
                hard_max_sim=args.hard_max_sim,
                easy_max_sim=args.easy_max_sim,
                hard_attempts=args.hard_attempts,
                easy_attempts=args.easy_attempts,
                seed=args.seed,
            )
            total_counts = {"positive": 0, "hard": 0, "easy": 0}
            for name, counts in achieved.items():
                logging.info(
                    "Split %s counts - positive: %d, hard: %d, easy: %d",
                    name,
                    counts.get("positive", 0),
                    counts.get("hard", 0),
                    counts.get("easy", 0),
                )
                for bucket in total_counts:
                    total_counts[bucket] += counts.get(bucket, 0)

            logging.info(
                "Aggregate counts - positive: %d, hard: %d, easy: %d",
                total_counts["positive"],
                total_counts["hard"],
                total_counts["easy"],
            )
            generator.save_pairs(combined_df, args.output)

            os.makedirs(args.split_output_dir, exist_ok=True)
            for name, split_df in split_frames.items():
                split_path = os.path.join(args.split_output_dir, f"{name}.parquet")
                split_df.to_parquet(split_path, index=False)
                logging.info("Saved %d pairs to %s", len(split_df), split_path)
            return

        df, counts, missing = generator.sample_structured_pairs(
            inventory=inventory,
            passes=args.passes,
            hard_min_sim=args.hard_min_sim,
            hard_max_sim=args.hard_max_sim,
            easy_max_sim=args.easy_max_sim,
            hard_attempts=args.hard_attempts,
            easy_attempts=args.easy_attempts,
        )
        logging.info(
            "Structured counts - positive: %d, hard: %d, easy: %d (missing: %s)",
            counts["positive"],
            counts["hard"],
            counts["easy"],
            missing,
        )
        generator.save_pairs(df, args.output)

        if args.no_split:
            return

        split_dfs, dropped = split_pairs_by_hpo(df, split_plan, seed=args.seed)
        if dropped:
            logging.warning("Dropped %d pairs because their HPO IDs spanned splits.", dropped)

        os.makedirs(args.split_output_dir, exist_ok=True)
        for name, split_df in split_dfs.items():
            split_path = os.path.join(args.split_output_dir, f"{name}.parquet")
            split_df.to_parquet(split_path, index=False)
            logging.info("Saved %d pairs to %s", len(split_df), split_path)
        return

    # Legacy bucketed sampling
    if args.preassign_splits and not args.no_split:
        combined_df, split_frames, achieved = sample_pairs_with_preassigned_splits(
            generator,
            inventory,
            total_targets,
            split_plan,
            seed=args.seed,
        )
        total_counts = {"positive": 0, "intermediate": 0, "negative": 0}
        for name, counts in achieved.items():
            logging.info(
                "Split %s counts - positive: %d, intermediate: %d, negative: %d",
                name,
                counts.get("positive", 0),
                counts.get("intermediate", 0),
                counts.get("negative", 0),
            )
            for bucket in total_counts:
                total_counts[bucket] += counts.get(bucket, 0)

        logging.info(
            "Aggregate counts - positive: %d, intermediate: %d, negative: %d",
            total_counts["positive"],
            total_counts["intermediate"],
            total_counts["negative"],
        )
        generator.save_pairs(combined_df, args.output)

        os.makedirs(args.split_output_dir, exist_ok=True)
        for name, split_df in split_frames.items():
            split_path = os.path.join(args.split_output_dir, f"{name}.parquet")
            split_df.to_parquet(split_path, index=False)
            logging.info("Saved %d pairs to %s", len(split_df), split_path)
        return

    df, counts = generator.sample_pairs(
        inventory=inventory,
        positive_target=args.positive_target,
        intermediate_target=args.intermediate_target,
        negative_target=args.negative_target,
    )
    logging.info(
        "Final counts - positive: %d, intermediate: %d, negative: %d",
        counts["positive"],
        counts["intermediate"],
        counts["negative"],
    )
    generator.save_pairs(df, args.output)

    if args.no_split:
        return

    split_dfs, dropped = split_pairs_by_hpo(df, split_plan, seed=args.seed)
    if dropped:
        logging.warning("Dropped %d pairs because their HPO IDs spanned splits.", dropped)

    os.makedirs(args.split_output_dir, exist_ok=True)
    for name, split_df in split_dfs.items():
        split_path = os.path.join(args.split_output_dir, f"{name}.parquet")
        split_df.to_parquet(split_path, index=False)
        logging.info("Saved %d pairs to %s", len(split_df), split_path)


if __name__ == "__main__":
    main()
