"""Load HPO terms (ID, label and synonyms when available).
Uses PyHPO. Returns a list of dicts with fields useful for searching.
"""
from __future__ import annotations

from collections import deque
from typing import List, Set
import logging

import pyhpo

_SYNONYM_ATTRS = ("synonyms", "alt_names", "alternative_names")


def _term_to_entry(term: pyhpo.term.HPOTerm, include_synonyms: bool) -> dict:
    names: Set[str] = set()
    primary = (term.name or "").strip()
    if primary:
        names.add(primary)
    if include_synonyms:
        for attr in _SYNONYM_ATTRS:
            syns = getattr(term, attr, None)
            if not syns:
                continue
            try:
                for raw in syns:
                    text = (raw or "").strip()
                    if text:
                        names.add(text)
            except Exception:
                # Some library versions return non-iterable types; silently skip
                continue
    sorted_names = sorted(names) if include_synonyms else [primary]
    return {
        "hpo_id": term.id,
        "label": term.name,
        "names": sorted_names,
    }


def load_hpo_terms(include_synonyms: bool = True) -> List[dict]:
    ontology = pyhpo.Ontology()
    terms: List[dict] = []
    for term in ontology:
        if (term.name or "").strip().lower() == "all":
            logging.info("Skipping 'All' phenotype")
            continue
        terms.append(_term_to_entry(term, include_synonyms))
    return terms


def load_hpo_subontology_terms(
    root_term: str,
    include_synonyms: bool = True,
    include_root: bool = True,
) -> List[dict]:
    """
    Returns all descendant terms of a given root phenotype.
    `root_term` can be an HPO identifier (HP:XXXXXXX) or the phenotype label.
    """
    ontology = pyhpo.Ontology()
    try:
        root = ontology.get_hpo_object(root_term)
    except RuntimeError as exc:
        raise ValueError(f"Root phenotype {root_term!r} not found") from exc

    collected: List[dict] = []
    visited: Set[str] = set()
    queue = deque([root])

    while queue:
        term = queue.popleft()
        if term.id in visited:
            continue
        visited.add(term.id)

        is_all_term = (term.name or "").strip().lower() == "all"
        if (term is not root or include_root) and not is_all_term:
            collected.append(_term_to_entry(term, include_synonyms))
        elif is_all_term:
            logging.info("Skipping 'All' phenotype")

        children = getattr(term, "children", set())
        for child in sorted(children, key=lambda item: item.id):
            if child.id not in visited:
                queue.append(child)

    return collected
