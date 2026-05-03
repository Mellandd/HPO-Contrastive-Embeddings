"""
Utilities for computing alternative HPO similarity metrics that are not
available directly within PyHPO.

Currently includes:
    - Relative Best Pair (RBP) similarity as described by Gong et al.
      (BMC Bioinformatics 2018), adapted to return a symmetric score
      between two HPOTerms scaled to [0, 1].
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Set

import pyhpo

_ONTOLOGY = pyhpo.Ontology()


def _format_disease_id(prefix: str, disease_id: int | str | None) -> str | None:
    if disease_id is None:
        return None
    return f"{prefix}:{disease_id}"


def _direct_disease_ids(term: pyhpo.term.HPOTerm) -> Set[str]:
    """Collect disease identifiers directly annotated to ``term``."""
    diseases: Set[str] = set()

    def _add(entries: Iterable, prefix: str) -> None:
        for entry in entries or []:
            formatted = _format_disease_id(prefix, getattr(entry, "id", None))
            if formatted:
                diseases.add(formatted)

    _add(term.omim_diseases, "OMIM")
    _add(term.orpha_diseases, "ORPHA")
    _add(term.decipher_diseases, "DECIPHER")
    return diseases


@lru_cache(maxsize=None)
def _descendant_diseases(hpo_id: str) -> frozenset[str]:
    """Return all diseases annotated to ``hpo_id`` or any descendant term."""
    try:
        term = _ONTOLOGY.get_hpo_object(hpo_id)
    except RuntimeError:
        return frozenset()

    diseases: Set[str] = set()
    stack = [term]
    visited: Set[str] = set()
    while stack:
        current = stack.pop()
        if current.id in visited:
            continue
        visited.add(current.id)
        diseases.update(_direct_disease_ids(current))
        stack.extend(child for child in current.children or [])
    return frozenset(diseases)


def _directional_rbp_score(
    source_diseases: frozenset[str],
    target_diseases: frozenset[str],
    alpha: float,
) -> float:
    if not source_diseases or not target_diseases:
        return 0.0
    intersection_size = len(source_diseases & target_diseases)
    if intersection_size == 0:
        return 0.0
    coverage = intersection_size / len(source_diseases)
    capped = min(alpha, 1.0 / len(target_diseases))
    return coverage * capped


def relative_best_pair(
    term1: pyhpo.term.HPOTerm,
    term2: pyhpo.term.HPOTerm,
    alpha: float = 0.01,
) -> float:
    """
    Compute the Relative Best Pair similarity between two HPO terms.

    The score is derived from the definition by Gong et al. by treating the
    set of diseases annotated to each term (including descendants) as the
    support distribution and comparing their overlap with directional scores
    clipped by ``alpha``. The symmetric result is scaled to [0, 1].

    Args:
        term1: First HPOTerm.
        term2: Second HPOTerm.
        alpha: Contribution cap per the original method (default: 0.01).

    Returns:
        A float in [0, 1] indicating semantic similarity.
    """
    diseases1 = _descendant_diseases(term1.id)
    diseases2 = _descendant_diseases(term2.id)
    if not diseases1 or not diseases2:
        return 0.0

    dir1 = _directional_rbp_score(diseases1, diseases2, alpha)
    dir2 = _directional_rbp_score(diseases2, diseases1, alpha)
    combined = 0.5 * (dir1 + dir2)

    if alpha <= 0:
        return min(1.0, combined)
    return min(1.0, combined / alpha)


__all__ = ["relative_best_pair"]
