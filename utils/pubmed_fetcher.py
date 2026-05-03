"""High-level functions for searching PMC and downloading full-text XML.
Uses Biopython Entrez with retries, backoff and rate limiting.
"""
from __future__ import annotations
from typing import Iterable, List, Optional
import time
import logging
from math import ceil
from Bio import Entrez
from Bio import Medline

from . import config

# Configure Entrez
Entrez.email = config.ENTREZ_EMAIL
if config.ENTREZ_API_KEY:
    Entrez.api_key = config.ENTREZ_API_KEY

_SLEEP = 1.0 / config.RATE_LIMIT_RPS  # seconds between calls

def _with_retries(fn, *args, **kwargs):
    """Execute a function with exponential backoff retries."""
    for attempt in range(config.MAX_RETRIES):
        try:
            res = fn(*args, **kwargs)
            time.sleep(_SLEEP)
            return res
        except Exception as e:
            delay = config.BACKOFF_BASE_SECONDS * (2 ** attempt)
            logging.warning(f"Attempt {attempt+1}/{config.MAX_RETRIES} failed: {e}. Retrying in {delay:.1f}s…")
            time.sleep(delay)
    raise RuntimeError("Maximum retries reached")


def search_pmc_open_access(term: str, retmax: Optional[int] = None) -> List[str]:
    """Search PMC Open Access IDs containing `term` (in title/abstract/indexed body).

    Returns: list of PMCIDs (numeric string, without 'PMC' prefix).
    """
    # Note: the open access filter in PMC is specified with "open access[filter]"
    query = f'"{term}" AND open access[filter]'
    if retmax is None:
        retmax = config.MAX_PMIDS_PER_TERM or 100000
    handle = _with_retries(Entrez.esearch, db="pmc", term=query, retmax=retmax)
    record = Entrez.read(handle)
    return list(record.get("IdList", []))


def fetch_pmc_fulltext_xml(pmc_ids: List[str]) -> Optional[str]:
    """Download full JATS XML for a list of PMCIDs (without the 'PMC' prefix).
    Returns an XML string potentially containing multiple <article> elements.
    """
    if not pmc_ids:
        return None
    id_str = ",".join(pmc_ids)
    handle = _with_retries(Entrez.efetch, db="pmc", id=id_str, retmode="xml")
    xml_data = handle.read()
    try:
        handle.close()
    except Exception:
        pass
    return xml_data


def batched_fetch_xml(pmc_ids: List[str], batch_size: int) -> Iterable[str]:
    """Yield XML batches for a large list of PMCIDs."""
    n = len(pmc_ids)
    for i in range(0, n, batch_size):
        batch = pmc_ids[i : i + batch_size]
        xml_str = fetch_pmc_fulltext_xml(batch)
        if xml_str:
            yield xml_str