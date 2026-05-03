"""Extraction of relevant sentences from JATS XML from PMC.
- Robust parsing of multiple articles per response
- PMCID retrieval per article
- Sentence segmentation (optional spaCy, NLTK fallback)
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET
import re
import hashlib
import logging

from . import config

# Tokenisers
_nlp = None
if config.USE_SPACY:
    try:
        import spacy
        _nlp = spacy.load(config.SPACY_MODEL, disable=["ner", "tagger", "lemmatizer"])  # senter only
        if not _nlp.has_pipe("senter"):
            _nlp.add_pipe("sentencizer")
    except Exception as e:
        logging.warning(f"spaCy not available ({e}), falling back to NLTK.")
        _nlp = None

if _nlp is None:
    import nltk
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize


_NUMERIC_BRACKET_RE = re.compile(r"\[\s*(?:\d+(?:\s*[-–]\s*\d+)?)(?:\s*[,;]\s*(?:\d+(?:\s*[-–]\s*\d+)?))*\s*\]")
_NUMERIC_PAREN_RE = re.compile(r"\(\s*(?:\d+(?:\s*[-–]\s*\d+)?)(?:\s*[,;]\s*(?:\d+(?:\s*[-–]\s*\d+)?))*\s*\)")
_PAREN_ETAL_RE = re.compile(r"\(\s*[^)]*et\s+al\.[^)]*\)", flags=re.IGNORECASE)
_PAREN_YEAR_RE = re.compile(r"\(\s*[A-Z][A-Za-z]+[^)]*\d{4}[^)]*\)")


def _clean_sentence_text(text: str) -> str:
    """Remove numeric references and simple citations."""
    cleaned = _NUMERIC_BRACKET_RE.sub("", text)
    cleaned = _NUMERIC_PAREN_RE.sub("", cleaned)
    cleaned = _PAREN_ETAL_RE.sub("", cleaned)
    cleaned = _PAREN_YEAR_RE.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip()


def _iter_articles(root: ET.Element):
    """Iterate articles in a response containing multiple <article> elements."""
    # Use namespace wildcard for broader compatibility
    for art in root.findall(".//{*}article"):
        yield art


def _get_pmcid(article_el: ET.Element) -> Optional[str]:
    # <article-id pub-id-type="pmcid">PMC1234567</article-id>
    for aid in article_el.findall(".//{*}article-id"):
        if aid.get("pub-id-type") == "pmcid" and aid.text:
            # Normalize by removing spaces, ensuring "PMC" prefix
            text = aid.text.strip()
            return text if text.startswith("PMC") else f"PMC{text}"
    return None


def _strip_tag(tag: str) -> str:
    """Return the local tag name (without namespace)."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _section_title(node: ET.Element) -> str | None:
    """Get the human-readable title of a section (title, sec-type, or tag)."""
    title_el = node.find("./{*}title")
    if title_el is not None:
        txt = _clean_text(" ".join(title_el.itertext()))
        if txt:
            return txt
    sec_type = node.get("sec-type")
    if sec_type:
        return sec_type
    tag = _strip_tag(node.tag)
    if tag not in {"sec", "body", "abstract"}:
        return tag
    return None


def _walk_paragraphs(node: ET.Element, section_stack: List[str]):
    """Depth-first walk of the hierarchy looking for <p> elements; yields (section_path, text)."""
    for child in list(node):
        tag = _strip_tag(child.tag)
        if tag == "sec":
            title = _section_title(child)
            new_stack = section_stack + ([title] if title else [])
            yield from _walk_paragraphs(child, new_stack)
        elif tag == "p":
            txt = _clean_text(" ".join(child.itertext()))
            if txt:
                label_parts = [part for part in section_stack if part]
                section_label = " > ".join(label_parts) if label_parts else "body"
                yield (section_label, txt)
        elif tag in {"fig", "table-wrap", "ref-list"}:
            # These sections rarely contribute narrative sentences; skip them.
            continue
        else:
            yield from _walk_paragraphs(child, section_stack)


def _iter_paragraph_texts(article_el: ET.Element) -> Iterable[Tuple[str, str]]:
    """Yield (section_hint, paragraph_text) from the body and abstract.
    section_hint: "abstract" | "body" | other parent tag if applicable.
    """
    # Abstract (may be multiple, e.g. structured abstract)
    for abs_el in article_el.findall(".//{*}abstract"):
        title = _section_title(abs_el)
        stack = ["abstract"]
        if title and title.lower() != "abstract":
            stack.append(title)
        yield from _walk_paragraphs(abs_el, stack)

    # Main body
    for body in article_el.findall(".//{*}body"):
        yield from _walk_paragraphs(body, ["body"])

    # Back matter (e.g. conclusions, acknowledgements) — optional depending on needs
    for back in article_el.findall(".//{*}back"):
        title = _section_title(back) or "back"
        yield from _walk_paragraphs(back, [title])


def _sentencize(text: str) -> List[str]:
    sentences: List[str] = []
    if _nlp is not None:
        doc = _nlp(text)
        for sent in doc.sents:
            content = sent.text.strip()
            if len(content) < config.SENTENCE_MIN_LEN:
                continue
            content = _clean_sentence_text(content)
            if content:
                sentences.append(content)
    else:
        from nltk.tokenize import sent_tokenize
        for raw in sent_tokenize(text):
            content = raw.strip()
            if len(content) < config.SENTENCE_MIN_LEN:
                continue
            content = _clean_sentence_text(content)
            if content:
                sentences.append(content)
    return sentences


def sentence_hash(sentence: str) -> str:
    return hashlib.md5(sentence.encode("utf-8")).hexdigest()


def extract_relevant_sentences(
    xml_str: str,
    hpo_label: str,
    hpo_id: str,
    pattern: Optional[re.Pattern] = None,
) -> List[dict]:
    """Return sentence records that contain the HPO term.

    Each record: {hpo_id, hpo_label, pmcid, section, sentence, hash}
    """
    if not xml_str:
        return []

    # Default regex: whole-word match, case-insensitive
    if pattern is None:
        pattern = re.compile(rf"\b{re.escape(hpo_label)}\b", re.IGNORECASE)

    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        logging.error(f"Invalid XML: {e}")
        return []

    out: List[dict] = []
    for art in _iter_articles(root):
        pmcid = _get_pmcid(art)
        if not pmcid:
            continue
        for section, para in _iter_paragraph_texts(art):
            for sent in _sentencize(para):
                if pattern.search(sent):
                    out.append({
                        "hpo_id": hpo_id,
                        "hpo_label": hpo_label,
                        "pmcid": pmcid,
                        "section": section,
                        "sentence": sent,
                        "hash": sentence_hash(sent),
                    })
    return out


def extract_sentences_from_xml(xml_str: str) -> List[Dict[str, Optional[str]]]:
    """
    Parse PMC JATS XML and return a list of sentences with metadata:
    [{ "pmcid": "PMC1234567", "section": "body"|"abstract", "sentence": "<text>" }, ...]
    """
    if not xml_str:
        return []

    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        logging.error(f"Invalid XML: {e}")
        return []

    out: List[Dict[str, Optional[str]]] = []
    for art in _iter_articles(root):
        pmcid = _get_pmcid(art)  # string with "PMC" prefix
        for section, para in _iter_paragraph_texts(art):
            # Tokenise into sentences (uses spaCy if config.USE_SPACY=True; otherwise NLTK)
            for sent in _sentencize(para):
                out.append({
                    "pmcid": str(pmcid) if pmcid is not None else None,
                    "section": section,
                    "sentence": sent.strip(),
                })
    return out


def extract_full_text_from_xml(xml_str: str) -> str:
    """Return all text from the article concatenated (abstract + body)."""
    if not xml_str:
        return ""
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return ""
    texts = []
    for art in _iter_articles(root):
        for _, para in _iter_paragraph_texts(art):
            texts.append(para.strip())
    return " ".join(texts)
