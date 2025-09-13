from typing import List, Dict, Any, Tuple, Optional
from services.RAG.log_utils import get_logger, snapshot

log = get_logger("chunk")

def split_paragraphs(text: str) -> List[str]:
    import re
    text = re.sub(r"\r\n?", "\n", text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras if paras else ([text.strip()] if text.strip() else [])


def split_sentences(text: str) -> List[str]:
    """Naive sentence splitter suitable for technical text.

    Splits on ., !, ? followed by whitespace/newline. Keeps punctuation.
    Falls back to returning the whole text if no boundaries found.
    """
    import re
    # Normalize whitespace
    s = re.sub(r"\s+", " ", text).strip()
    if not s:
        return []
    parts = re.split(r"(?<=[.!?])\s+", s)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts if parts else [s]


def merge_paras(paras: List[str], min_chars: int, max_chars: int) -> List[str]:
    out, buf = [], ""
    for p in paras:
        if not buf:
            buf = p
            continue
        if len(buf) < min_chars or (len(buf) + 2 + len(p) <= max_chars):
            buf = f"{buf}\n\n{p}"
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def _group_paras(paras: List[str], group_size: int) -> List[str]:
    """Group paragraphs into fixed-size windows (non-overlapping).

    - group_size=2 -> [p1+p2, p3+p4, ...]
    - If odd count, final chunk is a single paragraph.
    """
    if group_size <= 1:
        return paras
    out: List[str] = []
    i = 0
    n = len(paras)
    while i < n:
        j = min(i + group_size, n)
        out.append("\n\n".join(paras[i:j]))
        i = j
    return out


def _group_paras_overlap(paras: List[str], group_size: int, overlap: int) -> List[str]:
    """Group paragraphs into fixed-size windows with paragraph-level overlap.

    - group_size=2, overlap=1 -> [p1+p2, p2+p3, p3+p4, ...]
    - stride is max(1, group_size - overlap)
    """
    if group_size <= 1 or overlap <= 0:
        return _group_paras(paras, group_size)
    out: List[str] = []
    n = len(paras)
    stride = max(1, group_size - overlap)
    i = 0
    while i < n:
        j = min(i + group_size, n)
        if i >= j:
            break
        out.append("\n\n".join(paras[i:j]))
        if j >= n:
            break
        i += stride
    return out


def chunk(
    text: str,
    min_chars: int = 200,
    max_chars: int = 1600,
    overlap: int = 80,
    paras_per_chunk: Optional[int] = None,
    paras_overlap: Optional[int] = None,
    sentence_overlap: Optional[int] = None,
) -> List[str]:
    """Create chunks from raw text.

    Defaults preserve previous behavior (char-based merging) unless
    `paras_per_chunk` is provided (>1), in which case paragraph grouping
    is used instead. When `paras_overlap` > 0, sliding windows are formed
    with paragraph-level overlap; otherwise non-overlapping windows.
    """
    paras = split_paragraphs(text)
    if paras_per_chunk and paras_per_chunk > 1:
        if paras_overlap and paras_overlap > 0:
            base = _group_paras_overlap(paras, paras_per_chunk, paras_overlap)
        else:
            base = _group_paras(paras, paras_per_chunk)
        # If sentence-level overlap requested, prepend the last N sentences
        # from the previous chunk to the current chunk.
        if sentence_overlap and sentence_overlap > 0 and len(base) > 1:
            out_chunks: List[str] = []
            prev_sents: List[str] = []
            for i, ck in enumerate(base):
                if i == 0:
                    out_chunks.append(ck)
                    prev_sents = split_sentences(ck)
                    continue
                tail = prev_sents[-min(sentence_overlap, len(prev_sents)):] if prev_sents else []
                prefix = (" ".join(tail)).strip()
                if prefix:
                    merged = f"{prefix}\n\n{ck}"
                else:
                    merged = ck
                out_chunks.append(merged)
                prev_sents = split_sentences(ck)
            log.info(f"[CHUNK] produced={len(out_chunks)} first='{snapshot(out_chunks[0]) if out_chunks else ''}'")
            return out_chunks
    else:
        base = merge_paras(paras, min_chars, max_chars)
    # Char-level overlap is only applied for char-based chunks.
    # When using paragraph grouping, skip char-tail overlap to avoid
    # producing partial-paragraph prefixes.
    if overlap <= 0 or len(base) <= 1 or (paras_per_chunk and paras_per_chunk > 1):
        log.info(f"[CHUNK] produced={len(base)} first='{snapshot(base[0]) if base else ''}'")
        return base
    out = [base[0]]
    for i in range(1, len(base)):
        tail = base[i-1][-overlap:]
        sp = tail.find(" ")
        if sp > 0:
            tail = tail[sp+1:]
        out.append(f"{tail} {base[i]}")
    log.info(f"[CHUNK] produced={len(out)} first='{snapshot(out[0]) if out else ''}'")
    return out


def sha1_text(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def dedupe(chunks: List[str]) -> Tuple[List[str], Dict[int, Tuple[int, str]]]:
    seen: Dict[str, int] = {}
    keep, dup = [], {}
    for i, c in enumerate(chunks):
        h = sha1_text(c)
        if h in seen:
            dup[i] = (seen[h], h)
        else:
            seen[h] = len(keep)
            keep.append(c)
    log.info(f"[CHUNK] dedupe orig={len(chunks)} uniq={len(keep)} dups={len(dup)}")
    return keep, dup
