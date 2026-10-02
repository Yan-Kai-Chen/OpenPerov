"""Original deterministic scoring helper; portable extraction, algorithms unchanged.
Source SHA256: 4ae2d8c0bbcaf54b45e51251d01908de396f98c5f4757720a23fea8444bdae33
No machine paths, historical CLI or private reference data included.
"""
from __future__ import annotations
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "can",
    "could", "does", "for", "from", "has", "have", "how", "in", "into",
    "is", "it", "its", "may", "more", "most", "not", "of", "on", "or",
    "should", "such", "than", "that", "the", "their", "there", "these",
    "this", "those", "through", "to", "under", "use", "uses", "using",
    "via", "was", "were", "what", "when", "where", "which", "while",
    "why", "with", "without", "would", "answer", "paper", "study",
    "result", "results", "according", "review", "example", "examples",
}

CANON = {
    "illumination": "light",
    "illuminated": "light",
    "photo": "light",
    "photogenerated": "carrier",
    "charge": "carrier",
    "charges": "carrier",
    "hole": "carrier",
    "electron": "carrier",
    "carriers": "carrier",
    "collection": "extraction",
    "collect": "extraction",
    "extract": "extraction",
    "screen": "screening",
    "screened": "screening",
    "flatten": "flattening",
    "flattened": "flattening",
    "band": "band",
    "bands": "band",
    "bandgap": "gap",
    "bandgaps": "gap",
    "energy": "energy",
    "recombination": "recombine",
    "recombine": "recombine",
    "trap": "defect",
    "traps": "defect",
    "defects": "defect",
    "stability": "stable",
    "stabilization": "stable",
    "degradation": "degrade",
    "degraded": "degrade",
    "segregation": "separate",
    "segregated": "separate",
    "separation": "separate",
    "phase": "phase",
    "halide": "halide",
    "iodide": "halide",
    "bromide": "halide",
    "thermal": "heat",
    "heating": "heat",
    "temperature": "heat",
    "transport": "transport",
    "mobility": "transport",
    "conductivity": "transport",
    "passivation": "passivate",
    "passivates": "passivate",
    "passivated": "passivate",
    "interface": "surface",
    "interfacial": "surface",
    "surface": "surface",
    "octahedral": "octahedra",
    "octahedron": "octahedra",
    "octahedra": "octahedra",
    "tilt": "tilting",
    "tilts": "tilting",
    "distortion": "distort",
    "distortions": "distort",
    "jahn": "jahn",
    "teller": "teller",
    "current": "current",
    "jsc": "current",
    "voc": "voltage",
    "voltage": "voltage",
    "efficiency": "efficiency",
    "pce": "efficiency",
    "eqe": "eqe",
    "pl": "pl",
    "giwaxs": "giwaxs",
    "nmr": "nmr",
    "dft": "dft",
}

REASONING_MARKERS = {
    "because", "therefore", "thus", "whereas", "while", "contrast",
    "compared", "indicates", "suggests", "supports", "mechanism",
    "however", "thereby", "leads", "causes", "consistent", "explains",
    "rather", "not", "instead",
}

def norm_text(text: str) -> str:
    text = re.sub(r"[^\x00-\x7F]+", " ", text or "")
    text = text.lower()
    text = re.sub(r"[^a-z0-9.+-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def stem(tok: str) -> str:
    if tok in CANON:
        return CANON[tok]
    for suffix in ("ization", "isation", "ations", "tion", "ing", "ed", "es", "s"):
        if len(tok) > len(suffix) + 4 and tok.endswith(suffix):
            tok = tok[: -len(suffix)]
            break
    return CANON.get(tok, tok)

def tokens(text: str) -> list[str]:
    raw = re.findall(r"\d+\.\d+|\d+|[a-z][a-z0-9+-]*", norm_text(text))
    out: list[str] = []
    for tok in raw:
        clean = tok.strip("+-")
        if not clean:
            continue
        if clean in STOPWORDS:
            continue
        if clean.isalpha() and len(clean) <= 2:
            continue
        out.append(stem(clean))
    return out

def token_set(text: str) -> set[str]:
    return set(tokens(text))

def token_counts(text: str) -> Counter[str]:
    return Counter(tokens(text))

def cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0

def coverage(needles: set[str], haystack: set[str]) -> float:
    return len(needles & haystack) / len(needles) if needles else 0.0

def numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", norm_text(text)))

def specific_terms(text: str) -> set[str]:
    terms = set()
    for match in re.findall(r"\b[A-Za-z]*\d+[A-Za-z0-9.+-]*\b|\b[A-Z]{2,}[A-Za-z0-9.+-]*\b", text or ""):
        if len(match) >= 2:
            terms.add(match.lower())
    terms.update(numbers(text))
    return terms

def round_half(score: float) -> float:
    return max(0.0, min(4.0, round(score * 2) / 2))

def semantic_key_fraction(key_points: list[str], answer: str) -> tuple[float, list[dict[str, Any]]]:
    answer_tokens = token_set(answer)
    answer_nums = numbers(answer)
    details: list[dict[str, Any]] = []
    credit = 0.0
    for kp in key_points:
        kp_tokens = token_set(kp)
        kp_nums = numbers(kp)
        frac = coverage(kp_tokens, answer_tokens)
        num_ok = bool(not kp_nums or (kp_nums & answer_nums))
        # A key point with explicit numbers should not receive full semantic
        # rescue credit unless a relevant number is present.
        if frac >= 0.50 and num_ok:
            value = 1.0
            level = "full"
        elif frac >= 0.28 or (len(kp_tokens) <= 3 and frac >= 0.34):
            value = 0.5
            level = "partial"
        else:
            value = 0.0
            level = "none"
        credit += value
        details.append(
            {
                "key_point": kp,
                "semantic_overlap": round(frac, 4),
                "numeric_required": bool(kp_nums),
                "numeric_ok": num_ok,
                "semantic_credit": value,
                "semantic_level": level,
            }
        )
    return (credit / len(key_points) if key_points else 0.0), details

def answer_features(answer: str, reference: str, key_points: list[str], source_support: str) -> dict[str, Any]:
    answer_counts = token_counts(answer)
    ref_counts = token_counts(reference + " " + " ".join(key_points))
    answer_set = set(answer_counts)
    ref_set = set(ref_counts)
    source_set = token_set(source_support)
    ref_cos = cosine(answer_counts, ref_counts)
    ref_cov = coverage(ref_set, answer_set)
    source_cov = coverage(source_set, answer_set)
    specific_ref = specific_terms(reference + " " + " ".join(key_points) + " " + source_support)
    specific_cov = coverage(specific_ref, specific_terms(answer)) if specific_ref else 0.0
    num_ref = numbers(reference + " " + " ".join(key_points))
    num_cov = coverage(num_ref, numbers(answer)) if num_ref else 1.0
    marker_count = sum(1 for marker in REASONING_MARKERS if marker in answer_set)
    key_frac, key_details = semantic_key_fraction(key_points, answer)
    return {
        "answer_token_count": sum(answer_counts.values()),
        "semantic_key_fraction": round(key_frac, 4),
        "semantic_key_details": key_details,
        "semantic_ref_cosine": round(ref_cos, 4),
        "semantic_ref_coverage": round(ref_cov, 4),
        "semantic_source_coverage": round(source_cov, 4),
        "semantic_specific_coverage": round(specific_cov, 4),
        "semantic_numeric_coverage": round(num_cov, 4),
        "reasoning_marker_count": marker_count,
    }
