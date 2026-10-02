"""Original deterministic scoring helper; portable extraction, algorithms unchanged.
Source SHA256: a08920d0c1f78cf686fa75077619edf6418f9eddf8f9dc1d42a82a2160c1461f
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
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "has", "have", "having",
    "here", "how", "in", "into", "is", "it", "its", "may", "more", "most", "not",
    "of", "on", "or", "rather", "should", "shows", "show", "shown", "such", "than",
    "that", "the", "their", "then", "there", "these", "this", "those", "through",
    "to", "under", "use", "uses", "using", "via", "was", "were", "what", "when",
    "where", "which", "while", "why", "with", "without", "would",
    "answer", "candidate", "context", "evidence", "paper", "study", "result", "results",
}

REASONING_MARKERS = {
    "because", "therefore", "thus", "whereas", "while", "contrast", "compared",
    "indicates", "suggests", "supports", "mechanism", "trade", "tradeoff", "rather",
    "however", "thereby", "leads", "causes", "consistent", "explains", "therefore",
}

def normalize_text(text: str) -> str:
    # Keep ASCII science tokens and numeric values; mojibake unit artifacts are treated as separators.
    text = text.replace("^-", " ")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9.+-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokens(text: str) -> list[str]:
    norm = normalize_text(text)
    raw = re.findall(r"\d+\.\d+|\d+|[a-z][a-z0-9+-]*", norm)
    out = []
    for tok in raw:
        clean = tok.strip("+-")
        if not clean:
            continue
        if clean in STOPWORDS:
            continue
        if clean.isalpha() and len(clean) <= 2:
            continue
        out.append(clean)
    return out

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
    if not needles:
        return 0.0
    return len(needles & haystack) / len(needles)

def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", normalize_text(text)))

def extract_specific_terms(text: str) -> set[str]:
    # Terms with digits and multi-letter uppercase terms are useful proxies for article-specific evidence.
    terms = set()
    for match in re.findall(r"\b[A-Za-z]*\d+[A-Za-z0-9.+-]*\b|\b[A-Z]{2,}[A-Za-z0-9.+-]*\b", text):
        if len(match) >= 2:
            terms.add(match.lower())
    terms.update(extract_numbers(text))
    return terms

def round_half(score: float) -> float:
    return max(0.0, min(4.0, round(score * 2) / 2))

def score_key_point(key_point: str, answer_tokens: set[str]) -> dict[str, Any]:
    kp_tokens = set(tokens(key_point))
    kp_numbers = extract_numbers(key_point)
    if not kp_tokens:
        frac = 0.0
    else:
        frac = coverage(kp_tokens, answer_tokens)
    number_bonus = 0.0
    if kp_numbers:
        number_bonus = 0.15 if kp_numbers & answer_tokens else -0.10
    adjusted = max(0.0, min(1.0, frac + number_bonus))
    if adjusted >= 0.56 or (len(kp_tokens) <= 3 and adjusted >= 0.67):
        credit = "full"
        hit = True
        value = 1.0
    elif adjusted >= 0.30:
        credit = "partial"
        hit = True
        value = 0.5
    else:
        credit = "none"
        hit = False
        value = 0.0
    return {
        "key_point": key_point,
        "hit": hit,
        "credit": credit,
        "overlap_fraction": round(frac, 4),
        "credit_value": value,
    }

def quality_label(value: float, strong: float, adequate: float, weak: float) -> str:
    if value >= strong:
        return "strong"
    if value >= adequate:
        return "adequate"
    if value >= weak:
        return "weak"
    return "absent"

def score_row(row: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("candidate_answer") or ""
    reference = row.get("reference_expected_answer") or ""
    source_support = row.get("source_support") or ""
    key_points = row.get("scoring_key_points") or []

    answer_counts = token_counts(answer)
    reference_counts = token_counts(reference)
    source_counts = token_counts(source_support)
    answer_set = set(answer_counts)
    reference_set = set(reference_counts)
    source_set = set(source_counts)

    key_hits = [score_key_point(kp, answer_set) for kp in key_points]
    key_credit = sum(hit["credit_value"] for hit in key_hits)
    key_fraction = key_credit / len(key_hits) if key_hits else 0.0
    key_point_score = 2.0 * key_fraction

    ref_cos = cosine(answer_counts, reference_counts)
    ref_cov = coverage(reference_set, answer_set)
    ref_signal = max(min(ref_cos / 0.45, 1.0), min(ref_cov / 0.45, 1.0))
    reference_fidelity_score = 0.8 * ref_signal

    ref_specific = extract_specific_terms(reference + " " + source_support + " " + " ".join(key_points))
    answer_specific = extract_specific_terms(answer)
    specific_cov = coverage(ref_specific, answer_specific) if ref_specific else 0.0
    source_cov = coverage(source_set, answer_set)
    numeric_ref = extract_numbers(reference + " " + " ".join(key_points))
    numeric_cov = coverage(numeric_ref, extract_numbers(answer)) if numeric_ref else 0.0
    evidence_signal = max(specific_cov, 0.65 * source_cov + 0.35 * numeric_cov)
    evidence_score = 0.8 * min(evidence_signal / 0.45, 1.0)

    answer_token_count = sum(answer_counts.values())
    marker_count = sum(1 for marker in REASONING_MARKERS if marker in answer_set)
    if answer_token_count >= 50 and marker_count >= 2:
        reasoning_score = 0.4
    elif answer_token_count >= 35 and marker_count >= 1:
        reasoning_score = 0.3
    elif answer_token_count >= 20:
        reasoning_score = 0.2
    else:
        reasoning_score = 0.0

    raw_score = key_point_score + reference_fidelity_score + evidence_score + reasoning_score

    flags = []
    cap = 4.0
    if answer_token_count < 18:
        flags.append("very_short_answer_cap_1")
        cap = min(cap, 1.0)
    if key_fraction < 0.25:
        flags.append("low_key_point_coverage_cap_2")
        cap = min(cap, 2.0)
    if ref_cov < 0.12 and ref_cos < 0.18:
        flags.append("low_reference_alignment_cap_2")
        cap = min(cap, 2.0)
    if evidence_signal < 0.08 and key_fraction < 0.50:
        flags.append("weak_article_specific_evidence_cap_2")
        cap = min(cap, 2.0)
    if not answer.strip():
        flags.append("empty_answer")
        cap = 0.0

    capped_score = min(raw_score, cap)
    final_score = round_half(capped_score)
    missing_key_points = [hit["key_point"] for hit in key_hits if hit["credit"] == "none"]
    partial_key_points = [hit["key_point"] for hit in key_hits if hit["credit"] == "partial"]

    evidence_quality = quality_label(evidence_signal, strong=0.40, adequate=0.24, weak=0.10)
    reasoning_quality = quality_label(reasoning_score, strong=0.4, adequate=0.3, weak=0.2)
    confidence_signal = min(key_fraction, max(ref_cos, ref_cov), max(evidence_signal, 0.01))
    if flags or confidence_signal < 0.12:
        scorer_confidence = "low"
    elif confidence_signal < 0.25:
        scorer_confidence = "medium"
    else:
        scorer_confidence = "high"

    needs_review = bool(flags) or final_score <= 2.5 or scorer_confidence != "high"
    rationale = (
        f"Local deterministic score={final_score:.1f}; key coverage={key_fraction:.2f}, "
        f"reference cosine={ref_cos:.2f}, reference coverage={ref_cov:.2f}, "
        f"article-specific evidence signal={evidence_signal:.2f}."
    )

    return {
        "benchmark_id": row["benchmark_id"],
        "core_id": row.get("core_id"),
        "model_under_evaluation": row.get("model_under_evaluation", "deepseek-v4-pro"),
        "score": final_score,
        "max_score": 4,
        "score_method": "local_deterministic_rubric_overlap_v1",
        "component_scores": {
            "key_point_coverage": round(key_point_score, 4),
            "reference_answer_fidelity": round(reference_fidelity_score, 4),
            "article_specific_evidence_and_values": round(evidence_score, 4),
            "expert_reasoning_and_synthesis": round(reasoning_score, 4),
            "raw_total_before_caps": round(raw_score, 4),
            "cap_applied": cap,
        },
        "diagnostics": {
            "key_point_credit_fraction": round(key_fraction, 4),
            "reference_cosine": round(ref_cos, 4),
            "reference_token_coverage": round(ref_cov, 4),
            "source_support_token_coverage": round(source_cov, 4),
            "specific_term_coverage": round(specific_cov, 4),
            "numeric_coverage": round(numeric_cov, 4),
            "answer_token_count": answer_token_count,
            "reasoning_marker_count": marker_count,
        },
        "key_point_hits": key_hits,
        "missing_key_points": missing_key_points,
        "partial_key_points": partial_key_points,
        "heuristic_flags": flags,
        "article_specific_evidence_quality": evidence_quality,
        "reasoning_quality": reasoning_quality,
        "local_scorer_confidence": scorer_confidence,
        "needs_human_or_independent_judge_review": needs_review,
        "final_rationale": rationale,
    }
