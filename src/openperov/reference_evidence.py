#!/usr/bin/env python3
"""Build deterministic, natural-language Full800 evidence packets (v8).

This is a versioned extension of ``build_compact40_stage2_evidence_packets_v1``.
It preserves the frozen Top40 membership and Stage2 ordering while changing only
the model-facing evidence compilation layer:

* Science-Style RAG topic packs are the primary source for deep CORE papers.
* The previously frozen OpenAlex abstract facet is used for non-deep papers.
* Internal CORE/RAG/card/rank identifiers remain backend-only.
* The model sees one natural paper block, without a duplicated quick map.
* No model call, benchmark label, or expected source is used.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .reference_evidence_helpers import (
    atomic_write_json,
    atomic_write_text,
    read_jsonl,
    sha256,
    token_jaccard,
    tokens,
    trim_at_boundary,
    write_jsonl,
)


SPACE_RE = re.compile(r"\s+")
# Numbered-list markers may follow sentence punctuation, but a decimal such as
# ``0.37. The ...`` must never be mistaken for list item ``37.``.
LIST_MARKER_RE = re.compile(r"(?:^|(?<=[.:;]))\s*(?<!\d\.)\d{1,3}\.\s+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+~≈<>≤≥]?\d+(?:\.\d+)?(?:\s*[×x]\s*10(?:\^)?[-+]?\d+)?")
INTERNAL_ID_RE = re.compile(
    r"\b(?:CORE_\d+|DOC_\d+|RAG(?:CORE|PARENT|EXPERT)?_[A-Z0-9_-]+|"
    r"EVID_[A-Z0-9_-]+|(?:TC|MM|RB|FC|NG|LT|CL|SC)\d{3,}|"
    r"(?:C|M)\d{3}(?:-(?:C|M)?\d{3})?)\b",
    flags=re.IGNORECASE,
)
PAREN_INTERNAL_RE = re.compile(
    r"\s*\((?:\s*[A-Z]{1,4}\d{3,}(?:-[A-Z]?\d{3,})?\s*[,;/]?)+\)",
    flags=re.IGNORECASE,
)

GENERIC_UNITS = (
    "device and performance evidence summarizes",
    "fabrication evidence summarizes",
    "scientific facet:",
    "paper title:",
    "this perovskite literature record is titled",
    "candidate-stage",
    "training value",
    "asset value",
)

MALFORMED_UNIT_RE = re.compile(
    r"(?:\bfor\s*,\s*and\b|\bof\s*,\s*and\b|\bfrom\s*,\s*to\b|"
    r"\bfor\s+and\s+are\b|\bthe\s*,\s*and\s+the\b)",
    flags=re.IGNORECASE,
)
VERB_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has|have|had|shows?|showed|"
    r"reports?|reported|finds?|found|observes?|observed|indicates?|indicated|"
    r"supports?|supported|suggests?|suggested|demonstrates?|demonstrated|"
    r"reveals?|revealed|confirms?|confirmed|increases?|increased|decreases?|"
    r"decreased|improves?|improved|reduces?|reduced|suppresses?|suppressed|"
    r"retains?|retained|yields?|yielded|produces?|produced|forms?|formed|"
    r"reaches?|reached|exhibits?|exhibited|causes?|caused|limits?|limited|"
    r"provides?|provided|uses?|used|enables?|enabled|links?|linked|tracks?|"
    r"tracked|measures?|measured|compares?|compared|depends?|depended|"
    r"treats?|treated|addresses?|addressed|proposes?|proposed|integrates?|"
    r"integrated|establishes?|established|remains?|remained|serves?|served|"
    r"acts?|acted|gives?|gave|becomes?|became|undergoes?|underwent|"
    r"prevents?|prevented|blocks?|blocked|allows?|allowed)\b",
    flags=re.IGNORECASE,
)

FAMILY_TERMS = {
    "mechanism_diagnostic": {
        "mechanism", "diagnostic", "spectroscopy", "carrier", "defect",
        "migration", "interface", "recombination", "control", "kinetic",
        "field", "hysteresis", "degradation", "photoluminescence",
    },
    "stability_design_transfer": {
        "stability", "degradation", "lifetime", "humidity", "thermal",
        "illumination", "interface", "encapsulation", "migration", "control",
        "fabrication", "transfer", "failure", "design",
    },
    "stability_failure": {
        "stability", "failure", "degradation", "negative", "limitation",
        "humidity", "thermal", "illumination", "migration", "interface",
        "control", "lifetime", "hysteresis",
    },
    "design_transfer_synthesis": {
        "design", "fabrication", "process", "transfer", "composition",
        "interface", "stack", "performance", "efficiency", "stability",
        "condition", "control", "scalable", "mechanical", "temperature",
    },
}

CARD_TYPE_BOOSTS = {
    "mechanism_diagnostic": {
        "article_master_pack": 3.5,
        "mechanism_pack": 7.0,
        "stability_pack": 3.0,
        "device_performance_pack": 2.5,
        "fabrication_protocol_pack": 1.0,
        "lineage_pack": 0.5,
    },
    "stability_design_transfer": {
        "article_master_pack": 3.5,
        "stability_pack": 7.0,
        "mechanism_pack": 4.0,
        "fabrication_protocol_pack": 3.5,
        "device_performance_pack": 3.0,
        "lineage_pack": 0.5,
    },
    "stability_failure": {
        "article_master_pack": 3.5,
        "stability_pack": 7.0,
        "mechanism_pack": 5.0,
        "device_performance_pack": 2.5,
        "fabrication_protocol_pack": 2.0,
        "lineage_pack": 0.5,
    },
    "design_transfer_synthesis": {
        "article_master_pack": 3.5,
        "fabrication_protocol_pack": 7.0,
        "device_performance_pack": 5.5,
        "mechanism_pack": 4.0,
        "stability_pack": 3.5,
        "lineage_pack": 1.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--science-topic-packs", type=Path, required=True)
    parser.add_argument("--facet-corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int, default=800)
    parser.add_argument("--query-id", action="append", default=[])
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--hard-max-chars", type=int, default=70000)
    return parser.parse_args()


def clean_natural_text(text: str, title: str = "") -> str:
    value = html.unescape(str(text or ""))
    value = value.replace("\ufeff", " ").replace("\ufffd", "")
    value = value.replace("–", "-").replace("—", "-")
    value = value.replace("�", "")
    if title:
        value = re.sub(
            r"Paper title:\s*" + re.escape(title) + r"\.?\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )
    value = re.sub(
        r"Paper title:\s*[^.]{10,350}(?:\.|$)",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"Scientific facet:\s*[^.]+\.\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^\s*summary:\s*", "", value, flags=re.IGNORECASE)
    value = PAREN_INTERNAL_RE.sub("", value)
    value = INTERNAL_ID_RE.sub("", value)
    value = re.sub(r"\s*:\s*The\s+", ". The ", value)
    value = re.sub(r"\s*:\s*(?=\d{1,3}\.\s+)", ". ", value)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    value = re.sub(r"([.;:])\1+", r"\1", value)
    value = re.sub(r"\.\s*:\s*", ". ", value)
    value = re.sub(r"\.\s*;", ".", value)
    value = re.sub(r";\s*\.", ".", value)
    return SPACE_RE.sub(" ", value).strip(" ;,:.")


def split_units(text: str, title: str = "") -> list[str]:
    value = html.unescape(str(text or ""))
    value = LIST_MARKER_RE.sub("\n", value)
    value = clean_natural_text(value, title=title)
    units: list[str] = []
    for part in SENTENCE_RE.split(value):
        part = part.strip(" ;,:")
        part = re.sub(r"^\(?\d{1,2}\)\s*", "", part)
        if len(part) < 32:
            continue
        if len(part) > 720:
            subparts = re.split(
                r";\s+|\s+(?=(?:Condition|Control|Compared|Value|Significance|Boundary):)",
                part,
            )
            for subpart in subparts:
                subpart = subpart.strip(" ;,:")
                if len(subpart) >= 32:
                    units.append(subpart)
        else:
            units.append(part)
    return units


def ensure_sentence(text: str) -> str:
    value = SPACE_RE.sub(" ", text).strip(" ;,:")
    if value and value[-1] not in ".!?":
        value += "."
    return value


def informative_unit(text: str) -> bool:
    lowered = text.lower()
    if text and not (
        text[0].isupper()
        or text[0].isdigit()
        or text[0] in "(["
        or lowered.startswith(("p-i-n", "n-i-p"))
    ):
        return False
    if any(phrase in lowered for phrase in GENERIC_UNITS):
        return False
    if MALFORMED_UNIT_RE.search(text):
        return False
    words = [item.lower() for item in re.findall(r"[A-Za-z][A-Za-z0-9+./-]*", text)]
    if len(words) < 6:
        return False
    if len(text) >= 70 and not VERB_RE.search(text):
        return False
    if len(words) >= 24 and len(set(words)) / len(words) < 0.40:
        return False
    return True


def unit_score(
    text: str,
    query_tokens: set[str],
    family_terms: set[str],
    source_field: str,
) -> float:
    unit_tokens = tokens(text)
    overlap = len(query_tokens & unit_tokens)
    coverage = overlap / max(1, len(query_tokens))
    family_overlap = len(family_terms & unit_tokens)
    numeric_bonus = 1.0 if NUMBER_RE.search(text) else 0.0
    condition_bonus = 0.8 if re.search(
        r"\b(?:under|after|before|compared|versus|control|condition|temperature|"
        r"humidity|illumination|bias|cycle|lifetime|failure|negative|limitation)\b",
        text,
        flags=re.IGNORECASE,
    ) else 0.0
    field_bonus = 0.35 if source_field == "embedding_text" else 0.0
    return (
        overlap * 2.2
        + coverage * 10.0
        + family_overlap * 0.45
        + numeric_bonus
        + condition_bonus
        + field_bonus
        + min(len(text), 300) / 750.0
    )


def pack_score(
    pack: dict[str, Any],
    query_tokens: set[str],
    family: str,
) -> float:
    text_tokens = tokens(str(pack.get("embedding_text") or ""))
    overlap = len(query_tokens & text_tokens)
    coverage = overlap / max(1, len(query_tokens))
    family_overlap = len(FAMILY_TERMS.get(family, set()) & text_tokens)
    type_boost = CARD_TYPE_BOOSTS.get(family, {}).get(str(pack.get("card_type")), 1.0)
    return overlap * 2.3 + coverage * 10.0 + family_overlap * 0.35 + type_boost


def rank_profile(rank: int) -> tuple[int, int, int, int]:
    """Return max_chars, max_packs, max_units, retrieval_supplement_units."""
    if rank <= 8:
        return 1900, 3, 8, 2
    if rank <= 20:
        return 1300, 2, 6, 1
    return 800, 2, 4, 1


def select_topic_packs(
    packs: list[dict[str, Any]],
    question: str,
    family: str,
    max_packs: int,
) -> list[dict[str, Any]]:
    query_tokens = tokens(question)
    master = next(
        (item for item in packs if item.get("card_type") == "article_master_pack"),
        None,
    )
    ranked = sorted(
        [
            item
            for item in packs
            if not (
                item.get("card_type") == "lineage_pack"
                and family in {
                    "mechanism_diagnostic",
                    "stability_design_transfer",
                    "stability_failure",
                    "design_transfer_synthesis",
                }
            )
        ],
        key=lambda item: (-pack_score(item, query_tokens, family), str(item.get("card_id"))),
    )
    output: list[dict[str, Any]] = []
    if master is not None:
        output.append(master)
    for item in ranked:
        if item is master:
            continue
        if item.get("card_id") in {row.get("card_id") for row in output}:
            continue
        output.append(item)
        if len(output) >= max_packs:
            break
    if not output and ranked:
        output.append(ranked[0])
    return output[:max_packs]


def duplicate_of(text: str, prior: Iterable[str], threshold: float = 0.68) -> bool:
    normalized = SPACE_RE.sub(" ", text).strip().lower()
    for other in prior:
        other_normalized = SPACE_RE.sub(" ", other).strip().lower()
        if normalized == other_normalized:
            return True
        if min(len(normalized), len(other_normalized)) >= 90 and (
            normalized in other_normalized or other_normalized in normalized
        ):
            return True
        if token_jaccard(text, other) >= threshold:
            return True
    return False


def compile_deep_paper(
    packs: list[dict[str, Any]],
    question: str,
    family: str,
    rank: int,
) -> tuple[str, list[dict[str, Any]], list[str], list[str]]:
    max_chars, max_packs, max_units, retrieval_limit = rank_profile(rank)
    selected_packs = select_topic_packs(packs, question, family, max_packs)
    query_tokens = tokens(question)
    family_terms = FAMILY_TERMS.get(family, set())
    title = str((selected_packs[0].get("paper_identity") or {}).get("title") or "")

    candidates: list[dict[str, Any]] = []
    pack_order = {str(item.get("card_id")): index for index, item in enumerate(selected_packs)}
    for pack in selected_packs:
        for field in ("embedding_text", "retrieval_text"):
            for unit_index, unit in enumerate(split_units(str(pack.get(field) or ""), title=title)):
                unit = ensure_sentence(unit)
                if not informative_unit(unit):
                    continue
                if field == "retrieval_text" and not (
                    NUMBER_RE.search(unit)
                    or re.search(
                        r"\b(?:control|condition|under|compared|negative|failure|limitation|"
                        r"temperature|humidity|illumination|bias|cycle)\b",
                        unit,
                        flags=re.IGNORECASE,
                    )
                ):
                    continue
                if field == "retrieval_text":
                    if len(query_tokens & tokens(unit)) < 1:
                        continue
                    if len(unit) > 520 or unit.count(":") > 2:
                        continue
                    if not unit[0].isupper():
                        continue
                candidates.append(
                    {
                        "text": unit,
                        "score": unit_score(unit, query_tokens, family_terms, field),
                        "card_id": str(pack.get("card_id")),
                        "card_type": str(pack.get("card_type")),
                        "source_field": field,
                        "unit_index": unit_index,
                        "raw_source_text": str(pack.get(field) or ""),
                        "evidence_links": pack.get("evidence_links") or {},
                    }
                )

    chosen: list[dict[str, Any]] = []
    seen_texts: list[str] = []
    embedding_candidates = [item for item in candidates if item["source_field"] == "embedding_text"]
    retrieval_candidates = [item for item in candidates if item["source_field"] == "retrieval_text"]

    # Guarantee at least one coherent embedding sentence from every selected topic pack.
    for pack in selected_packs:
        options = [item for item in embedding_candidates if item["card_id"] == pack.get("card_id")]
        options.sort(key=lambda item: (-item["score"], item["unit_index"]))
        for item in options:
            if not duplicate_of(item["text"], seen_texts):
                chosen.append(item)
                seen_texts.append(item["text"])
                break

    for item in sorted(embedding_candidates, key=lambda row: (-row["score"], row["unit_index"])):
        if len([row for row in chosen if row["source_field"] == "embedding_text"]) >= max_units:
            break
        if duplicate_of(item["text"], seen_texts):
            continue
        chosen.append(item)
        seen_texts.append(item["text"])

    retrieval_added = 0
    for item in sorted(retrieval_candidates, key=lambda row: (-row["score"], row["unit_index"])):
        if retrieval_added >= retrieval_limit:
            break
        if duplicate_of(item["text"], seen_texts):
            continue
        chosen.append(item)
        seen_texts.append(item["text"])
        retrieval_added += 1

    chosen.sort(
        key=lambda row: (
            pack_order.get(row["card_id"], 999),
            0 if row["source_field"] == "embedding_text" else 1,
            row["unit_index"],
        )
    )

    emitted: list[dict[str, Any]] = []
    output_sentences: list[str] = []
    used_chars = 0
    for item in chosen:
        sentence = item["text"]
        addition = len(sentence) + (1 if output_sentences else 0)
        if used_chars + addition > max_chars:
            continue
        output_sentences.append(sentence)
        emitted.append(item)
        used_chars += addition
    if not output_sentences and chosen:
        # A single evidence sentence can exceed the rank budget. Keep it whole:
        # clipping at a decimal point can silently turn e.g. 0.52 into 0.
        complete_sentence = ensure_sentence(chosen[0]["text"])
        output_sentences = [complete_sentence]
        complete_item = dict(chosen[0])
        complete_item["text"] = complete_sentence
        emitted = [complete_item]

    body = " ".join(output_sentences)
    selected_card_ids = list(dict.fromkeys(item["card_id"] for item in emitted))
    selected_card_types = list(dict.fromkeys(item["card_type"] for item in emitted))
    return body, emitted, selected_card_ids, selected_card_types


def compile_abstract(
    facets: list[dict[str, Any]],
    question: str,
    max_chars: int = 900,
) -> tuple[str, dict[str, Any] | None]:
    abstract = next((item for item in facets if item.get("facet_type") == "openalex_abstract"), None)
    if abstract is None:
        return "", None
    title = str(abstract.get("title") or "")
    cleaned = clean_natural_text(str(abstract.get("text") or ""), title=title)
    if not cleaned:
        return "", abstract
    if len(cleaned) <= max_chars:
        return ensure_sentence(cleaned), abstract
    units = [ensure_sentence(unit) for unit in split_units(cleaned) if len(unit) >= 30]
    if not units:
        return ensure_sentence(cleaned), abstract
    query_tokens = tokens(question)
    ranked = sorted(
        enumerate(units),
        key=lambda pair: (-unit_score(pair[1], query_tokens, set(), "embedding_text"), pair[0]),
    )
    wanted_indices = {0}
    for index, _ in ranked:
        wanted_indices.add(index)
        candidate = " ".join(units[i] for i in sorted(wanted_indices))
        if len(candidate) >= max_chars * 0.78 or len(wanted_indices) >= 5:
            break
    selected = [units[i] for i in sorted(wanted_indices)]
    output: list[str] = []
    used = 0
    for unit in selected:
        if used + len(unit) + (1 if output else 0) > max_chars:
            continue
        output.append(unit)
        used += len(unit) + (1 if output else 0)
    return " ".join(output), abstract


def bibliographic_heading(identity: dict[str, Any]) -> str:
    title = str(identity.get("title") or "Untitled study").strip()
    journal = str(identity.get("journal") or "").strip()
    year = str(identity.get("year") or "").strip()
    details = ", ".join(item for item in (journal, year) if item)
    return f"### {title}" + (f" ({details})" if details else "")


def number_tokens(text: str) -> set[str]:
    return {SPACE_RE.sub("", item.group(0)).lower() for item in NUMBER_RE.finditer(text)}


def safe_identity_from_pack(pack: dict[str, Any]) -> dict[str, Any]:
    identity = dict(pack.get("paper_identity") or {})
    return {
        "title": identity.get("title") or "Untitled study",
        "journal": identity.get("journal"),
        "year": identity.get("year"),
        "doi": identity.get("doi"),
    }


def safe_identity_from_facet(facet: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": facet.get("title") or "Untitled study",
        "journal": facet.get("journal"),
        "year": facet.get("year"),
        "doi": facet.get("doi"),
    }


def build_model_text(question: str, blocks: list[str]) -> str:
    return "\n".join(
        [
            "# Scientific background for one perovskite question",
            "",
            "## Question",
            question,
            "",
            "## How to use this background",
            "Treat the following material as scientific background rather than as a document-reading task.",
            "Use only what is relevant, preserve experimental conditions and numerical values, and distinguish reported observations from interpretation.",
            "The absence of a detail in one study should not be treated as evidence that the effect does not exist.",
            "Answer from the science itself and do not discuss retrieval, ranking, corpus construction, or internal data organization.",
            "",
            "## Scientific evidence and context",
            "\n\n".join(blocks),
            "",
            "## Question to answer now",
            question,
            "",
            "Answer directly as a perovskite expert.",
            "",
        ]
    )


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return int(ordered[index])


def main() -> int:
    args = parse_args()
    started = time.time()
    for path in (args.questions, args.rankings, args.science_topic_packs, args.facet_corpus):
        if not path.is_file():
            raise FileNotFoundError(path)

    question_rows = list(read_jsonl(args.questions))
    ranking_rows = list(read_jsonl(args.rankings))
    if len(question_rows) != args.expected_queries or len(ranking_rows) != args.expected_queries:
        raise RuntimeError(
            f"Expected {args.expected_queries} questions/rankings, got "
            f"{len(question_rows)}/{len(ranking_rows)}"
        )
    questions = {str(row["query_id"]): row for row in question_rows}
    rankings = {str(row["query_id"]): row for row in ranking_rows}
    if set(questions) != set(rankings):
        raise RuntimeError("Question and ranking query IDs do not align")

    ordered_query_ids = [str(row["query_id"]) for row in question_rows]
    if args.query_id:
        requested = [str(query_id) for query_id in args.query_id]
        missing_requested = sorted(set(requested) - set(ordered_query_ids))
        if missing_requested:
            raise RuntimeError(f"Missing requested query IDs: {missing_requested}")
        ordered_query_ids = list(dict.fromkeys(requested))
    elif args.query_limit is not None:
        ordered_query_ids = ordered_query_ids[: args.query_limit]

    required_core_ids: set[str] = set()
    for query_id in ordered_query_ids:
        ranking = list(rankings[query_id].get("ranking") or [])[:40]
        if len(ranking) != 40:
            raise RuntimeError(f"{query_id}: expected 40 ranked papers, got {len(ranking)}")
        ids = [str(item["core_id"]) for item in ranking]
        if len(set(ids)) != 40:
            raise RuntimeError(f"{query_id}: duplicate paper in Top40")
        required_core_ids.update(ids)

    science_by_core: dict[str, list[dict[str, Any]]] = defaultdict(list)
    science_rows = 0
    for row in read_jsonl(args.science_topic_packs):
        science_rows += 1
        core_id = str(row["core_id"])
        if core_id in required_core_ids:
            science_by_core[core_id].append(row)

    non_deep_ids = required_core_ids - set(science_by_core)
    facets_by_core: dict[str, list[dict[str, Any]]] = defaultdict(list)
    facet_rows_scanned = 0
    for row in read_jsonl(args.facet_corpus):
        facet_rows_scanned += 1
        core_id = str(row.get("core_id"))
        if core_id in non_deep_ids:
            facets_by_core[core_id].append(row)
    missing_facets = sorted(non_deep_ids - set(facets_by_core))
    if missing_facets:
        raise RuntimeError(f"Missing frozen facet records for {len(missing_facets)} papers")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = args.output_dir / "stage2_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    packets: list[dict[str, Any]] = []
    model_inputs: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    input_manifest: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    selected_card_types: Counter[str] = Counter()
    char_counts: list[int] = []
    token_estimates: list[int] = []
    internal_leaks: list[dict[str, Any]] = []
    numeric_mismatches: list[dict[str, Any]] = []
    duplicate_sentences = 0
    identity_only_papers: list[dict[str, Any]] = []
    family_review_samples: dict[str, tuple[str, str]] = {}

    for query_index, query_id in enumerate(ordered_query_ids, start=1):
        question_row = questions[query_id]
        rank_row = rankings[query_id]
        question = str(question_row["question"]).strip()
        family = str(question_row.get("ability_family") or rank_row.get("ability_family") or "")
        ranking = list(rank_row["ranking"])[:40]
        cards: list[dict[str, Any]] = []
        blocks: list[str] = []

        for item in ranking:
            rank = int(item["rank"])
            core_id = str(item["core_id"])
            if core_id in science_by_core:
                packs = science_by_core[core_id]
                identity = safe_identity_from_pack(packs[0])
                body, emitted, card_ids, card_types = compile_deep_paper(
                    packs, question, family, rank
                )
                source_level = "science_style_core"
                source_counts[source_level] += 1
                selected_card_types.update(card_types)
                for sentence_index, source in enumerate(emitted, start=1):
                    raw_numbers = number_tokens(source["raw_source_text"])
                    emitted_numbers = number_tokens(source["text"])
                    if not emitted_numbers.issubset(raw_numbers):
                        numeric_mismatches.append(
                            {
                                "query_id": query_id,
                                "core_id": core_id,
                                "card_id": source["card_id"],
                                "sentence": source["text"],
                                "unexpected_numbers": sorted(emitted_numbers - raw_numbers),
                            }
                        )
                    registry.append(
                        {
                            "schema": "opensolar_natural40_source_registry_v8",
                            "query_id": query_id,
                            "rank": rank,
                            "core_id": core_id,
                            "title": identity["title"],
                            "source_level": source_level,
                            "sentence_index": sentence_index,
                            "source_id": source["card_id"],
                            "source_type": source["card_type"],
                            "source_field": source["source_field"],
                            "source_unit_index": source["unit_index"],
                            "model_facing_sentence": source["text"],
                            "evidence_links": source["evidence_links"],
                        }
                    )
                selected_source_ids = card_ids
                selected_source_types = card_types
            else:
                facets = facets_by_core[core_id]
                identity = safe_identity_from_facet(facets[0])
                body, abstract = compile_abstract(facets, question)
                if abstract is not None and body:
                    source_level = "frozen_abstract"
                    source_counts[source_level] += 1
                    registry.append(
                        {
                            "schema": "opensolar_natural40_source_registry_v8",
                            "query_id": query_id,
                            "rank": rank,
                            "core_id": core_id,
                            "title": identity["title"],
                            "source_level": source_level,
                            "sentence_index": 1,
                            "source_id": abstract.get("facet_id"),
                            "source_type": abstract.get("facet_type"),
                            "source_field": "text",
                            "source_unit_index": None,
                            "model_facing_sentence": body,
                            "evidence_links": {},
                        }
                    )
                    selected_source_ids = [str(abstract.get("facet_id"))]
                    selected_source_types = [str(abstract.get("facet_type"))]
                else:
                    source_level = "bibliographic_only"
                    source_counts[source_level] += 1
                    selected_source_ids = [str(facets[0].get("facet_id"))]
                    selected_source_types = [str(facets[0].get("facet_type"))]
                    identity_only_papers.append(
                        {"query_id": query_id, "rank": rank, "core_id": core_id, "title": identity["title"]}
                    )

            heading = bibliographic_heading(identity)
            blocks.append(heading + ("\n" + body if body else ""))
            body_units = split_units(body)
            normalized_units: set[str] = set()
            for unit in body_units:
                normalized = SPACE_RE.sub(" ", unit).strip().lower()
                if normalized in normalized_units:
                    duplicate_sentences += 1
                normalized_units.add(normalized)
            cards.append(
                {
                    "rank": rank,
                    "core_id": core_id,
                    "title": identity["title"],
                    "journal": identity.get("journal"),
                    "year": identity.get("year"),
                    "doi": identity.get("doi"),
                    "source_level": source_level,
                    "selected_source_ids": selected_source_ids,
                    "selected_source_types": selected_source_types,
                    "model_facing_text": body,
                }
            )

        model_text = build_model_text(question, blocks)
        if len(model_text) > args.hard_max_chars:
            raise RuntimeError(
                f"{query_id}: model input {len(model_text)} chars exceeds hard max {args.hard_max_chars}"
            )
        leaks = sorted(set(match.group(0) for match in INTERNAL_ID_RE.finditer(model_text)))
        if leaks:
            internal_leaks.append({"query_id": query_id, "leaks": leaks})

        input_path = input_dir / f"{query_id}.txt"
        atomic_write_text(input_path, model_text)
        input_hash = sha256(input_path)
        input_manifest.append(
            {
                "query_id": query_id,
                "relative_path": str(input_path.relative_to(args.output_dir)),
                "bytes": input_path.stat().st_size,
                "sha256": input_hash,
            }
        )
        char_counts.append(len(model_text))
        token_estimate = math.ceil(len(model_text) / 4)
        token_estimates.append(token_estimate)
        packets.append(
            {
                "schema": "opensolar_natural40_stage2_evidence_packet_v8",
                "query_id": query_id,
                "benchmark_id": question_row.get("benchmark_id"),
                "ability_family": family,
                "question": question,
                "top40_source": "frozen public_top40_stage2_rankings.jsonl",
                "top40_selection": rank_row.get("selection"),
                "top40_ordering": rank_row.get("ordering"),
                "private_labels_read": False,
                "source_core_injected": False,
                "human_expert_annotation": False,
                "cards": cards,
                "model_input_chars": len(model_text),
                "estimated_model_input_tokens": token_estimate,
                "model_input_sha256": input_hash,
            }
        )
        model_inputs.append(
            {
                "schema": "opensolar_natural40_model_input_v8",
                "query_id": query_id,
                "benchmark_id": question_row.get("benchmark_id"),
                "ability_family": family,
                "question": question,
                "input_text": model_text,
                "input_sha256": input_hash,
            }
        )
        family_review_samples.setdefault(family, (query_id, model_text))
        if query_index % 25 == 0 or query_index == len(ordered_query_ids):
            print(
                f"\033[36mV8 {query_index}/{len(ordered_query_ids)} "
                f"query={query_id} chars={len(model_text)} est_tokens={token_estimate}\033[0m",
                flush=True,
            )

    packet_path = args.output_dir / "evidence_packets.jsonl"
    model_inputs_path = args.output_dir / "model_inputs.jsonl"
    registry_path = args.output_dir / "source_registry.jsonl"
    input_manifest_path = args.output_dir / "stage2_inputs_manifest.jsonl"
    write_jsonl(packet_path, packets)
    write_jsonl(model_inputs_path, model_inputs)
    write_jsonl(registry_path, registry)
    write_jsonl(input_manifest_path, input_manifest)

    review_text = ["# Natural Pack v8 deterministic review sample", ""]
    for family, (query_id, model_text) in family_review_samples.items():
        review_text.extend([f"## {family}: {query_id}", "", model_text, ""])
    review_path = args.output_dir / "review_sample_by_family.md"
    atomic_write_text(review_path, "\n".join(review_text))

    qc_report = {
        "schema": "opensolar_natural40_full800_qc_v8",
        "status": "PASS" if not internal_leaks and not numeric_mismatches else "FAIL",
        "queries": len(packets),
        "paper_slots": sum(len(row["cards"]) for row in packets),
        "unique_required_papers": len(required_core_ids),
        "unique_science_style_core_papers": len(required_core_ids & set(science_by_core)),
        "unique_non_deep_papers": len(non_deep_ids),
        "source_slot_counts": dict(source_counts),
        "selected_topic_type_counts": dict(selected_card_types),
        "science_topic_rows_scanned": science_rows,
        "facet_rows_scanned": facet_rows_scanned,
        "missing_facet_papers": missing_facets,
        "identity_only_slot_count": len(identity_only_papers),
        "identity_only_examples": identity_only_papers[:20],
        "internal_identifier_leaks": internal_leaks,
        "numeric_preservation_mismatches": numeric_mismatches,
        "within_paper_exact_duplicate_sentences": duplicate_sentences,
        "model_input_chars": {
            "min": min(char_counts),
            "median": statistics.median(char_counts),
            "p90": percentile(char_counts, 0.90),
            "p95": percentile(char_counts, 0.95),
            "max": max(char_counts),
        },
        "estimated_model_input_tokens": {
            "min": min(token_estimates),
            "median": statistics.median(token_estimates),
            "p90": percentile(token_estimates, 0.90),
            "p95": percentile(token_estimates, 0.95),
            "max": max(token_estimates),
        },
        "private_labels_read": False,
        "source_core_injected": False,
        "human_expert_annotation": False,
        "model_calls_used": False,
    }
    qc_path = args.output_dir / "qc_report.json"
    atomic_write_json(qc_path, qc_report)

    output_files = [
        packet_path,
        model_inputs_path,
        registry_path,
        input_manifest_path,
        review_path,
        qc_path,
    ]
    manifest = {
        "schema": "opensolar_natural40_full800_manifest_v8",
        "status": qc_report["status"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(time.time() - started, 3),
        "policy": {
            "frozen_top40_membership": True,
            "frozen_stage2_ordering": True,
            "science_style_all_packs_considered": True,
            "embedding_text_is_natural_backbone": True,
            "retrieval_text_is_quantitative_supplement": True,
            "frozen_abstract_reused_for_non_deep_papers": True,
            "new_abstract_downloads": False,
            "model_facing_internal_ids": False,
            "model_facing_rank_or_paper_numbers": False,
            "model_calls_used": False,
        },
        "counts": qc_report,
        "inputs": {
            "questions": {"path": str(args.questions), "sha256": sha256(args.questions)},
            "rankings": {"path": str(args.rankings), "sha256": sha256(args.rankings)},
            "science_topic_packs": {
                "path": str(args.science_topic_packs),
                "sha256": sha256(args.science_topic_packs),
            },
            "facet_corpus": {"path": str(args.facet_corpus), "sha256": sha256(args.facet_corpus)},
        },
        "outputs": {
            str(path.relative_to(args.output_dir)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in output_files
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)

    sums = []
    for path in [*output_files, manifest_path]:
        sums.append(f"{sha256(path)}  {path.relative_to(args.output_dir)}")
    atomic_write_text(args.output_dir / "SHA256SUMS.txt", "\n".join(sums) + "\n")

    if qc_report["status"] != "PASS":
        raise RuntimeError(
            f"QC failed: identifier_leaks={len(internal_leaks)} "
            f"numeric_mismatches={len(numeric_mismatches)}"
        )
    print(
        f"\033[32mCOMPLETE v8 queries={len(packets)} papers={qc_report['paper_slots']} "
        f"median_tokens={qc_report['estimated_model_input_tokens']['median']} "
        f"output={args.output_dir}\033[0m",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
