#!/usr/bin/env python3
"""Compile all Top40 papers into compact, model-readable evidence packets.

The script deliberately reuses the frozen public Top40 and the existing
Science-style RAG database.  It does not train or call another model.

Model-facing policy:
* every Top40 paper appears once in the quick map;
* papers backed by the local Science-RAG corpus receive query-matched evidence;
* higher-ranked papers receive more evidence, but lower-ranked CORE papers are
  not discarded;
* internal CORE/RAG identifiers stay in the source registry, not in prose;
* no benchmark-private expected source is read or injected.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import sqlite3
import statistics
import time
import zlib
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+./-]{1,}|[0-9]+(?:\.[0-9]+)?")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])|\n+")
SPACE_RE = re.compile(r"\s+")
CORE_RE = re.compile(r"CORE_\d+", re.IGNORECASE)
EVID_RE = re.compile(r"\b(?:EVID|RAG(?:CORE|PARENT|EXPERT)?)[A-Z0-9_-]+\b")

STOP = {
    "the", "and", "for", "with", "from", "into", "that", "this", "are",
    "was", "were", "has", "have", "using", "under", "between", "through",
    "about", "which", "what", "when", "then", "than", "their", "while",
    "how", "why", "can", "could", "would", "should", "paper", "study",
    "explain", "build", "conclude", "one", "also", "its", "our", "but",
    "perovskite", "perovskites", "solar", "cell", "cells", "film", "films",
    "device", "devices", "paper", "study", "result", "results",
}

FAMILY_TERMS = {
    "mechanism_diagnostic": {
        "mechanism", "characterization", "diagnostic", "spectroscopy", "pl",
        "giwaxs", "xrd", "xps", "control", "carrier", "defect", "ion",
        "migration", "interface", "degradation", "hysteresis",
    },
    "stability_design_transfer": {
        "stability", "degradation", "protocol", "fabrication", "interface",
        "encapsulation", "humidity", "thermal", "illumination", "lifetime",
        "control", "mechanism", "design", "transfer",
    },
    "stability_failure": {
        "stability", "failure", "degradation", "limitation", "negative",
        "control", "humidity", "thermal", "illumination", "lifetime",
        "hysteresis", "migration", "interface",
    },
    "design_transfer_synthesis": {
        "design", "fabrication", "protocol", "transfer", "composition",
        "interface", "device", "control", "mechanism", "performance",
        "efficiency", "stability", "condition",
    },
}

# Direct scientific evidence groups retained for the compact packet.  Lineage,
# figure-parsing and integrated-story assets are useful elsewhere, but their
# metadata is too noisy for a fast answer-model prompt.
DIRECT_EVIDENCE_GROUPS = {
    "metric_performance",
    "mechanism_characterization",
    "protocol_fabrication",
    "device_stack_control",
    "table_reasoning",
    "fallback_science",
}

DIRECT_ATOMIC_MARKERS = (
    "stage1_metric_mention",
    "stage1_result_narrative",
    "stage1_text_claim",
    "stage1_control_baseline",
    "stage1_negative_or_null_result",
    "stage1_limitation",
    "stage1_statistics_uncertainty",
    "stage1_fabrication_claim",
    "stage1_method_unit",
    "stage1_device_structure_claim",
    "table_evidence_context_unit",
    "table_semantic_analysis_record",
)

NON_SCIENCE_PHRASES = (
    "candidate-stage",
    "training contrast",
    "downstream_reuse_patterns",
    "unresolved_questions_after_target",
    "paper_story:",
    "lineage_position:",
    "target_paper_advance_in_lineage",
    "available abstract signal",
    "review/background reuse",
    "raw fka overview",
    "detected raw panel-type",
    "no curve digitization",
    "extraction from main/si text",
    "asset_id:",
    "review-table",
    "table 1 row",
)

NOISE_PATTERNS = (
    r"Stage4 candidate fact\.\s*",
    r"Stage4 fact\.\s*",
    r"Candidate scientific content:\s*",
    r"Scientific content:\s*",
    r"Candidate-stage pack input[^.]*\.\s*",
    r"candidate-stage evidence[^.]*\.\s*",
    r"requires later calibration[^.]*\.\s*",
    r"Canonical source boundary:\s*[^.]+\.\s*",
    r"Priority status:\s*[^.]+\.\s*",
    r"Direct evidence IDs?:\s*[^.]+\.\s*",
    r"Notes?:\s*sparse[^.]*\.\s*",
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(text or "")
        if token.lower() not in STOP
    }


def token_jaccard(left: str, right: str) -> float:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / max(1, len(union))


def trim_at_boundary(text: str, max_chars: int) -> str:
    text = text.strip(" ;,:.-")
    if len(text) <= max_chars:
        return text
    clipped = text[: max_chars + 1]
    cut = max(clipped.rfind(". "), clipped.rfind("; "), clipped.rfind(", "))
    if cut >= int(max_chars * 0.6):
        clipped = clipped[: cut + 1]
    else:
        cut = clipped.rfind(" ")
        clipped = clipped[:cut] if cut > 0 else clipped[:max_chars]
    return clipped.rstrip(" ;,:.-") + "."


def clean_science_text(text: str, paper_title: str = "") -> str:
    value = html.unescape(str(text or ""))
    value = value.replace("\ufeff", " ").replace("\ufffd", "")
    value = value.replace("–", "-").replace("—", "-")
    if paper_title:
        value = re.sub(
            r"Paper title:\s*" + re.escape(paper_title) + r"\.?\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )
    value = re.sub(
        r"Paper title:\s*[^.]{10,300}(?:\.|$)",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"RAG parent pack:\s*core=[^;]+;\s*family=[^.]+\.\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"Evidence type:\s*[^.]+\.\s*Content:\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"Evidence type:\s*stage1[^.]*\.?\s*",
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
    for pattern in NOISE_PATTERNS:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE)
    value = CORE_RE.sub("the focal paper", value)
    value = EVID_RE.sub("", value)
    value = re.sub(r"^\s*\d{1,3}\.\s+", "", value)
    value = re.sub(
        r"\b(?:block_id|result_block_id|control_id|unit_id|negative_id):\s*\S+\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    replacements = {
        "reported_raw_metric:": "Reported result:",
        "negative_result:": "Negative result:",
        "negative_or_null_result:": "Negative result:",
        "describes_fabrication:": "Procedure:",
        "why_important:": "Significance:",
        "why_it_matters:": "Significance:",
        "learning_value:": "Significance:",
        "training_value:": "Significance:",
        "asset_value:": "Significance:",
        "core_finding:": "Finding:",
        "core_message:": "Finding:",
        "question_answered:": "Question:",
        "interpretation_boundary:": "Boundary:",
        "control_or_baseline:": "Control/baseline:",
        "controlled_variable:": "Controlled variable:",
        "result_summary:": "",
        "summary:": "",
        "narrative:": "",
        "block_title:": "",
        "section_or_figure:": "Source:",
        "section:": "Source:",
        "page_range:": "Pages:",
        "mechanistic_role:": "Role:",
        "control_type:": "Control:",
        "control_description:": "",
        "baseline:": "Baseline:",
        "outcome:": "Implication:",
        "measurement_condition:": "Measurement condition:",
        "source_location:": "Source location:",
        "raw value:": "Value:",
    }
    for old, new in replacements.items():
        value = re.sub(re.escape(old), new, value, flags=re.IGNORECASE)
    value = re.sub(
        r"\b(?:mechanism|characterization|overview|stability|fabrication|"
        r"specialized_context|device_performance|openalex_abstract):\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\b(?:characterization metric|mechanism claim|device metric|"
        r"fabrication protocol|figure grounding evidence):\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\bType:\s*[A-Za-z0-9_ /-]+\.\s*", "", value)
    value = re.sub(r"\b(?:Subject|Content):\s*", "", value)
    value = re.sub(r"\btarget article package\b", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"\s+document:\s*\S+.*?(?=(?:\s+(?:Claim|Finding|Result|Control|Boundary|Significance):)|$)",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    return SPACE_RE.sub(" ", value).strip(" ;,:.-")


def sentence_units(text: str) -> list[str]:
    units: list[str] = []
    for part in SENTENCE_SPLIT_RE.split(text):
        part = part.strip(" ;,:-")
        if len(part) < 35:
            continue
        if len(part) > 650:
            subparts = re.split(r";\s+|\s+(?=Condition:|Value:|Significance:)", part)
            units.extend(item.strip(" ;,:-") for item in subparts if len(item.strip()) >= 35)
        else:
            units.append(part)
    return units


def is_science_unit(text: str) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in NON_SCIENCE_PHRASES):
        return False
    if lowered.count("table ") >= 2:
        return False
    words = [token.lower() for token in TOKEN_RE.findall(text)]
    if len(words) >= 24 and len(set(words)) / len(words) < 0.43:
        return False
    return True


def best_excerpt(
    text: str,
    query_tokens: set[str],
    max_chars: int,
    max_sentences: int,
    paper_title: str = "",
) -> str:
    cleaned = clean_science_text(text, paper_title=paper_title)
    units = [unit for unit in sentence_units(cleaned) if is_science_unit(unit)]
    if not units:
        if len(cleaned) >= 20 and is_science_unit(cleaned):
            return trim_at_boundary(cleaned, max_chars)
        return ""
    ranked: list[tuple[float, int, str]] = []
    for index, unit in enumerate(units):
        unit_tokens = tokens(unit)
        overlap = len(query_tokens & unit_tokens)
        coverage = overlap / max(1, len(query_tokens))
        numeric_bonus = 0.8 if re.search(r"\d", unit) else 0.0
        condition_bonus = 0.6 if re.search(
            r"\b(condition|control|compared|under|after|before|versus|vs\.?|while)\b",
            unit,
            flags=re.IGNORECASE,
        ) else 0.0
        length_bonus = min(len(unit), 260) / 520.0
        metadata_penalty = 1.2 * len(
            re.findall(
                r"\b(?:subject|type|content|condition|value|metric mentions|source location):",
                unit,
                flags=re.IGNORECASE,
            )
        )
        ranked.append(
            (
                overlap * 2.0
                + coverage * 8.0
                + numeric_bonus
                + condition_bonus
                + length_bonus
                - metadata_penalty,
                index,
                unit,
            )
        )
    chosen = sorted(ranked, key=lambda item: (-item[0], item[1]))[:max_sentences]
    chosen.sort(key=lambda item: item[1])
    output = " ".join(item[2] for item in chosen)
    return trim_at_boundary(output, max_chars)


def decode_payload(blob: bytes) -> dict[str, Any]:
    return json.loads(zlib.decompress(blob).decode("utf-8-sig"))


class EvidenceStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database = sqlite3.connect(str(database_path), timeout=120)
        self.database.row_factory = sqlite3.Row
        self.database.execute("PRAGMA query_only=ON")
        self.cache: dict[str, list[dict[str, Any]]] = {}

    def close(self) -> None:
        self.database.close()

    def records_by_core(self, core_id: str) -> list[dict[str, Any]]:
        if core_id in self.cache:
            return self.cache[core_id]
        rows = self.database.execute(
            "SELECT rag_doc_id,core_id,rag_doc_type,evidence_group,"
            "retrieval_value_score,payload_zlib FROM evidence_docs WHERE core_id=?",
            (core_id,),
        ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            payload = decode_payload(row["payload_zlib"])
            body = str(
                payload.get("embedding_text_clean")
                or payload.get("retrieval_text")
                or payload.get("retrieval_text_full")
                or ""
            ).strip()
            records.append(
                {
                    "rag_doc_id": str(row["rag_doc_id"]),
                    "core_id": str(row["core_id"]),
                    "rag_doc_type": str(row["rag_doc_type"]),
                    "evidence_group": str(row["evidence_group"] or ""),
                    "retrieval_value_score": float(row["retrieval_value_score"] or 0.0),
                    "retrieval_title": str(payload.get("retrieval_title") or ""),
                    "text": body,
                    "source_metadata": payload.get("source_metadata") or {},
                    "routing_metadata": payload.get("routing_metadata") or {},
                    "provenance": payload.get("provenance") or {},
                }
            )
        self.cache[core_id] = records
        return records


def record_kind(record: dict[str, Any]) -> str:
    doc_type = str(record["rag_doc_type"]).lower()
    if "atomic" in doc_type:
        return "atomic"
    if "parent" in doc_type:
        return "parent"
    return "other"


def is_direct_science_record(record: dict[str, Any]) -> bool:
    if record_kind(record) != "atomic":
        return False
    group = str(record.get("evidence_group") or "").lower()
    doc_id = str(record.get("rag_doc_id") or "").lower()
    if group not in DIRECT_EVIDENCE_GROUPS:
        return False
    if not any(marker in doc_id for marker in DIRECT_ATOMIC_MARKERS):
        return False
    return is_science_unit(str(record.get("text") or ""))


def rank_records(
    records: list[dict[str, Any]],
    question: str,
    ability_family: str,
    paper_title: str,
) -> list[tuple[float, dict[str, Any], str]]:
    query_tokens = tokens(question)
    family_terms = FAMILY_TERMS.get(ability_family, set())
    ranked: list[tuple[float, dict[str, Any], str]] = []
    for record in records:
        if not is_direct_science_record(record):
            continue
        metadata = " ".join(
            [
                str(record["retrieval_title"]),
                str(record["evidence_group"]),
                json.dumps(record["routing_metadata"], ensure_ascii=False),
            ]
        )
        evidence_tokens = tokens(metadata + " " + record["text"])
        overlap = len(query_tokens & evidence_tokens)
        if overlap < 2:
            continue
        coverage = overlap / max(1, len(query_tokens))
        family_overlap = len(family_terms & evidence_tokens)
        value = max(0.0, float(record["retrieval_value_score"]))
        score = (
            overlap * 2.2
            + coverage * 10.0
            + family_overlap * 0.55
            + math.log1p(value) / 2.5
            + 2.0
        )
        excerpt = best_excerpt(
            record["text"],
            query_tokens=query_tokens,
            max_chars=330,
            max_sentences=1,
            paper_title=paper_title,
        )
        if excerpt and len(excerpt) >= 55 and len(tokens(excerpt)) >= 7:
            ranked.append((score, record, excerpt))
    return sorted(ranked, key=lambda item: (-item[0], item[1]["rag_doc_id"]))


def select_unique_records(
    ranked: list[tuple[float, dict[str, Any], str]],
    kind: str,
    count: int,
    existing_texts: list[str] | None = None,
) -> list[tuple[float, dict[str, Any], str]]:
    selected: list[tuple[float, dict[str, Any], str]] = []
    seen = list(existing_texts or [])
    for item in ranked:
        if record_kind(item[1]) != kind:
            continue
        if any(token_jaccard(item[2], prior) >= 0.72 for prior in seen):
            continue
        selected.append(item)
        seen.append(item[2])
        if len(selected) >= count:
            break
    return selected


def evidence_budget(rank: int) -> tuple[int, int, int]:
    if rank <= 8:
        return 3, 3, 720
    if rank <= 20:
        return 2, 2, 520
    return 1, 1, 320


def choose_query_rows(
    rankings_path: Path,
    query_ids: list[str],
    sample_per_family: int,
    query_limit: int | None,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    wanted = set(query_ids)
    selected: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    family_counts: dict[str, int] = {}
    for row in read_jsonl(rankings_path):
        query_id = str(row["query_id"])
        use = False
        if wanted:
            use = query_id in wanted
        elif sample_per_family > 0:
            family = str(row.get("ability_family") or "unknown")
            use = family_counts.get(family, 0) < sample_per_family
            if use:
                family_counts[family] = family_counts.get(family, 0) + 1
        else:
            use = query_limit is None or len(selected) < query_limit
        if not use:
            continue
        selected.append(query_id)
        rows[query_id] = row
        if wanted and wanted.issubset(rows):
            break
        if not wanted and query_limit is not None and len(selected) >= query_limit:
            break
    if wanted - set(rows):
        raise RuntimeError(f"Missing requested query IDs: {sorted(wanted - set(rows))}")
    if not selected:
        raise RuntimeError("No query rows selected")
    return selected, rows


def load_selected_pool_rows(
    pool_path: Path, selected_ids: list[str]
) -> dict[str, dict[str, Any]]:
    wanted = set(selected_ids)
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(pool_path):
        query_id = str(row["query_id"])
        if query_id in wanted:
            rows[query_id] = row
            if len(rows) == len(wanted):
                break
    missing = wanted - set(rows)
    if missing:
        raise RuntimeError(f"Missing pool rows: {sorted(missing)}")
    return rows


def build_one_packet(
    rank_row: dict[str, Any],
    pool_row: dict[str, Any],
    store: EvidenceStore,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    query_id = str(rank_row["query_id"])
    question = str(pool_row["question"])
    ability_family = str(rank_row.get("ability_family") or pool_row.get("ability_family") or "")
    ranking = list(rank_row.get("ranking") or [])[:40]
    if len(ranking) != 40:
        raise RuntimeError(f"{query_id}: expected exactly 40 ranked papers, got {len(ranking)}")
    pool_by_core = {str(item["core_id"]): item for item in pool_row["candidates"]}
    query_tokens = tokens(question)
    cards: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    quick_lines: list[str] = []
    detail_blocks: list[str] = []

    for position, ranked_item in enumerate(ranking, start=1):
        core_id = str(ranked_item["core_id"])
        candidate = pool_by_core.get(core_id)
        if candidate is None:
            raise RuntimeError(f"{query_id}: ranked core absent from public pool: {core_id}")
        paper_label = f"P{position:02d}"
        title = str(candidate.get("title") or "Untitled paper").strip()
        journal = str(candidate.get("journal") or "").strip()
        year = candidate.get("year")
        records = store.records_by_core(core_id)
        science_rag = bool(records)
        quick = best_excerpt(
            str(candidate.get("scientific_evidence") or ""),
            query_tokens=query_tokens,
            max_chars=210,
            max_sentences=1,
            paper_title=title,
        )
        if not quick:
            quick = "No reliable query-specific statement was extracted from the available public text."
        level = "SCIENCE-RAG CORE" if science_rag else "ABSTRACT/FACET ONLY"
        citation = ", ".join(part for part in (journal, str(year or "")) if part)
        suffix = f" ({citation})" if citation else ""
        quick_lines.append(f"- {title}{suffix}. {quick}")

        evidence_items: list[dict[str, Any]] = []
        context_items: list[dict[str, Any]] = []
        if science_rag:
            ranked_records = rank_records(records, question, ability_family, title)
            summary_sentences, atomic_count, summary_chars = evidence_budget(position)
            science_summary = ""
            if summary_sentences:
                science_summary = best_excerpt(
                    str(candidate.get("scientific_evidence") or ""),
                    query_tokens=query_tokens,
                    max_chars=summary_chars,
                    max_sentences=summary_sentences,
                    paper_title=title,
                )
            if science_summary:
                summary_label = f"{paper_label}-S"
                context_items.append(
                    {
                        "label": summary_label,
                        "kind": "science_facet_summary",
                        "text": science_summary,
                        "rag_doc_id": None,
                    }
                )
                registry.append(
                    {
                        "query_id": query_id,
                        "paper_label": paper_label,
                        "evidence_label": summary_label,
                        "rank": position,
                        "core_id": core_id,
                        "title": title,
                        "rag_doc_id": None,
                        "rag_doc_type": "public_scientific_facet",
                        "evidence_group": "candidate_scientific_evidence",
                        "presented_facet_types": candidate.get("presented_facet_types") or [],
                        "source_metadata": {},
                        "provenance": {
                            "source": "public_multilane_pool.jsonl",
                            "candidate_code": candidate.get("candidate_code"),
                        },
                        "model_facing_text": science_summary,
                    }
                )
            chosen_atomic = select_unique_records(
                ranked_records, kind="atomic", count=atomic_count
            )
            for evidence_index, (_, record, excerpt) in enumerate(chosen_atomic, start=1):
                label = f"{paper_label}-E{evidence_index}"
                item = {
                    "label": label,
                    "kind": "direct_evidence",
                    "text": excerpt,
                    "rag_doc_id": record["rag_doc_id"],
                }
                evidence_items.append(item)
                registry.append(
                    {
                        "query_id": query_id,
                        "paper_label": paper_label,
                        "evidence_label": label,
                        "rank": position,
                        "core_id": core_id,
                        "title": title,
                        "rag_doc_id": record["rag_doc_id"],
                        "rag_doc_type": record["rag_doc_type"],
                        "evidence_group": record["evidence_group"],
                        "source_metadata": record["source_metadata"],
                        "provenance": record["provenance"],
                        "model_facing_text": excerpt,
                    }
                )
        card = {
            "paper_label": paper_label,
            "rank": position,
            "core_id": core_id,
            "title": title,
            "journal": journal,
            "year": year,
            "evidence_level": "science_rag_core" if science_rag else "abstract_or_facet_only",
            "quick_map_text": quick,
            "evidence": evidence_items,
            "context": context_items,
        }
        cards.append(card)
        detail = [f"### {title}"]
        if evidence_items:
            detail.extend(
                f"- Direct experimental evidence: {item['text']}"
                for item in evidence_items
            )
        if context_items:
            detail.extend(
                f"- Scientific context: {item['text']}"
                for item in context_items
            )
        if evidence_items or context_items:
            detail_blocks.append("\n".join(detail))

    model_text = "\n".join(
        [
            "# Scientific background for one perovskite question",
            "",
            "## Question",
            question,
            "",
            "## How to use this background",
            "Treat the material below as background knowledge, not as a document-reading task.",
            "Use whatever is scientifically helpful and ignore material that is not relevant to the question.",
            "Preserve useful experimental conditions and numerical values, while separating reported observations from inference.",
            "Answer from the science itself rather than discussing how this background was collected or organized.",
            "",
            "## Research landscape",
            *quick_lines,
            "",
            "## Detailed scientific background",
            "\n\n".join(detail_blocks),
            "",
            "## Question to answer now",
            question,
            "",
            "Answer directly as a perovskite expert. Do not discuss literature retrieval or data processing.",
            "",
        ]
    )
    # Some legacy prose joins an internal CORE id directly to punctuation or
    # another underscore, which can bypass token-boundary cleaning above.
    # Apply the same model-facing sanitation once more to the assembled packet.
    model_text = CORE_RE.sub("the focal paper", model_text)
    model_text = EVID_RE.sub("", model_text)
    if len(cards) != 40 or len(quick_lines) != 40:
        raise RuntimeError(f"{query_id}: Top40 coverage invariant failed")
    forbidden = ("expected_core_ids", "reference_expected_answer", "gold_answer")
    if any(term in model_text for term in forbidden):
        raise RuntimeError(f"{query_id}: private-field text leaked into model packet")
    internal_schema = re.compile(
        r"\[P\d{2}(?:-[ES]\d*)?\]|CORE_\d+|\b(?:block_id|result_block_id|"
        r"question_answered|core_finding|training_value|asset_value):",
        flags=re.IGNORECASE,
    )
    internal_match = internal_schema.search(model_text)
    if internal_match:
        raise RuntimeError(
            f"{query_id}: internal identifier leaked into model packet: "
            f"{internal_match.group(0)!r}"
        )
    packet = {
        "schema": "opensolar_compact40_stage2_evidence_packet_v1",
        "query_id": query_id,
        "benchmark_id": rank_row.get("benchmark_id"),
        "ability_family": ability_family,
        "question": question,
        "top40_source": "frozen public_top40_stage2_rankings.jsonl",
        "top40_ordering": rank_row.get("ordering"),
        "private_labels_read": False,
        "human_expert_annotation": False,
        "cards": cards,
        "model_input_chars": len(model_text),
        "estimated_model_input_tokens": math.ceil(len(model_text) / 4),
    }
    return packet, registry, model_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--source-database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-id", action="append", default=[])
    parser.add_argument("--sample-per-family", type=int, default=1)
    parser.add_argument("--query-limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    selected_ids, rank_rows = choose_query_rows(
        args.rankings,
        query_ids=list(args.query_id),
        sample_per_family=max(0, int(args.sample_per_family)),
        query_limit=args.query_limit,
    )
    pool_rows = load_selected_pool_rows(args.pool, selected_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = args.output_dir / "stage2_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    packets: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    store = EvidenceStore(args.source_database)
    try:
        for index, query_id in enumerate(selected_ids, start=1):
            packet, packet_registry, model_text = build_one_packet(
                rank_rows[query_id], pool_rows[query_id], store
            )
            packets.append(packet)
            registry.extend(packet_registry)
            atomic_write_text(input_dir / f"{query_id}.txt", model_text)
            science_count = sum(
                card["evidence_level"] == "science_rag_core"
                for card in packet["cards"]
            )
            print(
                f"[{index:02d}/{len(selected_ids):02d}] {query_id} "
                f"papers=40 science_rag={science_count} "
                f"chars={packet['model_input_chars']} "
                f"est_tokens={packet['estimated_model_input_tokens']}",
                flush=True,
            )
    finally:
        store.close()

    packet_path = args.output_dir / "evidence_packets.jsonl"
    registry_path = args.output_dir / "source_registry.jsonl"
    write_jsonl(packet_path, packets)
    write_jsonl(registry_path, registry)
    preview_path = args.output_dir / "sample_preview.md"
    first_text = (input_dir / f"{selected_ids[0]}.txt").read_text(encoding="utf-8")
    atomic_write_text(
        preview_path,
        "# Compact40 Stage2 evidence packet preview\n\n"
        + f"Query: `{selected_ids[0]}`\n\n"
        + first_text,
    )
    char_counts = [int(row["model_input_chars"]) for row in packets]
    token_estimates = [int(row["estimated_model_input_tokens"]) for row in packets]
    science_counts = [
        sum(card["evidence_level"] == "science_rag_core" for card in row["cards"])
        for row in packets
    ]
    manifest_path = args.output_dir / "manifest.json"
    output_files = [packet_path, registry_path, preview_path] + [
        input_dir / f"{query_id}.txt" for query_id in selected_ids
    ]
    manifest = {
        "schema": "opensolar_compact40_stage2_evidence_manifest_v1",
        "status": "READY",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(time.time() - started, 3),
        "policy": {
            "all_top40_represented": True,
            "science_rag_core_receives_deep_evidence": True,
            "quick_map_papers": 40,
            "deep_detail_cutoff": 40,
            "rank_1_8_summary_sentences_atomic_budget": [3, 3],
            "rank_9_20_summary_sentences_atomic_budget": [2, 2],
            "rank_21_40_summary_sentences_atomic_budget": [1, 1],
            "private_labels_read": False,
            "model_facing_core_ids": False,
            "model_facing_paper_or_evidence_labels": False,
            "model_calls_used_for_compilation": False,
        },
        "counts": {
            "queries": len(packets),
            "papers": sum(len(row["cards"]) for row in packets),
            "papers_per_query_min": min(len(row["cards"]) for row in packets),
            "papers_per_query_max": max(len(row["cards"]) for row in packets),
            "science_rag_papers_total": sum(science_counts),
            "registry_records": len(registry),
        },
        "model_input": {
            "chars_min": min(char_counts),
            "chars_median": statistics.median(char_counts),
            "chars_max": max(char_counts),
            "estimated_tokens_min": min(token_estimates),
            "estimated_tokens_median": statistics.median(token_estimates),
            "estimated_tokens_max": max(token_estimates),
        },
        "query_ids": selected_ids,
        "inputs": {
            "pool": str(args.pool),
            "rankings": str(args.rankings),
            "source_database": str(args.source_database),
        },
        "files": {
            str(path.relative_to(args.output_dir)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in output_files
        },
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
